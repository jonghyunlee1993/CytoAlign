"""Target-marker-specific diagonal metrics for panel-aware kNN imputation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


def _matrix(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _normalized_weights(logits, mask):
    import torch

    active = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
    if active.ndim != 1 or active.numel() != logits.shape[1] or not active.any():
        raise ValueError("panel mask does not align with the common markers")
    masked = logits.masked_fill(~active[None, :], -torch.inf)
    return torch.softmax(masked, dim=1)


@dataclass(frozen=True)
class MetricFit:
    mode: str
    temperature: float
    regularization: float
    weights: np.ndarray
    losses: tuple[float, ...]


class AdaptiveMetricLearner:
    """Learn one non-negative diagonal metric globally or per target marker."""

    def __init__(
        self,
        *,
        mode: str,
        temperature: float,
        regularization: float,
        epochs: int,
        steps_per_epoch: int,
        query_batch_size: int,
        reference_batch_size: int,
        learning_rate: float,
        random_state: int,
    ):
        if mode not in {"global", "target"}:
            raise ValueError("mode must be 'global' or 'target'")
        if float(temperature) <= 0:
            raise ValueError("temperature must be positive")
        for name, value in (
            ("epochs", epochs),
            ("steps_per_epoch", steps_per_epoch),
            ("query_batch_size", query_batch_size),
            ("reference_batch_size", reference_batch_size),
        ):
            if int(value) < 1:
                raise ValueError(f"{name} must be positive")
        self.mode = mode
        self.temperature = float(temperature)
        self.regularization = float(regularization)
        self.epochs = int(epochs)
        self.steps_per_epoch = int(steps_per_epoch)
        self.query_batch_size = int(query_batch_size)
        self.reference_batch_size = int(reference_batch_size)
        self.learning_rate = float(learning_rate)
        self.random_state = int(random_state)

    def fit(
        self,
        common: np.ndarray,
        targets: np.ndarray,
        patient_groups: Sequence,
        marker_scales: np.ndarray,
        *,
        panel_masks: Sequence[np.ndarray],
        mask_dropout: bool,
        device: str,
    ) -> MetricFit:
        import torch
        import torch.nn.functional as functional

        common = _matrix(common, "common")
        targets = _matrix(targets, "targets")
        groups = np.asarray(patient_groups).astype(str)
        scales = np.asarray(marker_scales, dtype=np.float32)
        if common.shape[0] != targets.shape[0] or groups.size != common.shape[0]:
            raise ValueError("training rows and patient groups do not align")
        if scales.shape != (targets.shape[1],) or np.any(scales <= 0):
            raise ValueError("marker scales do not align with targets")
        unique_groups = np.unique(groups)
        if unique_groups.size < 2:
            raise ValueError("at least two patients are required for metric learning")
        masks = [np.asarray(mask, dtype=bool) for mask in panel_masks]
        if not masks or any(mask.shape != (common.shape[1],) for mask in masks):
            raise ValueError("panel masks do not align with common markers")

        torch.manual_seed(self.random_state)
        if device == "cuda":
            torch.cuda.manual_seed_all(self.random_state)
        rng = np.random.RandomState(self.random_state)
        n_metrics = 1 if self.mode == "global" else targets.shape[1]
        logits = torch.nn.Parameter(
            torch.zeros((n_metrics, common.shape[1]), dtype=torch.float32, device=device)
        )
        optimizer = torch.optim.Adam([logits], lr=self.learning_rate)
        scale_tensor = torch.as_tensor(scales, device=device)
        rows_by_group = {
            group: np.flatnonzero(groups == group) for group in unique_groups
        }
        other_rows = {
            group: np.flatnonzero(groups != group) for group in unique_groups
        }
        full_mask = np.ones(common.shape[1], dtype=bool)
        losses = []

        for _ in range(self.epochs):
            epoch_loss = 0.0
            for _ in range(self.steps_per_epoch):
                group = str(rng.choice(unique_groups))
                query_pool = rows_by_group[group]
                reference_pool = other_rows[group]
                query_rows = rng.choice(
                    query_pool,
                    self.query_batch_size,
                    replace=query_pool.size < self.query_batch_size,
                )
                reference_rows = rng.choice(
                    reference_pool,
                    self.reference_batch_size,
                    replace=reference_pool.size < self.reference_batch_size,
                )
                mask = masks[rng.randint(len(masks))] if mask_dropout else full_mask
                weights = _normalized_weights(logits, mask)
                query = torch.as_tensor(common[query_rows], device=device)
                reference = torch.as_tensor(common[reference_rows], device=device)
                reference_targets = torch.as_tensor(targets[reference_rows], device=device)
                query_targets = torch.as_tensor(targets[query_rows], device=device)
                squared_difference = (query[:, None, :] - reference[None, :, :]).square()

                if self.mode == "global":
                    distance = torch.einsum("brd,d->br", squared_difference, weights[0])
                    probability = torch.softmax(
                        -distance / self.temperature, dim=1
                    )
                    prediction = probability @ reference_targets
                else:
                    distance = torch.einsum("brd,td->brt", squared_difference, weights)
                    probability = torch.softmax(
                        -distance / self.temperature, dim=1
                    )
                    prediction = torch.einsum(
                        "brt,rt->bt", probability, reference_targets
                    )

                normalized_error = (prediction - query_targets) / scale_tensor
                data_loss = functional.smooth_l1_loss(
                    normalized_error, torch.zeros_like(normalized_error)
                )
                uniform = 1.0 / float(mask.sum())
                active_weights = weights[:, torch.as_tensor(mask, device=device)]
                penalty = ((active_weights - uniform) ** 2).mean() * mask.sum()
                loss = data_loss + self.regularization * penalty
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_loss += float(data_loss.detach())
            losses.append(epoch_loss / self.steps_per_epoch)

        weights = torch.softmax(logits, dim=1).detach().cpu().numpy()
        if self.mode == "global":
            weights = np.repeat(weights, targets.shape[1], axis=0)
        return MetricFit(
            mode=self.mode,
            temperature=self.temperature,
            regularization=self.regularization,
            weights=weights.astype(np.float32),
            losses=tuple(losses),
        )


def _median(values):
    """Match NumPy's even-sample median for tensors shaped [batch, k, marker]."""

    ordered, _ = values.sort(dim=1)
    middle = ordered.shape[1] // 2
    if ordered.shape[1] % 2:
        return ordered[:, middle]
    return 0.5 * (ordered[:, middle - 1] + ordered[:, middle])


def predict_with_candidate_reranking(
    reference_common: np.ndarray,
    reference_targets: np.ndarray,
    query_common: np.ndarray,
    metric_weights: Mapping[str, np.ndarray],
    panel_mask: np.ndarray,
    *,
    k: int,
    candidate_k: int,
    batch_size: int,
    device: str,
) -> dict[str, np.ndarray]:
    """Exact plain kNN plus adaptive reranking inside its Euclidean candidate set."""

    import torch

    reference_common = _matrix(reference_common, "reference_common")
    reference_targets = _matrix(reference_targets, "reference_targets")
    query_common = _matrix(query_common, "query_common")
    mask = np.asarray(panel_mask, dtype=bool)
    if (
        reference_common.shape[0] != reference_targets.shape[0]
        or query_common.shape[1] != reference_common.shape[1]
        or mask.shape != (reference_common.shape[1],)
        or not mask.any()
    ):
        raise ValueError("reference, query, target, and panel dimensions do not align")
    if int(k) < 1 or int(candidate_k) < int(k):
        raise ValueError("candidate_k must be at least k")
    candidate_k = min(int(candidate_k), reference_common.shape[0])
    k = min(int(k), candidate_k)

    reference = torch.as_tensor(reference_common[:, mask], device=device)
    targets = torch.as_tensor(reference_targets, device=device)
    reference_norm = reference.square().sum(dim=1)
    normalized_weights = {}
    for name, value in metric_weights.items():
        weights = np.asarray(value, dtype=np.float32)
        if weights.shape != (
            reference_targets.shape[1],
            reference_common.shape[1],
        ):
            raise ValueError(f"weights for {name!r} have an unexpected shape")
        weights = weights[:, mask]
        weights /= weights.sum(axis=1, keepdims=True)
        normalized_weights[name] = torch.as_tensor(weights, device=device)

    output = {
        "plain_knn": np.empty(
            (query_common.shape[0], reference_targets.shape[1]), dtype=np.float32
        ),
        **{
            name: np.empty(
                (query_common.shape[0], reference_targets.shape[1]), dtype=np.float32
            )
            for name in metric_weights
        },
    }
    for left in range(0, query_common.shape[0], int(batch_size)):
        right = min(query_common.shape[0], left + int(batch_size))
        query = torch.as_tensor(query_common[left:right, mask], device=device)
        distance = (
            query.square().sum(dim=1, keepdim=True)
            + reference_norm[None, :]
            - 2.0 * query @ reference.T
        )
        candidates = torch.topk(
            distance, candidate_k, dim=1, largest=False, sorted=True
        ).indices
        candidate_targets = targets[candidates]
        output["plain_knn"][left:right] = (
            _median(candidate_targets[:, :k]).cpu().numpy()
        )
        squared_difference = (
            query[:, None, :] - reference[candidates]
        ).square()
        for name, weights in normalized_weights.items():
            adaptive_distance = torch.einsum(
                "bcd,td->bct", squared_difference, weights
            )
            neighbors = torch.topk(
                adaptive_distance, k, dim=1, largest=False, sorted=False
            ).indices
            neighbor_targets = torch.gather(
                candidate_targets, 1, neighbors
            )
            output[name][left:right] = _median(neighbor_targets).cpu().numpy()
    return output
