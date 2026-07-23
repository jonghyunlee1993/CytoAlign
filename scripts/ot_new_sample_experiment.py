#!/usr/bin/env python3
"""Train-only OT distillation and target-free inference on new specimens."""

from __future__ import annotations

import argparse
import copy
import json
import random
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pseudopanel_experiment import (  # noqa: E402
    PreparedBlock,
    indices,
    load_cache,
    marker_scale,
    prepare_blocks,
    prepare_pooled_blocks,
    sample_blocks,
)
from src.evaluation.exact_metrics import (  # noqa: E402
    evaluate_exact_cells,
    select_residual_alpha,
)
from src.evaluation.population_metrics import (  # noqa: E402
    evaluate_matched_populations,
)
from src.matching.optimal_transport import (  # noqa: E402
    balanced_sinkhorn,
    barycentric_projection,
    coupling_diagnostics,
    squared_euclidean_cost,
)
from src.models.h_only import HOnlyRegressor  # noqa: E402
from src.preprocessing.common_space import EmpiricalPercentileTransformer  # noqa: E402
from src.data.splits import patient_id_from_specimen  # noqa: E402


@dataclass(frozen=True)
class TeacherData:
    h: np.ndarray
    x: np.ndarray
    labels: np.ndarray
    hidden_y: np.ndarray
    ot_y: np.ndarray
    uniform_y: np.ndarray
    diagnostics: dict


class TorchResidualRegressor:
    """Small deterministic MLP with an internal train-label holdout."""

    def __init__(
        self,
        *,
        hidden_dims: Sequence[int] = (128, 128),
        epochs: int = 30,
        batch_size: int = 1024,
        learning_rate: float = 1.0e-3,
        patience: int = 5,
        random_state: int = 42,
    ):
        self.hidden_dims = tuple(map(int, hidden_dims))
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.patience = int(patience)
        self.random_state = int(random_state)

    def fit(self, features: np.ndarray, target: np.ndarray, device):
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        values = np.asarray(features, dtype=np.float32)
        labels = np.asarray(target, dtype=np.float32)
        if values.ndim != 2 or labels.ndim != 2:
            raise ValueError("MLP features and target must be matrices")
        if values.shape[0] != labels.shape[0] or values.shape[0] < 10:
            raise ValueError("MLP features and target do not align")
        self.feature_mean_ = values.mean(axis=0, dtype=np.float64).astype(
            np.float32
        )
        self.feature_scale_ = values.std(axis=0, dtype=np.float64).astype(
            np.float32
        )
        self.feature_scale_[self.feature_scale_ < 1.0e-6] = 1.0
        self.target_mean_ = labels.mean(axis=0, dtype=np.float64).astype(
            np.float32
        )
        self.target_scale_ = labels.std(axis=0, dtype=np.float64).astype(
            np.float32
        )
        self.target_scale_[self.target_scale_ < 1.0e-6] = 1.0
        scaled_x = (values - self.feature_mean_) / self.feature_scale_
        scaled_y = (labels - self.target_mean_) / self.target_scale_
        rng = np.random.RandomState(self.random_state)
        order = rng.permutation(values.shape[0])
        n_validation = max(1, int(round(0.1 * values.shape[0])))
        validation_rows = order[:n_validation]
        training_rows = order[n_validation:]
        if training_rows.size < 2:
            raise ValueError("Too few MLP training rows after holdout")
        generator = torch.Generator()
        generator.manual_seed(self.random_state)
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(scaled_x[training_rows]),
                torch.from_numpy(scaled_y[training_rows]),
            ),
            batch_size=min(self.batch_size, training_rows.size),
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        layers = []
        width = values.shape[1]
        for hidden in self.hidden_dims:
            layers.extend((nn.Linear(width, hidden), nn.GELU()))
            width = hidden
        layers.append(nn.Linear(width, labels.shape[1]))
        self.model_ = nn.Sequential(*layers).to(device)
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=1.0e-5,
        )
        criterion = nn.SmoothL1Loss()
        validation_x = torch.from_numpy(scaled_x[validation_rows]).to(device)
        validation_y = torch.from_numpy(scaled_y[validation_rows]).to(device)
        history = []
        best_loss = float("inf")
        best_state = None
        stale = 0
        for epoch in range(self.epochs):
            self.model_.train()
            train_loss = 0.0
            train_rows = 0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.model_(batch_x), batch_y)
                loss.backward()
                optimizer.step()
                train_loss += float(loss.detach()) * batch_x.shape[0]
                train_rows += batch_x.shape[0]
            self.model_.eval()
            with torch.no_grad():
                validation_loss = float(
                    criterion(self.model_(validation_x), validation_y)
                )
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_teacher_loss": train_loss / train_rows,
                    "validation_teacher_loss": validation_loss,
                }
            )
            if validation_loss < best_loss - 1.0e-5:
                best_loss = validation_loss
                best_state = copy.deepcopy(self.model_.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is None:
            raise RuntimeError("MLP did not produce a finite checkpoint")
        self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.history_ = history
        self.best_validation_teacher_loss_ = best_loss
        return self

    def predict(
        self, features: np.ndarray, device, *, batch_size: int = 8192
    ) -> np.ndarray:
        import torch

        if not hasattr(self, "model_"):
            raise RuntimeError("MLP has not been fitted")
        values = np.asarray(features, dtype=np.float32)
        scaled = (values - self.feature_mean_) / self.feature_scale_
        pieces = []
        with torch.no_grad():
            for start in range(0, scaled.shape[0], int(batch_size)):
                current = torch.from_numpy(
                    scaled[start : start + int(batch_size)]
                ).to(device)
                pieces.append(self.model_(current).cpu().numpy())
        result = np.concatenate(pieces)
        return (
            result * self.target_scale_[None, :]
            + self.target_mean_[None, :]
        ).astype(np.float32)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--k-max", type=int, default=256)
    parser.add_argument("--k-min", type=int, default=32)
    parser.add_argument("--paired-epsilon-ratio", type=float, default=0.1)
    parser.add_argument("--pooled-epsilon-ratio", type=float, default=0.2)
    parser.add_argument("--sinkhorn-iterations", type=int, default=300)
    parser.add_argument("--mlp-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=(0.0, 0.1, 0.25, 0.5, 1.0)
    )
    parser.add_argument("--seed", type=int, default=4207)
    return parser.parse_args()


def _concatenate_blocks(
    panels: Mapping[str, dict],
    blocks,
    column_index: np.ndarray,
    side: str,
) -> np.ndarray:
    if side not in {"source", "target"}:
        raise ValueError("side must be source or target")
    name = f"{side}_indices"
    return np.concatenate(
        [
            panels[block.specimen]["values"][:, column_index][
                getattr(block, name)
            ]
            for block in blocks
        ]
    )


def _block_labels(blocks, side: str) -> np.ndarray:
    name = f"{side}_indices"
    return np.concatenate(
        [
            np.repeat(block.cell_type, getattr(block, name).size)
            for block in blocks
        ]
    ).astype(str)


def build_teacher(
    blocks: Sequence[PreparedBlock],
    marker_scales: np.ndarray,
    device,
    *,
    epsilon_ratio: float,
    sinkhorn_iterations: int,
) -> TeacherData:
    import torch

    h_values = []
    x_values = []
    labels = []
    hidden_y = []
    ot_y = []
    uniform_y = []
    entropies = []
    effective_targets = []
    top_probabilities = []
    row_errors = []
    column_errors = []
    expected_pair_mae = []
    uniform_pair_mae = []
    scale = torch.as_tensor(marker_scales, device=device)[None, None, :]
    for block in blocks:
        source_h = torch.as_tensor(block.source_h, device=device)
        target_h = torch.as_tensor(block.target_h, device=device)
        source_y = torch.as_tensor(block.source_y, device=device)
        target_y = torch.as_tensor(block.target_y, device=device)
        cost = squared_euclidean_cost(source_h, target_h)
        positive = cost[cost > 0]
        median_cost = (
            torch.median(positive)
            if positive.numel()
            else cost.new_tensor(1.0)
        )
        epsilon = max(float(epsilon_ratio) * float(median_cost), 1.0e-6)
        plan = balanced_sinkhorn(
            cost, epsilon=epsilon, iterations=sinkhorn_iterations
        )
        current_ot = barycentric_projection(plan, target_y)
        current_uniform = target_y.mean(dim=0, keepdim=True).expand(
            source_y.shape[0], -1
        )
        pair_error = (
            torch.abs(source_y[:, None, :] - target_y[None, :, :]) / scale
        ).mean(dim=2)
        expected_pair_mae.append(float((plan * pair_error).sum() / plan.sum()))
        uniform_pair_mae.append(float(pair_error.mean()))
        diagnostic = coupling_diagnostics(plan)
        entropies.append(diagnostic["normalized_entropy_mean"])
        effective_targets.append(diagnostic["effective_targets_mean"])
        top_probabilities.append(diagnostic["top_probability_mean"])
        row_errors.append(diagnostic["row_marginal_max_error"])
        column_errors.append(diagnostic["column_marginal_max_error"])
        h_values.append(block.source_h)
        x_values.append(block.source_x)
        hidden_y.append(block.source_y)
        labels.append(np.repeat(block.cell_type, block.source_y.shape[0]))
        ot_y.append(current_ot.cpu().numpy())
        uniform_y.append(current_uniform.cpu().numpy())
    hidden = np.concatenate(hidden_y)
    ot_target = np.concatenate(ot_y)
    uniform_target = np.concatenate(uniform_y)
    diagnostics = {
        "n_blocks": len(blocks),
        "n_source_cells": int(hidden.shape[0]),
        "epsilon_ratio": float(epsilon_ratio),
        "teacher_exact_normalized_mae": float(
            np.mean(np.abs(ot_target - hidden) / marker_scales[None, :])
        ),
        "uniform_teacher_exact_normalized_mae": float(
            np.mean(np.abs(uniform_target - hidden) / marker_scales[None, :])
        ),
        "expected_pair_mae_mean_over_blocks": float(
            np.mean(expected_pair_mae)
        ),
        "uniform_pair_mae_mean_over_blocks": float(
            np.mean(uniform_pair_mae)
        ),
        "normalized_entropy_mean": float(np.mean(entropies)),
        "effective_targets_mean": float(np.mean(effective_targets)),
        "top_probability_mean": float(np.mean(top_probabilities)),
        "row_marginal_max_error": float(np.max(row_errors)),
        "column_marginal_max_error": float(np.max(column_errors)),
    }
    return TeacherData(
        h=np.concatenate(h_values).astype(np.float32),
        x=np.concatenate(x_values).astype(np.float32),
        labels=np.concatenate(labels).astype(str),
        hidden_y=hidden.astype(np.float32),
        ot_y=ot_target.astype(np.float32),
        uniform_y=uniform_target.astype(np.float32),
        diagnostics=diagnostics,
    )


def split_arrays(
    panels: Mapping[str, dict],
    specimens: Sequence[str],
    transformer: EmpiricalPercentileTransformer,
    h_index: np.ndarray,
    x_index: np.ndarray,
    y_index: np.ndarray,
) -> dict:
    h = {}
    x = {}
    y = {}
    labels = {}
    for specimen in specimens:
        values = panels[specimen]["values"]
        h[specimen] = transformer.transform(values[:, h_index])
        x[specimen] = np.asarray(values[:, x_index], dtype=np.float32)
        y[specimen] = np.asarray(values[:, y_index], dtype=np.float32)
        labels[specimen] = np.asarray(panels[specimen]["labels"]).astype(str)
    return {
        "h": h,
        "x": x,
        "y": y,
        "labels": labels,
        "patients": {
            specimen: patient_id_from_specimen(specimen)
            for specimen in specimens
        },
    }


def encoded_features(
    h: np.ndarray,
    x: np.ndarray | None,
    labels: np.ndarray,
    classes: Sequence[str],
) -> np.ndarray:
    lookup = {label: index for index, label in enumerate(classes)}
    one_hot = np.zeros((labels.size, len(classes)), dtype=np.float32)
    for row, label in enumerate(labels):
        if label not in lookup:
            raise ValueError(f"Unknown cell type at inference: {label}")
        one_hot[row, lookup[label]] = 1.0
    pieces = [np.asarray(h, dtype=np.float32)]
    if x is not None:
        pieces.append(np.asarray(x, dtype=np.float32))
    pieces.append(one_hot)
    return np.concatenate(pieces, axis=1)


def predict_baseline(model, split: dict) -> dict[str, np.ndarray]:
    return {
        specimen: model.predict(
            split["h"][specimen], cell_types=split["labels"][specimen]
        )
        for specimen in split["h"]
    }


def evaluate(
    prediction: Mapping[str, np.ndarray],
    split: dict,
    marker_scales: np.ndarray,
) -> dict:
    return {
        "exact_cell": evaluate_exact_cells(
            prediction, split["y"], split["patients"], marker_scales
        ),
        "population": evaluate_matched_populations(
            prediction,
            split["labels"],
            split["y"],
            split["labels"],
            split["patients"],
            marker_scales,
            minimum_cells=5,
        ),
    }


def main() -> None:
    import torch

    args = arguments()
    started = time.time()
    if args.k_min < 2 or args.k_max < args.k_min:
        raise ValueError("Require 2 <= k_min <= k_max")
    if (
        min(args.paired_epsilon_ratio, args.pooled_epsilon_ratio) <= 0
        or args.sinkhorn_iterations < 1
        or args.mlp_epochs < 1
        or args.batch_size < 1
    ):
        raise ValueError("OT and MLP hyperparameters must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    if device.type == "cuda":
        probe = torch.randn(1024, 1024, device=device) @ torch.randn(
            1024, 1024, device=device
        )
        torch.cuda.synchronize(device)
        if not torch.isfinite(probe).all():
            raise RuntimeError("CUDA matrix multiplication failed")

    manifest, panels = load_cache(args.cache)
    pseudo = manifest["pseudo_panel"]
    markers = tuple(pseudo["full_markers"])
    h_index = indices(markers, pseudo["common_markers"])
    x_index = indices(markers, pseudo["source_exclusive_markers"])
    y_index = indices(markers, pseudo["target_exclusive_markers"])
    fold = manifest["split_manifest"]["folds"][0]
    train_specimens = tuple(sorted(fold["train_specimens"]))
    validation_specimens = tuple(sorted(fold["validation_specimens"]))
    test_specimens = tuple(sorted(fold["test_specimens"]))

    train_blocks = sample_blocks(
        panels,
        train_specimens,
        k_max=args.k_max,
        k_min=args.k_min,
        seed=args.seed + 10,
    )
    source_h_raw = _concatenate_blocks(
        panels, train_blocks, h_index, "source"
    )
    target_h_raw = _concatenate_blocks(
        panels, train_blocks, h_index, "target"
    )
    target_y = _concatenate_blocks(panels, train_blocks, y_index, "target")
    target_labels = _block_labels(train_blocks, "target")
    transformer = EmpiricalPercentileTransformer().fit(
        np.concatenate([source_h_raw, target_h_raw])
    )
    target_h = transformer.transform(target_h_raw)
    scales = marker_scale(target_y)
    baseline_h = HOnlyRegressor(architecture="ridge").fit(target_h, target_y)
    baseline_hl = HOnlyRegressor(
        architecture="ridge", condition_on_cell_type=True
    ).fit(target_h, target_y, cell_types=target_labels)

    paired_blocks = prepare_blocks(
        train_blocks,
        np.arange(len(train_blocks)),
        panels,
        transformer,
        h_index,
        x_index,
        y_index,
    )
    pooled_blocks = prepare_pooled_blocks(
        train_blocks,
        panels,
        transformer,
        h_index,
        x_index,
        y_index,
        seed=args.seed + 20,
    )
    paired_teacher = build_teacher(
        paired_blocks,
        scales,
        device,
        epsilon_ratio=args.paired_epsilon_ratio,
        sinkhorn_iterations=args.sinkhorn_iterations,
    )
    pooled_teacher = build_teacher(
        pooled_blocks,
        scales,
        device,
        epsilon_ratio=args.pooled_epsilon_ratio,
        sinkhorn_iterations=args.sinkhorn_iterations,
    )
    if (
        not np.allclose(paired_teacher.h, pooled_teacher.h)
        or not np.allclose(paired_teacher.x, pooled_teacher.x)
        or not np.array_equal(paired_teacher.labels, pooled_teacher.labels)
    ):
        raise RuntimeError("Paired and pooled teachers do not share source rows")

    validation = split_arrays(
        panels,
        validation_specimens,
        transformer,
        h_index,
        x_index,
        y_index,
    )
    test = split_arrays(
        panels,
        test_specimens,
        transformer,
        h_index,
        x_index,
        y_index,
    )
    validation_baseline_h = {
        specimen: baseline_h.predict(validation["h"][specimen])
        for specimen in validation["h"]
    }
    test_baseline_h = {
        specimen: baseline_h.predict(test["h"][specimen])
        for specimen in test["h"]
    }
    validation_baseline_hl = predict_baseline(baseline_hl, validation)
    test_baseline_hl = predict_baseline(baseline_hl, test)
    result = {
        "baseline_h": {
            "validation": evaluate(
                validation_baseline_h, validation, scales
            ),
            "test": evaluate(test_baseline_h, test, scales),
        },
        "baseline_hl": {
            "validation": evaluate(
                validation_baseline_hl, validation, scales
            ),
            "test": evaluate(test_baseline_hl, test, scales),
        },
    }

    classes = tuple(sorted(set(target_labels)))
    teacher_targets = {
        "paired_ot": paired_teacher.ot_y,
        "pooled_ot": pooled_teacher.ot_y,
        "pooled_uniform": pooled_teacher.uniform_y,
    }
    train_base = baseline_hl.predict(
        paired_teacher.h, cell_types=paired_teacher.labels
    )
    train_features = {
        "hl": paired_teacher.h,
        "hxl": np.concatenate(
            [paired_teacher.h, paired_teacher.x], axis=1
        ),
    }
    mlp_train_features = {
        "hl": encoded_features(
            paired_teacher.h, None, paired_teacher.labels, classes
        ),
        "hxl": encoded_features(
            paired_teacher.h,
            paired_teacher.x,
            paired_teacher.labels,
            classes,
        ),
    }

    for teacher_name, teacher_y in teacher_targets.items():
        residual_target = teacher_y - train_base
        for feature_name, values in train_features.items():
            method = f"ridge_{teacher_name}_{feature_name}"
            ridge = HOnlyRegressor(
                architecture="ridge", condition_on_cell_type=True
            ).fit(
                values,
                residual_target,
                cell_types=paired_teacher.labels,
            )
            validation_residual = {
                specimen: ridge.predict(
                    (
                        validation["h"][specimen]
                        if feature_name == "hl"
                        else np.concatenate(
                            [
                                validation["h"][specimen],
                                validation["x"][specimen],
                            ],
                            axis=1,
                        )
                    ),
                    cell_types=validation["labels"][specimen],
                )
                for specimen in validation["h"]
            }
            selected_alpha, validation_curve = select_residual_alpha(
                validation_baseline_hl,
                validation_residual,
                validation["y"],
                validation["patients"],
                scales,
                args.alphas,
            )
            test_residual = {
                specimen: ridge.predict(
                    (
                        test["h"][specimen]
                        if feature_name == "hl"
                        else np.concatenate(
                            [test["h"][specimen], test["x"][specimen]],
                            axis=1,
                        )
                    ),
                    cell_types=test["labels"][specimen],
                )
                for specimen in test["h"]
            }
            validation_prediction = {
                specimen: validation_baseline_hl[specimen]
                + selected_alpha * validation_residual[specimen]
                for specimen in validation["h"]
            }
            test_prediction = {
                specimen: test_baseline_hl[specimen]
                + selected_alpha * test_residual[specimen]
                for specimen in test["h"]
            }
            result[method] = {
                "selected_alpha": selected_alpha,
                "validation_alpha_curve": validation_curve,
                "validation": evaluate(
                    validation_prediction, validation, scales
                ),
                "test": evaluate(test_prediction, test, scales),
            }

        for feature_name, values in mlp_train_features.items():
            method = f"mlp_{teacher_name}_{feature_name}"
            mlp = TorchResidualRegressor(
                epochs=args.mlp_epochs,
                batch_size=args.batch_size,
                random_state=args.seed
                + len(result) * 17
                + (0 if feature_name == "hl" else 1),
            ).fit(values, residual_target, device)
            validation_residual = {}
            test_residual = {}
            for specimen in validation["h"]:
                features = encoded_features(
                    validation["h"][specimen],
                    (
                        None
                        if feature_name == "hl"
                        else validation["x"][specimen]
                    ),
                    validation["labels"][specimen],
                    classes,
                )
                validation_residual[specimen] = mlp.predict(features, device)
            selected_alpha, validation_curve = select_residual_alpha(
                validation_baseline_hl,
                validation_residual,
                validation["y"],
                validation["patients"],
                scales,
                args.alphas,
            )
            for specimen in test["h"]:
                features = encoded_features(
                    test["h"][specimen],
                    None if feature_name == "hl" else test["x"][specimen],
                    test["labels"][specimen],
                    classes,
                )
                test_residual[specimen] = mlp.predict(features, device)
            validation_prediction = {
                specimen: validation_baseline_hl[specimen]
                + selected_alpha * validation_residual[specimen]
                for specimen in validation["h"]
            }
            test_prediction = {
                specimen: test_baseline_hl[specimen]
                + selected_alpha * test_residual[specimen]
                for specimen in test["h"]
            }
            result[method] = {
                "selected_alpha": selected_alpha,
                "validation_alpha_curve": validation_curve,
                "training": {
                    "epochs_completed": len(mlp.history_),
                    "best_validation_teacher_loss": (
                        mlp.best_validation_teacher_loss_
                    ),
                    "first": mlp.history_[0],
                    "last": mlp.history_[-1],
                },
                "validation": evaluate(
                    validation_prediction, validation, scales
                ),
                "test": evaluate(test_prediction, test, scales),
            }
            if device.type == "cuda":
                torch.cuda.empty_cache()

    artifact = {
        "status": "ok",
        "host": socket.gethostname(),
        "cuda_device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "cuda_visible_memory_bytes": (
            int(torch.cuda.get_device_properties(device).total_memory)
            if device.type == "cuda"
            else None
        ),
        "seed": args.seed,
        "fold": 0,
        "sampling": manifest["sampling"],
        "contract": {
            "teacher_generation_split": "train_only",
            "validation_target_reference_used_for_inference": False,
            "test_target_reference_used_for_inference": False,
            "validation_and_test_specimens_unseen_during_fit": True,
            "ot_cost_features": list(pseudo["common_markers"]),
            "exclusive_markers_used_in_ot_cost": False,
            "source_target_train_cells_disjoint": True,
            "pooled_unpaired_target_excludes_source_specimen": True,
            "coarse_cell_type_used_by_teacher_and_conditioned_models": True,
            "hidden_source_y_used_for_training": False,
            "hidden_source_y_used_for_teacher_diagnostic_only": True,
            "alpha_selected_on_validation_exact_truth": True,
            "test_used_for_model_or_alpha_selection": False,
        },
        "hyperparameters": {
            "k_max": args.k_max,
            "k_min": args.k_min,
            "paired_epsilon_ratio_frozen_from_prior_pilot": (
                args.paired_epsilon_ratio
            ),
            "pooled_epsilon_ratio_frozen_from_prior_pilot": (
                args.pooled_epsilon_ratio
            ),
            "sinkhorn_iterations": args.sinkhorn_iterations,
            "mlp_epochs_maximum": args.mlp_epochs,
            "batch_size": args.batch_size,
            "alphas": list(args.alphas),
        },
        "train": {
            "n_blocks": len(train_blocks),
            "n_source_teacher_cells": int(paired_teacher.h.shape[0]),
            "n_target_baseline_cells": int(target_y.shape[0]),
            "paired_teacher": paired_teacher.diagnostics,
            "pooled_unpaired_teacher": pooled_teacher.diagnostics,
        },
        "methods": result,
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    summary = {
        name: {
            "alpha": values.get("selected_alpha"),
            "validation_mae": values["validation"]["exact_cell"][
                "patient_first_normalized_mae"
            ],
            "test_mae": values["test"]["exact_cell"][
                "patient_first_normalized_mae"
            ],
        }
        for name, values in result.items()
    }
    print(
        json.dumps(
            {
                "event": "complete",
                "output": str(args.output),
                "elapsed_seconds": artifact["elapsed_seconds"],
                "methods": summary,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
