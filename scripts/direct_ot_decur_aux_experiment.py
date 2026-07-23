#!/usr/bin/env python3
"""Direct paired-OT predictor with a training-only DeCUR projection head."""

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

from scripts.ot_new_sample_experiment import (  # noqa: E402
    _block_labels,
    _concatenate_blocks,
    build_teacher,
    encoded_features,
    evaluate,
    predict_baseline,
    split_arrays,
)
from src.data.pseudopanel_experiment import (  # noqa: E402
    indices,
    load_cache,
    marker_scale,
    prepare_blocks,
    sample_blocks,
)
from src.evaluation.exact_metrics import select_residual_alpha  # noqa: E402
from src.losses.decur import (  # noqa: E402
    weighted_decur_loss,
    weighted_normalized_cross_correlation,
)
from src.matching.optimal_transport import (  # noqa: E402
    balanced_sinkhorn,
    squared_euclidean_cost,
)
from src.models.h_only import HOnlyRegressor  # noqa: E402
from src.preprocessing.common_space import (  # noqa: E402
    EmpiricalPercentileTransformer,
)


@dataclass(frozen=True)
class AuxiliaryBlock:
    source_start: int
    source_stop: int
    target_start: int
    target_stop: int
    plan: np.ndarray


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--direct-reference", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--k-max", type=int, default=256)
    parser.add_argument("--k-min", type=int, default=32)
    parser.add_argument("--epsilon-ratio", type=float, default=0.1)
    parser.add_argument("--sinkhorn-iterations", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--projection-dim", type=int, default=32)
    parser.add_argument("--n-common", type=int, default=16)
    parser.add_argument("--decur-weight", type=float, default=0.05)
    parser.add_argument("--aux-blocks-per-step", type=int, default=8)
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=(0.0, 0.1, 0.25, 0.5, 1.0)
    )
    parser.add_argument("--seed", type=int, default=4207)
    return parser.parse_args()


def _one_hot(
    labels: np.ndarray,
    classes: Sequence[str],
) -> np.ndarray:
    lookup = {label: index for index, label in enumerate(classes)}
    result = np.zeros((labels.size, len(classes)), dtype=np.float32)
    for row, label in enumerate(labels):
        if label not in lookup:
            raise ValueError(f"Unknown cell type: {label}")
        result[row, lookup[label]] = 1.0
    return result


def build_auxiliary_data(
    prepared,
    baseline: HOnlyRegressor,
    classes: Sequence[str],
    device,
    *,
    epsilon_ratio: float,
    sinkhorn_iterations: int,
) -> tuple[np.ndarray, tuple[AuxiliaryBlock, ...], dict]:
    """Build target training views and retain the common-marker soft plans."""

    import torch

    target_views = []
    blocks = []
    source_offset = 0
    target_offset = 0
    diagonal_pair_costs = []
    uniform_pair_costs = []
    for block in prepared:
        source_h = torch.as_tensor(block.source_h, device=device)
        target_h = torch.as_tensor(block.target_h, device=device)
        cost = squared_euclidean_cost(source_h, target_h)
        positive = cost[cost > 0]
        median = (
            torch.median(positive)
            if positive.numel()
            else cost.new_tensor(1.0)
        )
        epsilon = max(float(epsilon_ratio) * float(median), 1.0e-6)
        plan = balanced_sinkhorn(
            cost,
            epsilon=epsilon,
            iterations=int(sinkhorn_iterations),
        )
        plan_array = plan.detach().cpu().numpy().astype(np.float32)
        labels = np.repeat(block.cell_type, block.target_h.shape[0]).astype(str)
        target_baseline = baseline.predict(
            block.target_h,
            cell_types=labels,
        )
        target_residual = block.target_y - target_baseline
        target_view = np.concatenate(
            [
                np.asarray(block.target_h, dtype=np.float32),
                np.asarray(target_residual, dtype=np.float32),
                _one_hot(labels, classes),
            ],
            axis=1,
        )
        target_views.append(target_view)
        source_stop = source_offset + block.source_h.shape[0]
        target_stop = target_offset + block.target_h.shape[0]
        blocks.append(
            AuxiliaryBlock(
                source_start=source_offset,
                source_stop=source_stop,
                target_start=target_offset,
                target_stop=target_stop,
                plan=plan_array,
            )
        )
        diagonal_pair_costs.append(float((plan * cost).sum() / plan.sum()))
        uniform_pair_costs.append(float(cost.mean()))
        source_offset = source_stop
        target_offset = target_stop
    return (
        np.concatenate(target_views).astype(np.float32),
        tuple(blocks),
        {
            "n_blocks": len(blocks),
            "n_source_cells": int(source_offset),
            "n_target_cells": int(target_offset),
            "ot_common_cost_mean": float(np.mean(diagonal_pair_costs)),
            "uniform_common_cost_mean": float(np.mean(uniform_pair_costs)),
        },
    )


def _standardization(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1.0e-6] = 1.0
    return mean, scale


def _uniform_like(plan: np.ndarray) -> np.ndarray:
    return np.full(plan.shape, 1.0 / float(plan.size), dtype=np.float32)


class DirectPredictor:
    """Two-layer residual MLP with an auxiliary-only projection pathway."""

    def __init__(
        self,
        input_dim: int,
        target_input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int,
        projection_dim: int,
    ):
        import torch
        from torch import nn

        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                )
                self.decoder = nn.Linear(hidden_dim, output_dim)
                self.source_projection = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, projection_dim),
                )
                self.target_encoder = nn.Sequential(
                    nn.Linear(target_input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                )
                self.target_projection = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, projection_dim),
                )

            def predict(self, values):
                return self.decoder(self.backbone(values))

            def source_concepts(self, values):
                return self.source_projection(self.backbone(values))

            def target_concepts(self, values):
                return self.target_projection(self.target_encoder(values))

        self.network = Network()


def correlation_diagnostics(
    model,
    source_values,
    target_values,
    blocks: Sequence[AuxiliaryBlock],
    train_mask,
    device,
    *,
    n_common: int,
    plan_kind: str,
) -> dict:
    import torch

    model.eval()
    correlations = []
    with torch.no_grad():
        for block in blocks:
            local_mask = train_mask[block.source_start : block.source_stop]
            if int(local_mask.sum()) < 2:
                continue
            device_mask = torch.as_tensor(
                local_mask,
                device=source_values.device,
            )
            source = source_values[block.source_start : block.source_stop][
                device_mask
            ]
            target = target_values[block.target_start : block.target_stop]
            plan = block.plan[local_mask]
            if plan_kind == "uniform":
                plan = _uniform_like(plan)
            correlation = weighted_normalized_cross_correlation(
                torch.as_tensor(source, device=device),
                torch.as_tensor(target, device=device),
                torch.as_tensor(plan, device=device),
            )
            correlations.append(correlation.cpu().numpy())
    matrix = np.mean(correlations, axis=0)
    common = matrix[:n_common, :n_common]
    common_off = common[~np.eye(n_common, dtype=bool)]
    unique = matrix[n_common:, n_common:]
    return {
        "n_blocks": len(correlations),
        "common_diagonal_mean": float(np.diag(common).mean()),
        "common_diagonal_minimum": float(np.diag(common).min()),
        "common_off_diagonal_absolute_mean": float(np.abs(common_off).mean()),
        "unique_absolute_mean": float(np.abs(unique).mean()),
    }


def train_variant(
    base_model,
    source_features: np.ndarray,
    target_features: np.ndarray,
    residual_target: np.ndarray,
    blocks: Sequence[AuxiliaryBlock],
    device,
    *,
    variant: str,
    n_common: int,
    decur_weight: float,
    epochs: int,
    batch_size: int,
    patience: int,
    aux_blocks_per_step: int,
    seed: int,
):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    if variant not in {"direct_ot", "paired_decur_aux", "uniform_decur_aux"}:
        raise ValueError(f"Unknown variant: {variant}")
    plan_kind = "uniform" if variant == "uniform_decur_aux" else "ot"
    weight = 0.0 if variant == "direct_ot" else float(decur_weight)
    source_mean, source_scale = _standardization(source_features)
    target_mean, target_scale = _standardization(target_features)
    output_mean, output_scale = _standardization(residual_target)
    scaled_source = (source_features - source_mean) / source_scale
    scaled_target = (target_features - target_mean) / target_scale
    scaled_output = (residual_target - output_mean) / output_scale
    rng = np.random.RandomState(int(seed))
    order = rng.permutation(source_features.shape[0])
    n_validation = max(1, int(round(0.1 * order.size)))
    validation_rows = order[:n_validation]
    training_rows = order[n_validation:]
    train_mask = np.zeros(order.size, dtype=bool)
    train_mask[training_rows] = True
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(scaled_source[training_rows]),
            torch.from_numpy(scaled_output[training_rows]),
        ),
        batch_size=min(int(batch_size), training_rows.size),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model = copy.deepcopy(base_model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.0e-3,
        weight_decay=1.0e-5,
    )
    criterion = nn.SmoothL1Loss()
    validation_x = torch.from_numpy(scaled_source[validation_rows]).to(device)
    validation_y = torch.from_numpy(scaled_output[validation_rows]).to(device)
    scaled_source_tensor = torch.from_numpy(scaled_source).to(device)
    scaled_target_tensor = torch.from_numpy(scaled_target).to(device)
    local_masks = [
        train_mask[block.source_start : block.source_stop] for block in blocks
    ]
    device_masks = [
        torch.as_tensor(mask, device=device) for mask in local_masks
    ]
    eligible = np.asarray(
        [index for index, mask in enumerate(local_masks) if int(mask.sum()) >= 2]
    )
    auxiliary_rng = np.random.RandomState(int(seed) + 1)
    initial_diagnostics = {}
    if weight:
        with torch.no_grad():
            source_concepts = torch.empty(
                (scaled_source.shape[0], model.source_projection[-1].out_features),
                device=device,
            )
            target_concepts = torch.empty(
                (scaled_target.shape[0], model.target_projection[-1].out_features),
                device=device,
            )
            for start in range(0, scaled_source.shape[0], 8192):
                source_concepts[start : start + 8192] = model.source_concepts(
                    scaled_source_tensor[start : start + 8192]
                )
            for start in range(0, scaled_target.shape[0], 8192):
                target_concepts[start : start + 8192] = model.target_concepts(
                    scaled_target_tensor[start : start + 8192]
                )
        initial_diagnostics = correlation_diagnostics(
            model,
            source_concepts,
            target_concepts,
            blocks,
            train_mask,
            device,
            n_common=n_common,
            plan_kind=plan_kind,
        )
    history = []
    best_loss = float("inf")
    best_state = None
    stale = 0
    for epoch in range(int(epochs)):
        model.train()
        totals = {
            "prediction": 0.0,
            "decur": 0.0,
            "common": 0.0,
            "unique": 0.0,
            "cross_block": 0.0,
        }
        n_steps = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            prediction_loss = criterion(model.predict(batch_x), batch_y)
            zero = prediction_loss.new_zeros(())
            decur_parts = {
                key: zero
                for key in ("loss", "common", "unique", "cross_block")
            }
            if weight:
                selected = auxiliary_rng.choice(
                    eligible,
                    size=min(int(aux_blocks_per_step), eligible.size),
                    replace=False,
                )
                current = {
                    key: []
                    for key in ("loss", "common", "unique", "cross_block")
                }
                for block_index in selected:
                    block = blocks[int(block_index)]
                    local_mask = local_masks[int(block_index)]
                    device_mask = device_masks[int(block_index)]
                    source = scaled_source_tensor[
                        block.source_start : block.source_stop
                    ][device_mask]
                    target = scaled_target_tensor[
                        block.target_start : block.target_stop
                    ]
                    plan = block.plan[local_mask]
                    if plan_kind == "uniform":
                        plan = _uniform_like(plan)
                    parts = weighted_decur_loss(
                        model.source_concepts(source),
                        model.target_concepts(target),
                        torch.as_tensor(plan, device=device),
                        n_common=int(n_common),
                        lambda_off_diagonal=0.005,
                        cross_block_weight=0.1,
                    )
                    for key in current:
                        current[key].append(parts[key])
                decur_parts = {
                    key: torch.stack(values).mean()
                    for key, values in current.items()
                }
            loss = prediction_loss + weight * decur_parts["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals["prediction"] += float(prediction_loss.detach())
            for key in ("decur", "common", "unique", "cross_block"):
                part = "loss" if key == "decur" else key
                totals[key] += float(decur_parts[part].detach())
            n_steps += 1
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                criterion(model.predict(validation_x), validation_y)
            )
        summary = {
            "epoch": epoch + 1,
            **{key: value / n_steps for key, value in totals.items()},
            "validation_teacher_loss": validation_loss,
        }
        history.append(summary)
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == int(epochs):
            print(
                json.dumps(
                    {"event": "direct_ot_decur_epoch", "variant": variant, **summary}
                ),
                flush=True,
            )
        if validation_loss < best_loss - 1.0e-5:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a finite checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    final_diagnostics = {}
    if weight:
        with torch.no_grad():
            source_concepts = torch.cat(
                [
                    model.source_concepts(scaled_source_tensor[start : start + 8192])
                    for start in range(0, scaled_source.shape[0], 8192)
                ]
            )
            target_concepts = torch.cat(
                [
                    model.target_concepts(scaled_target_tensor[start : start + 8192])
                    for start in range(0, scaled_target.shape[0], 8192)
                ]
            )
        final_diagnostics = correlation_diagnostics(
            model,
            source_concepts,
            target_concepts,
            blocks,
            train_mask,
            device,
            n_common=n_common,
            plan_kind=plan_kind,
        )
    state = {
        "model": model,
        "source_mean": source_mean,
        "source_scale": source_scale,
        "output_mean": output_mean,
        "output_scale": output_scale,
        "history": history,
        "best_validation_teacher_loss": best_loss,
        "initial_correlation": initial_diagnostics,
        "final_correlation": final_diagnostics,
    }
    return state


def predict_residuals(
    state: dict,
    features: np.ndarray,
    device,
    *,
    batch_size: int = 8192,
) -> np.ndarray:
    import torch

    scaled = (
        np.asarray(features, dtype=np.float32) - state["source_mean"]
    ) / state["source_scale"]
    pieces = []
    with torch.no_grad():
        for start in range(0, scaled.shape[0], int(batch_size)):
            batch = torch.from_numpy(
                scaled[start : start + int(batch_size)]
            ).to(device)
            pieces.append(state["model"].predict(batch).cpu().numpy())
    prediction = np.concatenate(pieces)
    return (
        prediction * state["output_scale"][None, :]
        + state["output_mean"][None, :]
    ).astype(np.float32)


def main() -> None:
    import torch

    args = arguments()
    started = time.time()
    if args.k_min < 2 or args.k_max < args.k_min:
        raise ValueError("Require 2 <= k_min <= k_max")
    if (
        args.epsilon_ratio <= 0
        or args.sinkhorn_iterations < 1
        or args.epochs < 1
        or args.batch_size < 1
        or args.patience < 1
        or args.decur_weight < 0
        or args.aux_blocks_per_step < 1
        or not 1 <= args.n_common <= args.projection_dim
    ):
        raise ValueError("Invalid training hyperparameters")
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
    baseline = HOnlyRegressor(
        architecture="ridge",
        condition_on_cell_type=True,
    ).fit(target_h, target_y, cell_types=target_labels)
    prepared = prepare_blocks(
        train_blocks,
        np.arange(len(train_blocks)),
        panels,
        transformer,
        h_index,
        x_index,
        y_index,
    )
    teacher = build_teacher(
        prepared,
        scales,
        device,
        epsilon_ratio=args.epsilon_ratio,
        sinkhorn_iterations=args.sinkhorn_iterations,
    )
    classes = tuple(sorted(set(target_labels)))
    source_features = encoded_features(
        teacher.h,
        teacher.x,
        teacher.labels,
        classes,
    )
    train_base = baseline.predict(
        teacher.h,
        cell_types=teacher.labels,
    )
    residual_target = teacher.ot_y - train_base
    target_features, auxiliary_blocks, auxiliary_diagnostics = (
        build_auxiliary_data(
            prepared,
            baseline,
            classes,
            device,
            epsilon_ratio=args.epsilon_ratio,
            sinkhorn_iterations=args.sinkhorn_iterations,
        )
    )
    if auxiliary_diagnostics["n_source_cells"] != source_features.shape[0]:
        raise RuntimeError("Auxiliary blocks do not match direct teacher rows")

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
    validation_baseline = predict_baseline(baseline, validation)
    test_baseline = predict_baseline(baseline, test)
    baseline_metrics = {
        "validation": evaluate(validation_baseline, validation, scales),
        "test": evaluate(test_baseline, test, scales),
    }

    base_wrapper = DirectPredictor(
        source_features.shape[1],
        target_features.shape[1],
        residual_target.shape[1],
        hidden_dim=args.hidden_dim,
        projection_dim=args.projection_dim,
    )
    base_model = base_wrapper.network
    methods = {}
    checkpoints = {}
    for variant in ("direct_ot", "paired_decur_aux", "uniform_decur_aux"):
        state = train_variant(
            base_model,
            source_features,
            target_features,
            residual_target,
            auxiliary_blocks,
            device,
            variant=variant,
            n_common=args.n_common,
            decur_weight=args.decur_weight,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            aux_blocks_per_step=args.aux_blocks_per_step,
            seed=args.seed + 86,
        )
        validation_residual = {}
        test_residual = {}
        for specimen in validation["h"]:
            features = encoded_features(
                validation["h"][specimen],
                validation["x"][specimen],
                validation["labels"][specimen],
                classes,
            )
            validation_residual[specimen] = predict_residuals(
                state, features, device
            )
        selected_alpha, validation_curve = select_residual_alpha(
            validation_baseline,
            validation_residual,
            validation["y"],
            validation["patients"],
            scales,
            args.alphas,
        )
        for specimen in test["h"]:
            features = encoded_features(
                test["h"][specimen],
                test["x"][specimen],
                test["labels"][specimen],
                classes,
            )
            test_residual[specimen] = predict_residuals(
                state, features, device
            )
        validation_prediction = {
            specimen: validation_baseline[specimen]
            + selected_alpha * validation_residual[specimen]
            for specimen in validation["h"]
        }
        test_prediction = {
            specimen: test_baseline[specimen]
            + selected_alpha * test_residual[specimen]
            for specimen in test["h"]
        }
        methods[variant] = {
            "selected_alpha": selected_alpha,
            "validation_alpha_curve": validation_curve,
            "validation": evaluate(
                validation_prediction,
                validation,
                scales,
            ),
            "test": evaluate(test_prediction, test, scales),
            "training": {
                "epochs_completed": len(state["history"]),
                "best_validation_teacher_loss": (
                    state["best_validation_teacher_loss"]
                ),
                "first": state["history"][0],
                "last": state["history"][-1],
                "initial_correlation": state["initial_correlation"],
                "final_correlation": state["final_correlation"],
            },
        }
        checkpoints[variant] = {
            key: value.detach().cpu()
            for key, value in state["model"].state_dict().items()
        }
        del state
        if device.type == "cuda":
            torch.cuda.empty_cache()

    direct_reference = None
    if args.direct_reference is not None:
        reference = json.loads(args.direct_reference.read_text())
        frozen = reference["methods"]["mlp_paired_ot_hxl"]
        direct_reference = {
            "artifact": str(args.direct_reference),
            "method": "mlp_paired_ot_hxl",
            "selected_alpha": frozen["selected_alpha"],
            "validation": frozen["validation"],
            "test": frozen["test"],
        }
    checkpoint_path = args.output.with_suffix(".pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoints, checkpoint_path)
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
            "teacher_and_auxiliary_split": "train_only",
            "pairing": "same_specimen_same_cell_type_soft_ot",
            "ot_cost": "common_markers_only",
            "exclusive_markers_used_in_ot_cost": False,
            "source_target_training_cells_disjoint": True,
            "hidden_source_y_used_for_training": False,
            "validation_target_reference_used_for_inference": False,
            "test_target_reference_used_for_inference": False,
            "validation_and_test_specimens_unseen_during_fit": True,
            "target_encoder_used_at_inference": False,
            "projection_head_used_at_inference": False,
            "prediction_decoder_receives_projection_output": False,
            "test_used_for_model_or_alpha_selection": False,
        },
        "hyperparameters": {
            "k_max": args.k_max,
            "k_min": args.k_min,
            "epsilon_ratio": args.epsilon_ratio,
            "sinkhorn_iterations": args.sinkhorn_iterations,
            "epochs_maximum": args.epochs,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "hidden_dim": args.hidden_dim,
            "projection_dim": args.projection_dim,
            "n_common": args.n_common,
            "n_unique": args.projection_dim - args.n_common,
            "decur_weight": args.decur_weight,
            "aux_blocks_per_step": args.aux_blocks_per_step,
            "alphas": list(args.alphas),
        },
        "train": {
            "teacher": teacher.diagnostics,
            "auxiliary": auxiliary_diagnostics,
        },
        "baseline_hl": baseline_metrics,
        "direct_ot_reference": direct_reference,
        "methods": methods,
        "checkpoint": str(checkpoint_path),
        "elapsed_seconds": time.time() - started,
    }
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "event": "complete",
                "output": str(args.output),
                "elapsed_seconds": artifact["elapsed_seconds"],
                "methods": {
                    name: {
                        "alpha": values["selected_alpha"],
                        "validation_mae": values["validation"]["exact_cell"][
                            "patient_first_normalized_mae"
                        ],
                        "test_mae": values["test"]["exact_cell"][
                            "patient_first_normalized_mae"
                        ],
                    }
                    for name, values in methods.items()
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
