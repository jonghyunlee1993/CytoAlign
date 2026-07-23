"""End-to-end paired cross-panel training and evaluation."""

from __future__ import annotations

import json
import os
import random
import socket
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src.data.aml import COARSE_CELL_TYPES
from src.data.cross_panel import CrossPanelDataset, load_cross_panel_dataset
from src.data.splits import patient_id_from_specimen
from src.evaluation.population_metrics import evaluate_matched_populations
from src.evaluation.uncertainty import evaluate_uncertainty
from src.matching.optimal_transport import (
    balanced_sinkhorn,
    barycentric_projection,
    squared_euclidean_cost,
)
from src.models.cytoalign import CytoAlign, encode_features
from src.models.cytofmerge import CyTOFMergeRegressor
from src.models.h_only import (
    CellTypeMedianRegressor,
    GlobalMedianRegressor,
    HOnlyRegressor,
)
from src.models.mlp import MLPRegressor
from src.preprocessing.common_space import CrossPanelCommonSpace


def _marker_scale(values: np.ndarray) -> np.ndarray:
    iqr = np.quantile(values, 0.75, axis=0) - np.quantile(values, 0.25, axis=0)
    return np.maximum.reduce(
        [iqr, 0.1 * np.std(values, axis=0), np.full(iqr.shape, 1.0e-3)]
    )


def _sample(rows: int, maximum: int, rng: np.random.RandomState) -> np.ndarray:
    indices = np.arange(rows)
    if rows > maximum:
        indices = rng.choice(indices, maximum, replace=False)
    return np.sort(indices)


def _training_arrays(
    panel: Mapping[str, object],
    specimens: Sequence[str],
    n_common: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    common = []
    exclusive = []
    labels = []
    groups = []
    for specimen in specimens:
        current = panel[specimen]
        common.append(current.values[:, :n_common])
        exclusive.append(current.values[:, n_common:])
        labels.append(current.cell_types.astype(str))
        groups.append(np.repeat(specimen, len(current.values)))
    return (
        np.concatenate(common),
        np.concatenate(exclusive),
        np.concatenate(labels),
        np.concatenate(groups),
    )


def _fit_common_space(
    dataset: CrossPanelDataset,
    train_specimens: Sequence[str],
    *,
    n_knots: int,
    maximum_cells: int,
    seed: int,
) -> CrossPanelCommonSpace:
    n_common = len(dataset.common_markers)
    source_h, _, _, _ = _training_arrays(dataset.source, train_specimens, n_common)
    target_h, _, _, _ = _training_arrays(dataset.target, train_specimens, n_common)
    rng = np.random.RandomState(seed)
    source_rows = _sample(len(source_h), maximum_cells, rng)
    target_rows = _sample(len(target_h), maximum_cells, rng)
    return CrossPanelCommonSpace.fit(
        source_h[source_rows],
        target_h[target_rows],
        n_knots=n_knots,
    )


def _teacher(
    dataset: CrossPanelDataset,
    specimens: Sequence[str],
    target_specimens: Sequence[str],
    common_space: CrossPanelCommonSpace,
    *,
    k_max: int,
    k_min: int,
    epsilon_ratio: float,
    sinkhorn_iterations: int,
    seed: int,
    device: str,
    pooled_targets: bool = False,
    condition_on_cell_type: bool = True,
) -> dict:
    import torch

    n_common = len(dataset.common_markers)
    rng = np.random.RandomState(seed)
    common, exclusive, targets, labels, groups = [], [], [], [], []
    block_count = 0
    if pooled_targets:
        pooled_target_values = np.concatenate(
            [dataset.target[specimen].values for specimen in target_specimens]
        )
        pooled_target_labels = np.concatenate(
            [
                dataset.target[specimen].cell_types.astype(str)
                for specimen in target_specimens
            ]
        )
    for specimen, target_specimen in zip(specimens, target_specimens):
        source = dataset.source[specimen]
        source_labels = source.cell_types.astype(str)
        if pooled_targets:
            target_values = pooled_target_values
            target_labels = pooled_target_labels
        else:
            target = dataset.target[target_specimen]
            target_values = target.values
            target_labels = target.cell_types.astype(str)
        if condition_on_cell_type:
            blocks = [
                (
                    np.flatnonzero(source_labels == cell_type),
                    np.flatnonzero(target_labels == cell_type),
                )
                for cell_type in sorted(set(source_labels) & set(target_labels))
            ]
        else:
            blocks = [
                (
                    np.arange(source.values.shape[0]),
                    np.arange(target_values.shape[0]),
                )
            ]
        for source_rows, target_rows in blocks:
            k = min(k_max, source_rows.size, target_rows.size)
            if k < k_min:
                continue
            source_rows = rng.choice(source_rows, k, replace=False)
            target_rows = rng.choice(target_rows, k, replace=False)
            source_h = common_space.source_percentiles(
                source.values[source_rows, :n_common]
            )
            target_h = common_space.target_percentiles(
                target_values[target_rows, :n_common]
            )
            source_tensor = torch.as_tensor(source_h, device=device)
            target_tensor = torch.as_tensor(target_h, device=device)
            cost = squared_euclidean_cost(source_tensor, target_tensor)
            positive = cost[cost > 0]
            median = float(torch.median(positive)) if positive.numel() else 1.0
            plan = balanced_sinkhorn(
                cost,
                epsilon=max(epsilon_ratio * median, 1.0e-6),
                iterations=sinkhorn_iterations,
            )
            target_y = torch.as_tensor(
                target_values[target_rows, n_common:], device=device
            )
            common.append(source_h)
            exclusive.append(source.values[source_rows, n_common:])
            targets.append(barycentric_projection(plan, target_y).cpu().numpy())
            labels.append(source_labels[source_rows])
            groups.append(np.repeat(specimen, k))
            block_count += 1
    return {
        "common": np.concatenate(common).astype(np.float32),
        "exclusive": np.concatenate(exclusive).astype(np.float32),
        "targets": np.concatenate(targets).astype(np.float32),
        "labels": np.concatenate(labels).astype(str),
        "groups": np.concatenate(groups).astype(str),
        "n_blocks": block_count,
    }


def _split_view(
    dataset: CrossPanelDataset,
    specimens: Sequence[str],
    common_space: CrossPanelCommonSpace,
) -> dict:
    n_common = len(dataset.common_markers)
    source_h, source_x, source_labels = {}, {}, {}
    target_y, target_labels = {}, {}
    for specimen in specimens:
        source = dataset.source[specimen]
        target = dataset.target[specimen]
        source_h[specimen] = common_space.source_percentiles(
            source.values[:, :n_common]
        )
        source_x[specimen] = source.values[:, n_common:]
        source_labels[specimen] = source.cell_types.astype(str)
        target_y[specimen] = target.values[:, n_common:]
        target_labels[specimen] = target.cell_types.astype(str)
    return {
        "source_h": source_h,
        "source_x": source_x,
        "source_labels": source_labels,
        "target_y": target_y,
        "target_labels": target_labels,
        "patients": {
            specimen: patient_id_from_specimen(specimen) for specimen in specimens
        },
    }


def _pooled_view(view: dict) -> dict:
    specimens = sorted(view["source_h"])
    key = "pooled"
    return {
        "source_h": {key: np.concatenate([view["source_h"][s] for s in specimens])},
        "source_x": {key: np.concatenate([view["source_x"][s] for s in specimens])},
        "source_labels": {
            key: np.concatenate([view["source_labels"][s] for s in specimens])
        },
        "target_y": {key: np.concatenate([view["target_y"][s] for s in specimens])},
        "target_labels": {
            key: np.concatenate([view["target_labels"][s] for s in specimens])
        },
        "patients": {key: key},
    }


def _pooled_predictions(predictions: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {"pooled": np.concatenate([predictions[key] for key in sorted(predictions)])}


def _without_cell_type_strata(view: dict) -> dict:
    return {
        **view,
        "source_labels": {
            specimen: np.repeat("All", len(values))
            for specimen, values in view["source_h"].items()
        },
        "target_labels": {
            specimen: np.repeat("All", len(values))
            for specimen, values in view["target_y"].items()
        },
    }


def _evaluate(
    predictions: Mapping[str, np.ndarray], view: dict, scales: np.ndarray
) -> dict:
    return evaluate_matched_populations(
        predictions,
        view["source_labels"],
        view["target_y"],
        view["target_labels"],
        view["patients"],
        scales,
        minimum_cells=5,
    )


def _baseline_predictions(
    model, view: dict, *, condition_on_cell_type: bool = True
) -> dict[str, np.ndarray]:
    return {
        specimen: model.predict(
            view["source_h"][specimen],
            cell_types=(
                view["source_labels"][specimen] if condition_on_cell_type else None
            ),
        )
        for specimen in view["source_h"]
    }


def _knn_predictions_with_diagnostics(
    model, view: dict, *, condition_on_cell_type: bool = True
):
    predictions = {}
    diagnostics = {}
    for specimen in view["source_h"]:
        predictions[specimen], diagnostics[specimen] = model.predict(
            view["source_h"][specimen],
            cell_types=(
                view["source_labels"][specimen] if condition_on_cell_type else None
            ),
            return_diagnostics=True,
        )
    return predictions, diagnostics


def _mlp_predictions(
    model: MLPRegressor,
    view: dict,
    classes,
    device,
    use_x: bool,
    *,
    include_cell_types: bool = True,
):
    return {
        specimen: model.predict(
            encode_features(
                view["source_h"][specimen],
                view["source_x"][specimen] if use_x else None,
                view["source_labels"][specimen],
                classes,
                include_cell_types=include_cell_types,
            ),
            device=device,
        )
        for specimen in view["source_h"]
    }


def _add(left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray], alpha: float):
    return {key: left[key] + alpha * right[key] for key in left}


def _marker_add(
    left: Mapping[str, np.ndarray],
    right: Mapping[str, np.ndarray],
    alphas: Sequence[float],
):
    scale = np.asarray(alphas, dtype=float)[None, :]
    return {key: left[key] + scale * right[key] for key in left}


def _marker_score(baseline, residual, view, scales, marker: int, alpha: float) -> float:
    marker_view = {
        **view,
        "target_y": {
            specimen: values[:, marker : marker + 1]
            for specimen, values in view["target_y"].items()
        },
    }
    marker_predictions = {
        specimen: baseline[specimen][:, marker : marker + 1]
        + alpha * residual[specimen][:, marker : marker + 1]
        for specimen in baseline
    }
    return _evaluate(marker_predictions, marker_view, scales[marker : marker + 1])[
        "patient_first_normalized_wasserstein"
    ]


def _select_marker_alphas(baseline, residual, view, scales, alphas):
    selected = []
    curves = []
    for marker in range(scales.size):
        curve = {
            str(alpha): _marker_score(
                baseline, residual, view, scales, marker, alpha
            )
            for alpha in alphas
        }
        selected.append(min(alphas, key=lambda alpha: (curve[str(alpha)], alpha)))
        curves.append(curve)
    return np.asarray(selected, dtype=float), curves


def _select_alpha(baseline, residual, view, scales, alphas):
    curve = {
        str(alpha): _evaluate(_add(baseline, residual, alpha), view, scales)[
            "patient_first_normalized_wasserstein"
        ]
        for alpha in alphas
    }
    selected = min(alphas, key=lambda alpha: (curve[str(alpha)], alpha))
    return float(selected), curve


def _mlp(config: dict, seed: int) -> MLPRegressor:
    return MLPRegressor(
        hidden_dims=config["hidden_dims"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        patience=config["patience"],
        random_state=seed,
    )


def _paired_subset(specimens: Sequence[str], count: int, seed: int) -> tuple[str, ...]:
    ordered = np.asarray(sorted(specimens), dtype=object)
    permutation = np.random.RandomState(seed).permutation(ordered)
    return tuple(map(str, permutation[: int(count)]))


def _paired_targets(
    specimens: Sequence[str], pairing: str, seed: int
) -> tuple[str, ...]:
    source = np.asarray(tuple(map(str, specimens)), dtype=object)
    if source.size == 0:
        return ()
    if pairing == "matched":
        return tuple(map(str, source))
    if pairing == "unpaired":
        return tuple(map(str, source))
    if pairing != "shuffled":
        raise ValueError("OT pairing must be 'matched', 'shuffled', or 'unpaired'")
    if np.unique([patient_id_from_specimen(value) for value in source]).size < 2:
        raise ValueError("Shuffled pairing requires at least two patients")
    rng = np.random.RandomState(seed)
    for _ in range(10_000):
        target = rng.permutation(source)
        if all(
            patient_id_from_specimen(left) != patient_id_from_specimen(right)
            for left, right in zip(source, target)
        ):
            return tuple(map(str, target))
    raise RuntimeError("Could not construct a patient-disjoint shuffled pairing")


def run_experiment(config: dict) -> dict:
    """Run one fold and seed, optionally sweeping paired-specimen counts."""

    import torch

    started = time.time()
    fold_index = int(config["experiment"]["fold"])
    seed = int(config["experiment"]["seed"])
    device = str(config["training"]["device"])
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    hardware = {"host": socket.gethostname(), "device": device}
    if device == "cuda":
        probe = torch.randn(1024, 1024, device=device) @ torch.randn(
            1024, 1024, device=device
        )
        torch.cuda.synchronize()
        if not torch.isfinite(probe).all():
            raise RuntimeError("CUDA matrix multiplication failed")
        properties = torch.cuda.get_device_properties(0)
        hardware.update(
            {
                "gpu": torch.cuda.get_device_name(0),
                "gpu_memory_bytes": int(properties.total_memory),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            }
        )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset = load_cross_panel_dataset(config["data"])
    fold = dataset.splits["folds"][fold_index]
    train_specimens = fold["train_specimens"]
    validation_specimens = fold["validation_specimens"]
    test_specimens = fold["test_specimens"]
    common_space = _fit_common_space(
        dataset,
        train_specimens,
        n_knots=int(config["preprocessing"]["n_knots"]),
        maximum_cells=int(config["preprocessing"]["max_fit_cells"]),
        seed=seed,
    )

    n_common = len(dataset.common_markers)
    target_h_raw, target_y, target_labels, target_groups = _training_arrays(
        dataset.target, train_specimens, n_common
    )
    target_h = common_space.target_percentiles(target_h_raw)
    rng = np.random.RandomState(seed + 1)
    fit_rows = _sample(len(target_h), int(config["training"]["max_fit_cells"]), rng)
    target_h = target_h[fit_rows]
    target_y = target_y[fit_rows]
    target_labels = target_labels[fit_rows]
    target_groups = target_groups[fit_rows]
    scales = _marker_scale(target_y)
    classes = tuple(COARSE_CELL_TYPES)
    label_conditioning = bool(config["training"].get("label_conditioning", True))
    selection_uses_cell_types = bool(
        config["evaluation"].get(
            "label_stratified_selection", label_conditioning
        )
    )

    ridge = HOnlyRegressor(condition_on_cell_type=label_conditioning).fit(
        target_h,
        target_y,
        cell_types=target_labels if label_conditioning else None,
    )
    direct_features = encode_features(
        target_h,
        None,
        target_labels,
        classes,
        include_cell_types=label_conditioning,
    )
    direct_mlp = _mlp(config["training"]["mlp"], seed + 10).fit(
        direct_features, target_y, groups=target_groups, device=device
    )
    global_median = GlobalMedianRegressor().fit(target_y)
    type_median = CellTypeMedianRegressor().fit(target_y, target_labels)
    knn = CyTOFMergeRegressor(
        k=int(config["training"]["knn"]["k"]),
        condition_on_cell_type=label_conditioning,
        max_reference_cells=int(config["training"]["knn"]["max_reference_cells"]),
        n_jobs=int(config["training"]["knn"]["n_jobs"]),
        random_state=seed,
    ).fit(
        target_h,
        target_y,
        reference_cell_types=target_labels if label_conditioning else None,
        reference_groups=target_groups,
    )
    residual_baseline_name = str(
        config["training"].get("residual_baseline", "ridge_hl")
    )
    baselines = {"ridge_hl": ridge, "knn_hl": knn}
    if residual_baseline_name not in baselines:
        raise ValueError(
            f"Unknown residual baseline {residual_baseline_name!r}; "
            f"choose one of {tuple(baselines)}"
        )
    baseline = baselines[residual_baseline_name]

    validation = _split_view(dataset, validation_specimens, common_space)
    test = _split_view(dataset, test_specimens, common_space)
    diagnostics_enabled = bool(
        config["evaluation"].get("uncertainty_diagnostics", False)
    )
    test_knn_diagnostics = None
    validation_knn = _baseline_predictions(
        knn, validation, condition_on_cell_type=label_conditioning
    )
    if diagnostics_enabled:
        test_knn, test_knn_diagnostics = _knn_predictions_with_diagnostics(
            knn, test, condition_on_cell_type=label_conditioning
        )
    else:
        test_knn = _baseline_predictions(
            knn, test, condition_on_cell_type=label_conditioning
        )
    baseline_predictions = {
        "ridge_hl": (
            _baseline_predictions(
                ridge, validation, condition_on_cell_type=label_conditioning
            ),
            _baseline_predictions(
                ridge, test, condition_on_cell_type=label_conditioning
            ),
        ),
        "knn_hl": (validation_knn, test_knn),
    }
    validation_baseline, test_baseline = baseline_predictions[residual_baseline_name]

    def median_predictions(model, view):
        return {
            specimen: model.predict(view["source_labels"][specimen])
            for specimen in view["source_h"]
        }

    def global_predictions(view):
        return {
            specimen: global_median.predict(len(view["source_h"][specimen]))
            for specimen in view["source_h"]
        }

    validation_direct = _mlp_predictions(
        direct_mlp,
        validation,
        classes,
        device,
        False,
        include_cell_types=label_conditioning,
    )
    test_direct = _mlp_predictions(
        direct_mlp,
        test,
        classes,
        device,
        False,
        include_cell_types=label_conditioning,
    )
    shared_predictions = {
        "global_median": (global_predictions(validation), global_predictions(test)),
        "cell_type_median": (
            median_predictions(type_median, validation),
            median_predictions(type_median, test),
        ),
        **baseline_predictions,
        "mlp_hl": (validation_direct, test_direct),
    }
    shared_methods = {
        name: {
            "validation": _evaluate(validation_prediction, validation, scales),
            "test": _evaluate(test_prediction, test, scales),
        }
        for name, (validation_prediction, test_prediction) in shared_predictions.items()
    }

    output = (
        Path(config["output"]["root"])
        / config["experiment"]["name"]
        / f"fold_{fold_index}"
        / f"seed_{seed}"
    )
    output.mkdir(parents=True, exist_ok=True)

    paired_counts = config["experiment"].get("paired_counts")
    curve_mode = paired_counts is not None
    if paired_counts is None:
        paired_counts = [len(train_specimens)]
    paired_counts = tuple(map(int, paired_counts))
    alphas = tuple(map(float, config["evaluation"]["alphas"]))
    pairing = str(config["training"]["ot"].get("pairing", "matched"))
    selection_pairing = str(
        config["evaluation"].get("selection_pairing", "matched")
    )
    if selection_pairing not in {"matched", "unpaired"}:
        raise ValueError("selection_pairing must be 'matched' or 'unpaired'")
    marker_gate_enabled = bool(config["evaluation"].get("marker_gate", False))
    paired_curve = {}
    for paired_count in paired_counts:
        point_started = time.time()
        paired_specimens = _paired_subset(train_specimens, paired_count, seed + 19)
        paired_targets = _paired_targets(paired_specimens, pairing, seed + 21)
        if paired_count == 0:
            teacher_blocks = 0
            proposed = None
            ot_hl_alpha = proposed_alpha = 0.0
            ot_hl_curve = proposed_curve = {
                str(alpha): shared_methods[residual_baseline_name]["validation"][
                    "patient_first_normalized_wasserstein"
                ]
                for alpha in alphas
            }
            validation_ot_prediction = validation_baseline
            test_ot_prediction = test_baseline
            validation_proposed_prediction = validation_baseline
            test_proposed_prediction = test_baseline
            marker_alphas = np.zeros(scales.size)
            marker_curves = []
            test_marker_prediction = test_baseline
            uncertainty = None
        else:
            teacher = _teacher(
                dataset,
                paired_specimens,
                paired_targets,
                common_space,
                k_max=int(config["training"]["ot"]["k_max"]),
                k_min=int(config["training"]["ot"]["k_min"]),
                epsilon_ratio=float(config["training"]["ot"]["epsilon_ratio"]),
                sinkhorn_iterations=int(
                    config["training"]["ot"]["sinkhorn_iterations"]
                ),
                seed=seed + 20,
                device=device,
                pooled_targets=pairing == "unpaired",
                condition_on_cell_type=label_conditioning,
            )
            teacher_blocks = teacher["n_blocks"]
            teacher_baseline = baseline.predict(
                teacher["common"],
                cell_types=teacher["labels"] if label_conditioning else None,
            )
            residual_target = teacher["targets"] - teacher_baseline
            ot_hl = _mlp(config["training"]["mlp"], seed + 30).fit(
                encode_features(
                    teacher["common"],
                    None,
                    teacher["labels"],
                    classes,
                    include_cell_types=label_conditioning,
                ),
                residual_target,
                groups=teacher["groups"],
                device=device,
            )
            proposed = _mlp(config["training"]["mlp"], seed + 40).fit(
                encode_features(
                    teacher["common"],
                    teacher["exclusive"],
                    teacher["labels"],
                    classes,
                    include_cell_types=label_conditioning,
                ),
                residual_target,
                groups=teacher["groups"],
                device=device,
            )
            validation_ot_hl = _mlp_predictions(
                ot_hl,
                validation,
                classes,
                device,
                False,
                include_cell_types=label_conditioning,
            )
            test_ot_hl = _mlp_predictions(
                ot_hl,
                test,
                classes,
                device,
                False,
                include_cell_types=label_conditioning,
            )
            validation_proposed = _mlp_predictions(
                proposed,
                validation,
                classes,
                device,
                True,
                include_cell_types=label_conditioning,
            )
            test_proposed = _mlp_predictions(
                proposed,
                test,
                classes,
                device,
                True,
                include_cell_types=label_conditioning,
            )
            if selection_pairing == "unpaired":
                selection_view = _pooled_view(validation)
                selection_baseline = _pooled_predictions(validation_baseline)
                selection_ot_hl = _pooled_predictions(validation_ot_hl)
                selection_proposed = _pooled_predictions(validation_proposed)
            else:
                selection_view = validation
                selection_baseline = validation_baseline
                selection_ot_hl = validation_ot_hl
                selection_proposed = validation_proposed
            if not selection_uses_cell_types:
                selection_view = _without_cell_type_strata(selection_view)
            ot_hl_alpha, ot_hl_curve = _select_alpha(
                selection_baseline,
                selection_ot_hl,
                selection_view,
                scales,
                alphas,
            )
            proposed_alpha, proposed_curve = _select_alpha(
                selection_baseline,
                selection_proposed,
                selection_view,
                scales,
                alphas,
            )
            validation_ot_prediction = _add(
                validation_baseline, validation_ot_hl, ot_hl_alpha
            )
            test_ot_prediction = _add(test_baseline, test_ot_hl, ot_hl_alpha)
            validation_proposed_prediction = _add(
                validation_baseline, validation_proposed, proposed_alpha
            )
            test_proposed_prediction = _add(
                test_baseline, test_proposed, proposed_alpha
            )
            if marker_gate_enabled:
                marker_alphas, marker_curves = _select_marker_alphas(
                    selection_baseline,
                    selection_proposed,
                    selection_view,
                    scales,
                    alphas,
                )
                test_marker_prediction = _marker_add(
                    test_baseline, test_proposed, marker_alphas
                )
            else:
                marker_alphas = np.repeat(proposed_alpha, scales.size)
                marker_curves = []
                test_marker_prediction = test_proposed_prediction
            uncertainty = None
            if diagnostics_enabled:
                if residual_baseline_name != "knn_hl":
                    raise ValueError(
                        "Uncertainty diagnostics require residual_baseline=knn_hl"
                    )
                uncertainty = evaluate_uncertainty(
                    test_baseline,
                    test_proposed,
                    test_knn_diagnostics,
                    test,
                    scales,
                    alphas,
                    proposed_alpha,
                    dataset.target_exclusive_columns,
                )

        methods = dict(shared_methods)
        methods["ot_hl"] = {
            "selected_alpha": ot_hl_alpha,
            "validation_alpha_curve": ot_hl_curve,
            "validation": _evaluate(validation_ot_prediction, validation, scales),
            "test": _evaluate(test_ot_prediction, test, scales),
        }
        methods["cytoalign"] = {
            "selected_alpha": proposed_alpha,
            "validation_alpha_curve": proposed_curve,
            "validation": _evaluate(validation_proposed_prediction, validation, scales),
            "test": _evaluate(test_proposed_prediction, test, scales),
        }
        if marker_gate_enabled:
            methods["cytoalign_marker_gate"] = {
                "selected_alphas": marker_alphas.tolist(),
                "validation_alpha_curves": marker_curves,
                "test": _evaluate(test_marker_prediction, test, scales),
            }
        model_path = None
        if bool(config["output"].get("save_models", True)):
            model_name = (
                f"model_paired_{paired_count}.pkl" if curve_mode else "model.pkl"
            )
            model = CytoAlign(
                common_space=common_space,
                baseline=baseline,
                residual=proposed,
                classes=classes,
                alpha=proposed_alpha,
                source_modality=dataset.source_modality,
                target_modality=dataset.target_modality,
                source_common_columns=dataset.source_common_columns,
                source_exclusive_columns=dataset.source_exclusive_columns,
                target_markers=dataset.target_exclusive_columns,
                label_conditioning=label_conditioning,
            )
            model.save(output / model_name)
            model_path = str(output / model_name)
        paired_curve[str(paired_count)] = {
            "paired_specimens": list(paired_specimens),
            "paired_target_specimens": list(paired_targets),
            "pairing": pairing,
            "teacher_blocks": teacher_blocks,
            "methods": methods,
            "uncertainty": uncertainty,
            "model": model_path,
            "elapsed_seconds": time.time() - point_started,
        }
        if device == "cuda":
            torch.cuda.empty_cache()

    result = {
        "status": "ok",
        "experiment": config["experiment"]["name"],
        "fold": fold_index,
        "seed": seed,
        "residual_baseline": residual_baseline_name,
        "pairing": pairing,
        "selection_pairing": selection_pairing,
        "label_usage": {
            "training_conditioning": label_conditioning,
            "validation_selection_stratified": selection_uses_cell_types,
            "test_evaluation_stratified": True,
            "label_assisted_methods": ["cell_type_median"],
        },
        "hardware": hardware,
        "direction": f"{dataset.source_modality}_to_{dataset.target_modality}",
        "markers": {
            "common": list(dataset.common_markers),
            "omitted_common": list(dataset.omitted_common_markers),
            "source_exclusive": list(dataset.source_exclusive_columns),
            "target_exclusive": list(dataset.target_exclusive_columns),
        },
        "split": {
            "train_specimens": len(train_specimens),
            "validation_specimens": len(validation_specimens),
            "test_specimens": len(test_specimens),
        },
        "shared_methods": shared_methods,
        "paired_curve": paired_curve,
        "elapsed_seconds": time.time() - started,
    }
    if not curve_mode:
        point = paired_curve[str(paired_counts[0])]
        result.update(
            {
                "teacher_blocks": point["teacher_blocks"],
                "methods": point["methods"],
                "model": point["model"],
            }
        )
    (output / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result
