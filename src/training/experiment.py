"""End-to-end paired cross-panel training and evaluation."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src.data.aml import COARSE_CELL_TYPES
from src.data.cross_panel import CrossPanelDataset, load_cross_panel_dataset
from src.data.splits import patient_id_from_specimen
from src.evaluation.population_metrics import evaluate_matched_populations
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
    common_space: CrossPanelCommonSpace,
    *,
    k_max: int,
    k_min: int,
    epsilon_ratio: float,
    sinkhorn_iterations: int,
    seed: int,
    device: str,
) -> dict:
    import torch

    n_common = len(dataset.common_markers)
    rng = np.random.RandomState(seed)
    common, exclusive, targets, labels, groups = [], [], [], [], []
    block_count = 0
    for specimen in specimens:
        source = dataset.source[specimen]
        target = dataset.target[specimen]
        source_labels = source.cell_types.astype(str)
        target_labels = target.cell_types.astype(str)
        for cell_type in sorted(set(source_labels) & set(target_labels)):
            source_rows = np.flatnonzero(source_labels == cell_type)
            target_rows = np.flatnonzero(target_labels == cell_type)
            k = min(k_max, source_rows.size, target_rows.size)
            if k < k_min:
                continue
            source_rows = rng.choice(source_rows, k, replace=False)
            target_rows = rng.choice(target_rows, k, replace=False)
            source_h = common_space.source_percentiles(
                source.values[source_rows, :n_common]
            )
            target_h = common_space.target_percentiles(
                target.values[target_rows, :n_common]
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
                target.values[target_rows, n_common:], device=device
            )
            common.append(source_h)
            exclusive.append(source.values[source_rows, n_common:])
            targets.append(barycentric_projection(plan, target_y).cpu().numpy())
            labels.append(np.repeat(cell_type, k))
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


def _baseline_predictions(model, view: dict) -> dict[str, np.ndarray]:
    return {
        specimen: model.predict(
            view["source_h"][specimen],
            cell_types=view["source_labels"][specimen],
        )
        for specimen in view["source_h"]
    }


def _mlp_predictions(model: MLPRegressor, view: dict, classes, device, use_x: bool):
    return {
        specimen: model.predict(
            encode_features(
                view["source_h"][specimen],
                view["source_x"][specimen] if use_x else None,
                view["source_labels"][specimen],
                classes,
            ),
            device=device,
        )
        for specimen in view["source_h"]
    }


def _add(left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray], alpha: float):
    return {key: left[key] + alpha * right[key] for key in left}


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


def run_experiment(config: dict) -> dict:
    """Run one fold and seed, save metrics and a deployable model bundle."""

    import torch

    started = time.time()
    fold_index = int(config["experiment"]["fold"])
    seed = int(config["experiment"]["seed"])
    device = str(config["training"]["device"])
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
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

    baseline = HOnlyRegressor(condition_on_cell_type=True).fit(
        target_h, target_y, cell_types=target_labels
    )
    direct_features = encode_features(target_h, None, target_labels, classes)
    direct_mlp = _mlp(config["training"]["mlp"], seed + 10).fit(
        direct_features, target_y, groups=target_groups, device=device
    )
    teacher = _teacher(
        dataset,
        train_specimens,
        common_space,
        k_max=int(config["training"]["ot"]["k_max"]),
        k_min=int(config["training"]["ot"]["k_min"]),
        epsilon_ratio=float(config["training"]["ot"]["epsilon_ratio"]),
        sinkhorn_iterations=int(config["training"]["ot"]["sinkhorn_iterations"]),
        seed=seed + 20,
        device=device,
    )
    teacher_baseline = baseline.predict(teacher["common"], cell_types=teacher["labels"])
    residual_target = teacher["targets"] - teacher_baseline
    ot_hl = _mlp(config["training"]["mlp"], seed + 30).fit(
        encode_features(teacher["common"], None, teacher["labels"], classes),
        residual_target,
        groups=teacher["groups"],
        device=device,
    )
    proposed = _mlp(config["training"]["mlp"], seed + 40).fit(
        encode_features(
            teacher["common"], teacher["exclusive"], teacher["labels"], classes
        ),
        residual_target,
        groups=teacher["groups"],
        device=device,
    )

    global_median = GlobalMedianRegressor().fit(target_y)
    type_median = CellTypeMedianRegressor().fit(target_y, target_labels)
    knn = CyTOFMergeRegressor(
        k=int(config["training"]["knn"]["k"]),
        condition_on_cell_type=True,
        max_reference_cells=int(config["training"]["knn"]["max_reference_cells"]),
        n_jobs=int(config["training"]["knn"]["n_jobs"]),
        random_state=seed,
    ).fit(
        target_h,
        target_y,
        reference_cell_types=target_labels,
        reference_groups=target_groups,
    )

    validation = _split_view(dataset, validation_specimens, common_space)
    test = _split_view(dataset, test_specimens, common_space)
    validation_baseline = _baseline_predictions(baseline, validation)
    test_baseline = _baseline_predictions(baseline, test)
    validation_ot_hl = _mlp_predictions(ot_hl, validation, classes, device, False)
    test_ot_hl = _mlp_predictions(ot_hl, test, classes, device, False)
    validation_proposed = _mlp_predictions(proposed, validation, classes, device, True)
    test_proposed = _mlp_predictions(proposed, test, classes, device, True)
    alphas = tuple(map(float, config["evaluation"]["alphas"]))
    ot_hl_alpha, ot_hl_curve = _select_alpha(
        validation_baseline, validation_ot_hl, validation, scales, alphas
    )
    proposed_alpha, proposed_curve = _select_alpha(
        validation_baseline, validation_proposed, validation, scales, alphas
    )

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

    def knn_predictions(view):
        return {
            specimen: knn.predict(
                view["source_h"][specimen],
                query_cell_types=view["source_labels"][specimen],
            )
            for specimen in view["source_h"]
        }

    validation_direct = _mlp_predictions(direct_mlp, validation, classes, device, False)
    test_direct = _mlp_predictions(direct_mlp, test, classes, device, False)
    predictions = {
        "global_median": (global_predictions(validation), global_predictions(test)),
        "cell_type_median": (
            median_predictions(type_median, validation),
            median_predictions(type_median, test),
        ),
        "ridge_hl": (validation_baseline, test_baseline),
        "knn_hl": (knn_predictions(validation), knn_predictions(test)),
        "mlp_hl": (validation_direct, test_direct),
        "ot_hl": (
            _add(validation_baseline, validation_ot_hl, ot_hl_alpha),
            _add(test_baseline, test_ot_hl, ot_hl_alpha),
        ),
        "cytoalign": (
            _add(validation_baseline, validation_proposed, proposed_alpha),
            _add(test_baseline, test_proposed, proposed_alpha),
        ),
    }
    methods = {
        name: {
            "validation": _evaluate(validation_prediction, validation, scales),
            "test": _evaluate(test_prediction, test, scales),
        }
        for name, (validation_prediction, test_prediction) in predictions.items()
    }
    methods["ot_hl"].update(
        {"selected_alpha": ot_hl_alpha, "validation_alpha_curve": ot_hl_curve}
    )
    methods["cytoalign"].update(
        {
            "selected_alpha": proposed_alpha,
            "validation_alpha_curve": proposed_curve,
        }
    )

    output = (
        Path(config["output"]["root"])
        / config["experiment"]["name"]
        / f"fold_{fold_index}"
        / f"seed_{seed}"
    )
    output.mkdir(parents=True, exist_ok=True)
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
    )
    model.save(output / "model.pkl")
    result = {
        "status": "ok",
        "experiment": config["experiment"]["name"],
        "fold": fold_index,
        "seed": seed,
        "direction": f"{dataset.source_modality}_to_{dataset.target_modality}",
        "markers": {
            "common": list(dataset.common_markers),
            "source_exclusive": list(dataset.source_exclusive_columns),
            "target_exclusive": list(dataset.target_exclusive_columns),
        },
        "split": {
            "train_specimens": len(train_specimens),
            "validation_specimens": len(validation_specimens),
            "test_specimens": len(test_specimens),
        },
        "teacher_blocks": teacher["n_blocks"],
        "methods": methods,
        "model": str(output / "model.pkl"),
        "elapsed_seconds": time.time() - started,
    }
    (output / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result
