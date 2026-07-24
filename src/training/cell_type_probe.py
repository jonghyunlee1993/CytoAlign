"""Patient-held-out probes of cell-type information in translated cells."""

from __future__ import annotations

import json
import os
import random
import socket
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.aml import COARSE_CELL_TYPES
from src.data.cross_panel import CrossPanelDataset, load_cross_panel_dataset
from src.data.splits import patient_id_from_specimen
from src.models.cytoalign import encode_features
from src.models.cytofmerge import CyTOFMergeRegressor
from src.training.experiment import (
    _baseline_predictions,
    _fit_common_space,
    _marker_add,
    _marker_scale,
    _mlp,
    _mlp_predictions,
    _sample,
    _select_marker_alphas,
    _split_view,
    _teacher,
    _training_arrays,
    _without_cell_type_strata,
)


FINE_CELL_TYPES = (
    "Blast",
    "Monocyte",
    "T cell CD4",
    "T cell CD8",
    "T cell DN",
    "T cell DP",
    "B cell",
    "NK cell",
)
RARE_T_CELL_TYPES = ("T cell DN", "T cell DP")
T_CELL_TYPES = ("T cell CD4", "T cell CD8", "T cell DN", "T cell DP")
REPRESENTATIONS = (
    "source_h",
    "source_h_plus_x",
    "translated_y_h_only",
    "translated_y_h_residual_ungated",
    "translated_y_h_shuffled_x_ungated",
    "translated_y_h_plus_x_ungated",
    "translated_y_h_plus_x_gate",
    "target_like_h_y_h_only",
    "target_like_h_y_h_plus_x_gate",
)


def _hardware_probe(device: str) -> dict:
    import torch

    hardware = {"host": socket.gethostname(), "device": device}
    if device != "cuda":
        return hardware
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
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
    return hardware


def _fine_labels(dataset: CrossPanelDataset, specimen: str) -> np.ndarray:
    labels = dataset.source[specimen].fine_cell_types
    if labels is None:
        raise ValueError("Fine cell labels were not retained by the data loader")
    return np.asarray(labels).astype(str)


def balanced_probe_rows(
    dataset: CrossPanelDataset,
    specimens: Sequence[str],
    *,
    per_specimen_class_cap: int,
    total_class_cap: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Select a patient-diverse, approximately class-balanced probe training set."""

    rng = np.random.RandomState(int(seed))
    candidates: dict[str, list[tuple[str, int]]] = {
        label: [] for label in FINE_CELL_TYPES
    }
    for specimen in specimens:
        labels = _fine_labels(dataset, specimen)
        for label in FINE_CELL_TYPES:
            rows = np.flatnonzero(labels == label)
            if rows.size > int(per_specimen_class_cap):
                rows = rng.choice(
                    rows, int(per_specimen_class_cap), replace=False
                )
            candidates[label].extend((str(specimen), int(row)) for row in rows)

    selected: dict[str, list[int]] = {str(specimen): [] for specimen in specimens}
    for label in FINE_CELL_TYPES:
        current = candidates[label]
        if len(current) > int(total_class_cap):
            keep = rng.choice(len(current), int(total_class_cap), replace=False)
            current = [current[index] for index in keep]
        for specimen, row in current:
            selected[specimen].append(row)
    return {
        specimen: np.asarray(sorted(set(rows)), dtype=np.int64)
        for specimen, rows in selected.items()
        if rows
    }


def _all_rows(
    dataset: CrossPanelDataset, specimens: Sequence[str]
) -> dict[str, np.ndarray]:
    return {
        specimen: np.arange(
            dataset.source[specimen].values.shape[0], dtype=np.int64
        )
        for specimen in specimens
    }


def _translated_representations(
    dataset: CrossPanelDataset,
    rows_by_specimen: Mapping[str, np.ndarray],
    common_space,
    knn,
    h_residual,
    shuffled_x_residual,
    h_plus_x_residual,
    marker_alphas: np.ndarray,
    *,
    device: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    n_common = len(dataset.common_markers)
    pieces = {name: [] for name in REPRESENTATIONS}
    fine_labels = []
    coarse_labels = []
    patients = []
    for specimen in sorted(rows_by_specimen):
        rows = np.asarray(rows_by_specimen[specimen], dtype=np.int64)
        source = dataset.source[specimen]
        h = common_space.source_percentiles(source.values[rows, :n_common])
        x = source.values[rows, n_common:]
        h_only = knn.predict(h)
        h_residual_prediction = h_residual.predict(
            encode_features(
                h,
                None,
                source.cell_types[rows],
                (),
                include_cell_types=False,
            ),
            device=device,
        )
        shuffled_x_residual_prediction = shuffled_x_residual.predict(
            encode_features(
                h,
                x,
                source.cell_types[rows],
                (),
                include_cell_types=False,
            ),
            device=device,
        )
        h_plus_x_residual_prediction = h_plus_x_residual.predict(
            encode_features(
                h,
                x,
                source.cell_types[rows],
                (),
                include_cell_types=False,
            ),
            device=device,
        )
        h_plus_x_ungated = h_only + h_plus_x_residual_prediction
        h_plus_x_gate = (
            h_only
            + marker_alphas[None, :] * h_plus_x_residual_prediction
        )
        current = {
            "source_h": h,
            "source_h_plus_x": np.concatenate([h, x], axis=1),
            "translated_y_h_only": h_only,
            "translated_y_h_residual_ungated": (
                h_only + h_residual_prediction
            ),
            "translated_y_h_shuffled_x_ungated": (
                h_only + shuffled_x_residual_prediction
            ),
            "translated_y_h_plus_x_ungated": h_plus_x_ungated,
            "translated_y_h_plus_x_gate": h_plus_x_gate,
            "target_like_h_y_h_only": np.concatenate([h, h_only], axis=1),
            "target_like_h_y_h_plus_x_gate": np.concatenate(
                [h, h_plus_x_gate], axis=1
            ),
        }
        for name in REPRESENTATIONS:
            pieces[name].append(np.asarray(current[name], dtype=np.float32))
        fine_labels.append(_fine_labels(dataset, specimen)[rows])
        coarse_labels.append(source.cell_types[rows].astype(str))
        patients.append(
            np.repeat(patient_id_from_specimen(specimen), rows.size)
        )
    return (
        {name: np.concatenate(values) for name, values in pieces.items()},
        np.concatenate(fine_labels).astype(str),
        np.concatenate(coarse_labels).astype(str),
        np.concatenate(patients).astype(str),
    )


def shuffle_rows_within_groups(
    values: np.ndarray,
    groups: Sequence,
    *,
    seed: int,
) -> np.ndarray:
    """Destroy row-level X association while preserving each group marginal."""

    array = np.asarray(values)
    group_values = np.asarray(groups).astype(str)
    if len(array) != len(group_values):
        raise ValueError("values and groups must have the same number of rows")
    rng = np.random.RandomState(int(seed))
    shuffled = np.empty_like(array)
    for group in np.unique(group_values):
        rows = np.flatnonzero(group_values == group)
        shuffled[rows] = array[rng.permutation(rows)]
    return shuffled


def probe_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    classes: Sequence[str],
) -> dict:
    """Return imbalance-aware multiclass and per-class probe metrics."""

    labels = np.asarray(labels).astype(str)
    predictions = np.asarray(predictions).astype(str)
    classes = tuple(map(str, classes))
    matrix = confusion_matrix(labels, predictions, labels=classes)
    supports = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    recalls = np.divide(
        np.diag(matrix),
        supports,
        out=np.zeros(len(classes), dtype=float),
        where=supports > 0,
    )
    precisions = np.divide(
        np.diag(matrix),
        predicted,
        out=np.zeros(len(classes), dtype=float),
        where=predicted > 0,
    )
    per_class = {}
    auprcs = []
    for index, label in enumerate(classes):
        binary = labels == label
        auprc = (
            float(average_precision_score(binary, probabilities[:, index]))
            if np.any(binary)
            else None
        )
        if auprc is not None:
            auprcs.append(auprc)
        per_class[label] = {
            "support": int(supports[index]),
            "recall": float(recalls[index]) if supports[index] else None,
            "precision": float(precisions[index]) if supports[index] else None,
            "auprc": auprc,
        }

    def class_mean(labels_to_average: Sequence[str], metric: str):
        values = [
            per_class[label][metric]
            for label in labels_to_average
            if label in per_class and per_class[label][metric] is not None
        ]
        return float(np.mean(values)) if values else None

    return {
        "accuracy": float(np.mean(labels == predictions)),
        "balanced_accuracy": float(np.mean(recalls[supports > 0])),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=list(classes),
                average="macro",
                zero_division=0,
            )
        ),
        "macro_auprc": float(np.mean(auprcs)) if auprcs else None,
        "t_subtype_macro_auprc": class_mean(T_CELL_TYPES, "auprc"),
        "rare_t_macro_auprc": class_mean(RARE_T_CELL_TYPES, "auprc"),
        "confusion_matrix": matrix.tolist(),
        "classes": list(classes),
        "per_class": per_class,
    }


def _probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    test_patients: np.ndarray,
    classes: Sequence[str],
    *,
    c_value: float,
    seed: int,
) -> tuple[dict, dict[str, dict]]:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(c_value),
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=int(seed),
        ),
    ).fit(train_features, train_labels)
    predictions = model.predict(test_features)
    raw_probabilities = model.predict_proba(test_features)
    fitted_classes = model[-1].classes_.astype(str)
    lookup = {label: index for index, label in enumerate(fitted_classes)}
    probabilities = np.zeros((len(test_labels), len(classes)), dtype=np.float64)
    for index, label in enumerate(classes):
        if label in lookup:
            probabilities[:, index] = raw_probabilities[:, lookup[label]]
    overall = probe_metrics(test_labels, predictions, probabilities, classes)
    patient_metrics = {}
    for patient in np.unique(test_patients):
        rows = test_patients == patient
        patient_metrics[str(patient)] = probe_metrics(
            test_labels[rows],
            predictions[rows],
            probabilities[rows],
            classes,
        )
    return overall, patient_metrics


def run_cell_type_probe_fold(config: dict) -> dict:
    """Fit a label-free translator and test fine/coarse label recoverability."""

    import torch

    started = time.time()
    fold_index = int(config["experiment"]["fold"])
    seed = int(config["experiment"]["seed"])
    device = str(config["training"]["device"])
    hardware = _hardware_probe(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset = load_cross_panel_dataset(config["data"])
    observed_fine = {
        label
        for specimen in dataset.source.values()
        for label in np.unique(specimen.fine_cell_types)
    }
    if observed_fine != set(FINE_CELL_TYPES):
        raise ValueError(
            f"Unexpected source fine-label taxonomy: {sorted(observed_fine)}"
        )
    fold = dataset.splits["folds"][fold_index]
    train_specimens = tuple(map(str, fold["train_specimens"]))
    validation_specimens = tuple(map(str, fold["validation_specimens"]))
    test_specimens = tuple(map(str, fold["test_specimens"]))
    outer_train_specimens = tuple(sorted(train_specimens + validation_specimens))
    common_space = _fit_common_space(
        dataset,
        train_specimens,
        n_knots=int(config["preprocessing"]["n_knots"]),
        maximum_cells=int(config["preprocessing"]["max_fit_cells"]),
        seed=seed,
    )

    n_common = len(dataset.common_markers)
    target_h_raw, target_y, _, target_groups = _training_arrays(
        dataset.target, train_specimens, n_common
    )
    target_h = common_space.target_percentiles(target_h_raw)
    rng = np.random.RandomState(seed + 1)
    fit_rows = _sample(len(target_h), int(config["training"]["max_fit_cells"]), rng)
    target_h = target_h[fit_rows]
    target_y = target_y[fit_rows]
    target_groups = target_groups[fit_rows]
    scales = _marker_scale(target_y)
    knn_config = config["training"]["knn"]
    knn = CyTOFMergeRegressor(
        k=int(knn_config["k"]),
        condition_on_cell_type=False,
        max_reference_cells=int(knn_config["max_reference_cells"]),
        n_jobs=int(knn_config["n_jobs"]),
        random_state=seed,
    ).fit(target_h, target_y, reference_groups=target_groups)

    validation = _split_view(dataset, validation_specimens, common_space)
    validation_h_only = _baseline_predictions(
        knn, validation, condition_on_cell_type=False
    )
    ot_config = config["training"]["ot"]
    teacher = _teacher(
        dataset,
        train_specimens,
        train_specimens,
        common_space,
        k_max=int(ot_config["k_max"]),
        k_min=int(ot_config["k_min"]),
        epsilon_ratio=float(ot_config["epsilon_ratio"]),
        sinkhorn_iterations=int(ot_config["sinkhorn_iterations"]),
        seed=seed + 20,
        device=device,
        pooled_targets=False,
        condition_on_cell_type=False,
    )
    teacher_baseline = knn.predict(teacher["common"])
    residual_target = teacher["targets"] - teacher_baseline
    h_residual = _mlp(config["training"]["mlp"], seed + 30).fit(
        encode_features(
            teacher["common"],
            None,
            teacher["labels"],
            (),
            include_cell_types=False,
        ),
        residual_target,
        groups=teacher["groups"],
        device=device,
    )
    shuffled_exclusive = shuffle_rows_within_groups(
        teacher["exclusive"],
        teacher["groups"],
        seed=seed + 35,
    )
    shuffled_x_residual = _mlp(config["training"]["mlp"], seed + 35).fit(
        encode_features(
            teacher["common"],
            shuffled_exclusive,
            teacher["labels"],
            (),
            include_cell_types=False,
        ),
        residual_target,
        groups=teacher["groups"],
        device=device,
    )
    h_plus_x_residual = _mlp(config["training"]["mlp"], seed + 40).fit(
        encode_features(
            teacher["common"],
            teacher["exclusive"],
            teacher["labels"],
            (),
            include_cell_types=False,
        ),
        residual_target,
        groups=teacher["groups"],
        device=device,
    )
    validation_residual = _mlp_predictions(
        h_plus_x_residual,
        validation,
        (),
        device,
        True,
        include_cell_types=False,
    )
    alphas = tuple(map(float, config["evaluation"]["alphas"]))
    marker_alphas, marker_curves = _select_marker_alphas(
        validation_h_only,
        validation_residual,
        _without_cell_type_strata(validation),
        scales,
        alphas,
    )

    probe_config = config["probe"]
    train_rows = balanced_probe_rows(
        dataset,
        outer_train_specimens,
        per_specimen_class_cap=int(probe_config["per_specimen_class_cap"]),
        total_class_cap=int(probe_config["total_class_cap"]),
        seed=seed + 101,
    )
    test_rows = _all_rows(dataset, test_specimens)
    train_features, train_fine, train_coarse, _ = _translated_representations(
        dataset,
        train_rows,
        common_space,
        knn,
        h_residual,
        shuffled_x_residual,
        h_plus_x_residual,
        marker_alphas,
        device=device,
    )
    test_features, test_fine, test_coarse, test_patients = (
        _translated_representations(
            dataset,
            test_rows,
            common_space,
            knn,
            h_residual,
            shuffled_x_residual,
            h_plus_x_residual,
            marker_alphas,
            device=device,
        )
    )

    resolutions = {}
    for resolution, train_labels, test_labels, classes in (
        ("fine", train_fine, test_fine, FINE_CELL_TYPES),
        ("coarse", train_coarse, test_coarse, COARSE_CELL_TYPES),
    ):
        methods = {}
        for method_index, method in enumerate(REPRESENTATIONS):
            overall, patient_metrics = _probe(
                train_features[method],
                train_labels,
                test_features[method],
                test_labels,
                test_patients,
                classes,
                c_value=float(probe_config["classifier_c"]),
                seed=seed + 1000 * (resolution == "coarse") + method_index,
            )
            methods[method] = {
                "overall": overall,
                "patients": patient_metrics,
            }
        resolutions[resolution] = {
            "classes": list(classes),
            "methods": methods,
        }

    output = (
        Path(config["output"]["root"])
        / str(config["experiment"]["name"])
        / f"fold_{fold_index}"
        / f"seed_{seed}"
    )
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "ok",
        "experiment": str(config["experiment"]["name"]),
        "fold": fold_index,
        "seed": seed,
        "hardware": hardware,
        "design": {
            "split_unit": "patient",
            "outer_folds": int(config["data"]["split"]["n_splits"]),
            "cells_per_specimen_cap": int(config["data"]["max_cells_per_specimen"]),
            "translator_uses_cell_labels": False,
            "test_sampling": "all uniformly reservoir-sampled source cells",
            "probe_training_sampling": "fine-class balanced caps across outer-train specimens",
            "probe_train_cells": int(len(train_fine)),
            "probe_test_cells": int(len(test_fine)),
            "translator_train_specimens": len(train_specimens),
            "gate_validation_specimens": len(validation_specimens),
            "outer_test_specimens": len(test_specimens),
        },
        "markers": {
            "common": list(dataset.common_markers),
            "source_exclusive": list(dataset.source_exclusive_columns),
            "target_exclusive": list(dataset.target_exclusive_columns),
        },
        "translation": {
            "baseline": "label-free H-only kNN median",
            "controls": [
                "capacity-matched label-free H residual MLP",
                "label-free residual MLP trained with X shuffled within specimen",
            ],
            "extension": "label-free H+X residual MLP with correct X",
            "gate": "label-free marker-wise validation alpha",
            "selected_alphas": marker_alphas.tolist(),
            "alpha_candidates": list(alphas),
            "validation_alpha_curves": marker_curves,
            "teacher_blocks": int(teacher["n_blocks"]),
        },
        "resolutions": resolutions,
        "elapsed_seconds": time.time() - started,
    }
    result_path = output / "cell_type_probe.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    result["result_path"] = str(result_path)
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return float(np.mean(finite)) if finite else None


def _interval(values: Sequence[float]) -> list[float] | None:
    finite = np.asarray([value for value in values if np.isfinite(value)])
    if not finite.size:
        return None
    return [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]


def summarize_cell_type_probe(
    experiment_root: str | Path,
    *,
    expected_folds: int = 5,
    bootstrap_replicates: int = 2000,
    seed: int = 4207,
) -> dict:
    """Aggregate patient-first probe metrics and paired representation deltas."""

    root = Path(experiment_root)
    paths = sorted(root.glob("fold_*/seed_*/cell_type_probe.json"))
    if len(paths) != int(expected_folds):
        raise ValueError(
            f"Expected exactly {expected_folds} fold results, found {len(paths)}"
        )
    payloads = [json.loads(path.read_text()) for path in paths]
    folds = sorted(int(payload["fold"]) for payload in payloads)
    if folds != list(range(int(expected_folds))):
        raise ValueError(f"Unexpected folds: {folds}")
    rng = np.random.RandomState(int(seed))
    output_resolutions = {}
    for resolution in ("fine", "coarse"):
        classes = tuple(payloads[0]["resolutions"][resolution]["classes"])
        patient_rows: dict[str, dict[str, dict]] = {
            method: {} for method in REPRESENTATIONS
        }
        confusion = {
            method: np.zeros((len(classes), len(classes)), dtype=np.int64)
            for method in REPRESENTATIONS
        }
        for payload in payloads:
            methods = payload["resolutions"][resolution]["methods"]
            for method in REPRESENTATIONS:
                confusion[method] += np.asarray(
                    methods[method]["overall"]["confusion_matrix"], dtype=np.int64
                )
                patient_rows[method].update(methods[method]["patients"])
        patients = sorted(patient_rows[REPRESENTATIONS[0]])
        method_summary = {}
        for method in REPRESENTATIONS:
            rows = patient_rows[method]
            method_summary[method] = {
                metric: _mean_or_none(
                    [rows[patient][metric] for patient in patients]
                )
                for metric in (
                    "accuracy",
                    "balanced_accuracy",
                    "macro_f1",
                    "macro_auprc",
                    "t_subtype_macro_auprc",
                    "rare_t_macro_auprc",
                )
            }
            method_summary[method]["per_class"] = {
                label: {
                    metric: _mean_or_none(
                        [
                            rows[patient]["per_class"][label][metric]
                            for patient in patients
                        ]
                    )
                    for metric in ("recall", "precision", "auprc")
                }
                for label in classes
            }
            method_summary[method]["pooled_confusion_matrix"] = confusion[
                method
            ].tolist()

        comparisons = {
            "h_plus_x_gate_minus_h_only": (
                "translated_y_h_plus_x_gate",
                "translated_y_h_only",
            ),
            "h_plus_x_ungated_minus_h_only": (
                "translated_y_h_plus_x_ungated",
                "translated_y_h_only",
            ),
            "h_plus_x_ungated_minus_h_residual": (
                "translated_y_h_plus_x_ungated",
                "translated_y_h_residual_ungated",
            ),
            "h_plus_x_ungated_minus_shuffled_x": (
                "translated_y_h_plus_x_ungated",
                "translated_y_h_shuffled_x_ungated",
            ),
        }
        comparison_summary = {}
        for name, (left, right) in comparisons.items():
            comparison_summary[name] = {}
            scalar_metrics = (
                "balanced_accuracy",
                "macro_f1",
                "macro_auprc",
                "t_subtype_macro_auprc",
                "rare_t_macro_auprc",
            )
            for metric in scalar_metrics:
                paired_values = [
                    patient_rows[left][patient][metric]
                    - patient_rows[right][patient][metric]
                    for patient in patients
                    if patient_rows[left][patient][metric] is not None
                    and patient_rows[right][patient][metric] is not None
                ]
                if not paired_values:
                    comparison_summary[name][metric] = {
                        "delta": None,
                        "patient_bootstrap_95ci": None,
                        "patients": 0,
                    }
                    continue
                paired = np.asarray(
                    [
                        value for value in paired_values
                    ],
                    dtype=float,
                )
                boot = [
                    float(np.mean(rng.choice(paired, paired.size, replace=True)))
                    for _ in range(int(bootstrap_replicates))
                ]
                comparison_summary[name][metric] = {
                    "delta": float(np.mean(paired)),
                    "patient_bootstrap_95ci": _interval(boot),
                    "patients": int(paired.size),
                }
            comparison_summary[name]["rare_classes"] = {}
            for label in RARE_T_CELL_TYPES:
                if label not in classes:
                    continue
                comparison_summary[name]["rare_classes"][label] = {}
                for metric in ("auprc", "recall"):
                    paired_values = [
                        patient_rows[left][patient]["per_class"][label][metric]
                        - patient_rows[right][patient]["per_class"][label][metric]
                        for patient in patients
                        if patient_rows[left][patient]["per_class"][label][metric]
                        is not None
                        and patient_rows[right][patient]["per_class"][label][metric]
                        is not None
                    ]
                    if not paired_values:
                        comparison_summary[name]["rare_classes"][label][metric] = {
                            "delta": None,
                            "patient_bootstrap_95ci": None,
                            "patients": 0,
                        }
                        continue
                    paired = np.asarray(
                        [
                            value for value in paired_values
                        ],
                        dtype=float,
                    )
                    boot = [
                        float(
                            np.mean(rng.choice(paired, paired.size, replace=True))
                        )
                        for _ in range(int(bootstrap_replicates))
                    ]
                    comparison_summary[name]["rare_classes"][label][metric] = {
                        "delta": float(np.mean(paired)),
                        "patient_bootstrap_95ci": _interval(boot),
                        "patients": int(paired.size),
                    }
        output_resolutions[resolution] = {
            "classes": list(classes),
            "patients": len(patients),
            "methods": method_summary,
            "comparisons": comparison_summary,
        }

    result = {
        "status": "ok",
        "folds": folds,
        "representations": list(REPRESENTATIONS),
        "bootstrap": {
            "unit": "patient",
            "replicates": int(bootstrap_replicates),
            "seed": int(seed),
        },
        "resolutions": output_resolutions,
        "fold_results": [str(path) for path in paths],
    }
    output = root / "cell_type_probe_cv_summary.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    result["summary_path"] = str(output)
    return result
