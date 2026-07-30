"""Same-cell pseudo-masking benchmark for marker and label recoverability."""

from __future__ import annotations

import hashlib
import json
import os
import random
import socket
import time
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.benchmark.self_recoverability_cache import load_cached_specimen
from src.data.markers import canonical_marker_name
from src.data.splits import patient_id_from_specimen
from src.models.gpu_knn import TorchKNNMedianRegressor
from src.models.mlp import MLPRegressor


PLATFORM_FINE_CLASSES: dict[str, tuple[str, ...]] = {
    "spectral_flow": (
        "Blast",
        "Monocyte",
        "T cell CD4",
        "T cell CD8",
        "T cell DN",
        "T cell DP",
        "B cell",
        "NK cell",
    ),
    "cytof": (
        "Blast",
        "Monocyte",
        "T cell CD4",
        "T cell CD8",
        "T cell DN",
        "T cell gd",
        "B cell",
        "NK cell",
    ),
}


def _digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: Mapping, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _hardware(device: str) -> dict:
    import torch

    result = {
        "host": socket.gethostname(),
        "device_request": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        probe = torch.randn(1024, 1024, device=device) @ torch.randn(
            1024, 1024, device=device
        )
        torch.cuda.synchronize()
        if not torch.isfinite(probe).all():
            raise RuntimeError("CUDA matrix-multiplication probe failed")
        properties = torch.cuda.get_device_properties(torch.device(device))
        result.update(
            {
                "gpu": properties.name,
                "gpu_memory_bytes": int(properties.total_memory),
                "cuda_probe": "passed",
            }
        )
        del probe
    return result


def _canonical_lookup(markers: Sequence[str]) -> dict[str, int]:
    canonical = [canonical_marker_name(marker) for marker in markers]
    duplicates = sorted(
        {marker for marker in canonical if canonical.count(marker) > 1}
    )
    if duplicates:
        raise ValueError(f"Duplicate canonical markers: {duplicates}")
    return {marker: index for index, marker in enumerate(canonical)}


def panel_indices(
    full_markers: Sequence[str],
    observed_markers: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]:
    """Resolve an observed panel and its pseudo-unobserved complement."""

    full = tuple(map(str, full_markers))
    lookup = _canonical_lookup(full)
    observed_canonical = tuple(
        canonical_marker_name(marker) for marker in observed_markers
    )
    if len(set(observed_canonical)) != len(observed_canonical):
        raise ValueError("Observed panel contains duplicate canonical markers")
    missing = sorted(set(observed_canonical) - set(lookup))
    if missing:
        raise ValueError(f"Observed panel markers are unavailable: {missing}")
    observed_indices = np.asarray(
        [lookup[marker] for marker in observed_canonical], dtype=np.int64
    )
    observed_set = set(observed_indices.tolist())
    hidden_indices = np.asarray(
        [index for index in range(len(full)) if index not in observed_set],
        dtype=np.int64,
    )
    if hidden_indices.size == 0:
        raise ValueError("Observed panel leaves no pseudo-unobserved markers")
    return (
        observed_indices,
        hidden_indices,
        tuple(full[index] for index in observed_indices),
        tuple(full[index] for index in hidden_indices),
    )


def _sample_indices(
    n_rows: int,
    maximum: int,
    *,
    seed: int,
) -> np.ndarray:
    if n_rows <= int(maximum):
        return np.arange(n_rows, dtype=np.int64)
    return np.sort(
        np.random.RandomState(int(seed)).choice(
            n_rows,
            int(maximum),
            replace=False,
        )
    )


def _load_patient_balanced_training(
    *,
    cache_root: str | Path,
    modality: str,
    specimens: Sequence[str],
    cells_per_patient: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    patient_to_specimens: dict[str, list[str]] = defaultdict(list)
    for specimen in specimens:
        patient_to_specimens[patient_id_from_specimen(specimen)].append(str(specimen))
    value_pieces = []
    label_pieces = []
    patient_pieces = []
    expected_markers = None
    for patient in sorted(patient_to_specimens):
        current_values = []
        current_labels = []
        for specimen in sorted(patient_to_specimens[patient]):
            cached = load_cached_specimen(cache_root, modality, specimen)
            if expected_markers is None:
                expected_markers = cached["markers"]
            elif expected_markers != cached["markers"]:
                raise ValueError(f"Cached marker mismatch for {modality}/{specimen}")
            current_values.append(cached["values"])
            current_labels.append(cached["labels"])
        values = np.concatenate(current_values)
        labels = np.concatenate(current_labels)
        rows = _sample_indices(
            len(values),
            int(cells_per_patient),
            seed=(
                int(seed)
                + int.from_bytes(
                    hashlib.sha256(patient.encode("utf-8")).digest()[:4],
                    "little",
                )
            )
            % (2**32),
        )
        value_pieces.append(values[rows])
        label_pieces.append(labels[rows])
        patient_pieces.append(np.repeat(patient, len(rows)))
    if expected_markers is None:
        raise ValueError("No training specimens were loaded")
    return (
        np.concatenate(value_pieces).astype(np.float32, copy=False),
        np.concatenate(label_pieces).astype(str),
        np.concatenate(patient_pieces).astype(str),
        tuple(expected_markers),
    )


def _patient_balanced_bank(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    maximum: int,
    seed: int,
) -> np.ndarray:
    if len(values) <= int(maximum):
        return np.arange(len(values), dtype=np.int64)
    unique = np.unique(groups)
    quota = max(1, int(np.ceil(int(maximum) / len(unique))))
    rng = np.random.RandomState(int(seed))
    pieces = []
    for group in unique:
        candidates = np.flatnonzero(groups == group)
        if len(candidates) > quota:
            candidates = rng.choice(candidates, quota, replace=False)
        pieces.append(candidates)
    selected = np.concatenate(pieces)
    if len(selected) > int(maximum):
        selected = rng.choice(selected, int(maximum), replace=False)
    return np.sort(selected.astype(np.int64))


def _class_patient_balanced_rows(
    labels: np.ndarray,
    patients: np.ndarray,
    classes: Sequence[str],
    *,
    maximum_per_class: int,
    seed: int,
) -> np.ndarray:
    selected = []
    for class_index, label in enumerate(classes):
        candidates = np.flatnonzero(labels == label)
        if not len(candidates):
            continue
        local = _patient_balanced_bank(
            np.empty((len(candidates), 0)),
            patients[candidates],
            maximum=int(maximum_per_class),
            seed=int(seed) + 1009 * class_index,
        )
        selected.append(candidates[local])
    if not selected:
        raise ValueError("No configured label classes occur in training data")
    return np.sort(np.concatenate(selected))


def _robust_location_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(values, axis=0)
    q25, q75 = np.quantile(values, (0.25, 0.75), axis=0)
    scale = q75 - q25
    mad = np.median(np.abs(values - median[None, :]), axis=0)
    scale = np.where(scale > 1.0e-6, scale, 1.4826 * mad)
    scale = np.where(scale > 1.0e-6, scale, 1.0)
    return median.astype(np.float32), scale.astype(np.float32)


def _scale(
    values: np.ndarray,
    location: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return np.asarray((values - location[None, :]) / scale[None, :], dtype=np.float32)


def _classifier(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    c_value: float,
    maximum_iterations: int,
    seed: int,
):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(c_value),
            class_weight="balanced",
            max_iter=int(maximum_iterations),
            solver="lbfgs",
            random_state=int(seed),
        ),
    ).fit(features, labels)


def _aligned_probabilities(model, features: np.ndarray, classes: Sequence[str]):
    raw = model.predict_proba(features)
    fitted = model[-1].classes_.astype(str)
    lookup = {label: index for index, label in enumerate(fitted)}
    aligned = np.zeros((len(features), len(classes)), dtype=np.float64)
    for index, label in enumerate(classes):
        if label in lookup:
            aligned[:, index] = raw[:, lookup[label]]
    return model.predict(features).astype(str), aligned


def biology_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    classes: Sequence[str],
) -> dict:
    """Imbalance-aware fixed-classifier metrics for one specimen."""

    labels = np.asarray(labels).astype(str)
    predictions = np.asarray(predictions).astype(str)
    classes = tuple(map(str, classes))
    matrix = confusion_matrix(labels, predictions, labels=classes)
    support = matrix.sum(axis=1)
    predicted_support = matrix.sum(axis=0)
    recalls = np.divide(
        np.diag(matrix),
        support,
        out=np.zeros(len(classes), dtype=float),
        where=support > 0,
    )
    precisions = np.divide(
        np.diag(matrix),
        predicted_support,
        out=np.zeros(len(classes), dtype=float),
        where=predicted_support > 0,
    )
    per_class = {}
    finite_auprc = []
    for index, label in enumerate(classes):
        binary = labels == label
        auprc = (
            float(average_precision_score(binary, probabilities[:, index]))
            if np.any(binary)
            else None
        )
        if auprc is not None:
            finite_auprc.append(auprc)
        observed_prevalence = float(np.mean(binary))
        predicted_prevalence = float(np.mean(predictions == label))
        per_class[label] = {
            "support": int(support[index]),
            "predicted_support": int(predicted_support[index]),
            "recall": float(recalls[index]) if support[index] else None,
            "precision": (
                float(precisions[index]) if predicted_support[index] else None
            ),
            "auprc": auprc,
            "observed_prevalence": observed_prevalence,
            "predicted_prevalence": predicted_prevalence,
            "prevalence_error": predicted_prevalence - observed_prevalence,
        }
    present = support > 0
    f1_values = np.divide(
        2.0 * precisions * recalls,
        precisions + recalls,
        out=np.zeros(len(classes), dtype=float),
        where=(precisions + recalls) > 0,
    )
    return {
        "accuracy": float(np.mean(labels == predictions)),
        "balanced_accuracy": float(np.mean(recalls[present])) if np.any(present) else None,
        "macro_f1": float(np.mean(f1_values[present])) if np.any(present) else None,
        "macro_auprc": (
            float(np.mean(finite_auprc)) if finite_auprc else None
        ),
        "classes": list(classes),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }


def marker_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    median_prediction: np.ndarray,
    scales: np.ndarray,
    marker_names: Sequence[str],
) -> list[dict]:
    """Return exact same-cell and dynamic-range metrics marker by marker."""

    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    median_prediction = np.asarray(median_prediction, dtype=np.float64)
    output = []
    for index, marker in enumerate(marker_names):
        observed = truth[:, index]
        predicted = prediction[:, index]
        null = median_prediction[:, index]
        mae = float(np.mean(np.abs(predicted - observed)))
        null_mae = float(np.mean(np.abs(null - observed)))
        if np.ptp(observed) <= 0 or np.ptp(predicted) <= 0:
            correlation = None
        else:
            value = spearmanr(observed, predicted).statistic
            correlation = float(value) if np.isfinite(value) else None
        observed_iqr = float(np.quantile(observed, 0.75) - np.quantile(observed, 0.25))
        predicted_iqr = float(
            np.quantile(predicted, 0.75) - np.quantile(predicted, 0.25)
        )
        output.append(
            {
                "marker": str(marker),
                "mae": mae,
                "normalized_mae": mae / float(scales[index]),
                "null_mae": null_mae,
                "null_relative_skill": (
                    1.0 - mae / null_mae if null_mae > 0 else None
                ),
                "spearman": correlation,
                "observed_iqr": observed_iqr,
                "predicted_iqr": predicted_iqr,
                "dynamic_range_retention": (
                    predicted_iqr / observed_iqr if observed_iqr > 0 else None
                ),
                "normalized_median_error": (
                    abs(float(np.median(predicted) - np.median(observed)))
                    / float(scales[index])
                ),
                "normalized_q90_error": (
                    abs(float(np.quantile(predicted, 0.90) - np.quantile(observed, 0.90)))
                    / float(scales[index])
                ),
            }
        )
    return output


def _representation_biology(
    *,
    full_model,
    hidden_model,
    observed_model,
    full_values: np.ndarray,
    labels: np.ndarray,
    observed_indices: np.ndarray,
    hidden_indices: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    classes: Sequence[str],
) -> dict:
    observed = full_values[:, observed_indices]
    hidden = full_values[:, hidden_indices]
    output = {}

    def evaluate(name: str, model, features: np.ndarray):
        predicted, probabilities = _aligned_probabilities(model, features, classes)
        output[name] = biology_metrics(
            labels,
            predicted,
            probabilities,
            classes,
        )

    evaluate("full_true", full_model, full_values)
    evaluate("hidden_true", hidden_model, hidden)
    evaluate("observed_true", observed_model, observed)
    for name, current in predictions.items():
        hybrid = np.asarray(full_values, dtype=np.float32).copy()
        hybrid[:, hidden_indices] = current
        evaluate(f"full_hybrid_{name}", full_model, hybrid)
        evaluate(f"hidden_{name}", hidden_model, current)
    return output


def run_self_recoverability(
    config: dict,
    *,
    modality: str,
    panel_name: str,
    fold_index: int,
    seed: int,
) -> dict:
    """Run one modality/panel/fold/seed same-cell pseudo-masking experiment."""

    import torch

    started = time.time()
    modality = str(modality)
    panel_name = str(panel_name)
    if modality not in PLATFORM_FINE_CLASSES:
        raise ValueError(f"Unsupported modality: {modality}")
    panel_config = config["panels"].get(panel_name)
    if panel_config is None:
        raise ValueError(f"Unknown panel: {panel_name}")
    device = str(config["training"]["device"])
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    hardware = _hardware(device)

    split_path = Path(config["data"]["split_manifest"])
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    fold = next(
        (
            current
            for current in split_manifest["folds"]
            if int(current["fold_index"]) == int(fold_index)
        ),
        None,
    )
    if fold is None:
        raise ValueError(f"Fold {fold_index} is unavailable")
    fit_specimens = tuple(
        list(fold["train_specimens"]) + list(fold["validation_specimens"])
    )
    test_specimens = tuple(fold["test_specimens"])
    fit_values, fit_labels, fit_patients, full_markers = (
        _load_patient_balanced_training(
            cache_root=config["data"]["cache_root"],
            modality=modality,
            specimens=fit_specimens,
            cells_per_patient=int(config["training"]["cells_per_fit_patient"]),
            seed=int(config["training"]["fit_sample_seed"]),
        )
    )
    (
        observed_indices,
        hidden_indices,
        observed_markers,
        hidden_markers,
    ) = panel_indices(full_markers, panel_config["markers"])
    observed_fit = fit_values[:, observed_indices]
    hidden_fit = fit_values[:, hidden_indices]
    observed_location, observed_scale = _robust_location_scale(observed_fit)
    hidden_median, hidden_scale = _robust_location_scale(hidden_fit)
    scaled_observed_fit = _scale(
        observed_fit,
        observed_location,
        observed_scale,
    )

    knn_config = config["training"]["knn"]
    bank_rows = _patient_balanced_bank(
        scaled_observed_fit,
        fit_patients,
        maximum=int(knn_config["max_reference_cells"]),
        seed=int(config["training"]["fit_sample_seed"]),
    )
    knn = TorchKNNMedianRegressor(
        k=int(knn_config["k"]),
        device=device,
        query_chunk_size=knn_config.get("query_chunk_size"),
        distance_memory_fraction=float(
            knn_config.get("distance_memory_fraction", 0.08)
        ),
    ).fit(
        scaled_observed_fit[bank_rows],
        hidden_fit[bank_rows],
    )

    mlp_config = config["training"]["mlp"]
    mlp = MLPRegressor(
        hidden_dims=mlp_config["hidden_dims"],
        epochs=int(mlp_config["epochs"]),
        batch_size=int(mlp_config["batch_size"]),
        learning_rate=float(mlp_config["learning_rate"]),
        patience=int(mlp_config["patience"]),
        random_state=int(seed),
    ).fit(
        scaled_observed_fit,
        hidden_fit,
        groups=fit_patients,
        device=device,
    )

    classes = PLATFORM_FINE_CLASSES[modality]
    unknown_fit = sorted(set(np.unique(fit_labels)) - set(classes))
    if unknown_fit:
        raise ValueError(f"Unexpected {modality} training labels: {unknown_fit}")
    classifier_config = config["classifier"]
    classifier_rows = _class_patient_balanced_rows(
        fit_labels,
        fit_patients,
        classes,
        maximum_per_class=int(classifier_config["maximum_cells_per_class"]),
        seed=int(classifier_config["sample_seed"]),
    )
    classifier_kwargs = {
        "c_value": float(classifier_config["c"]),
        "maximum_iterations": int(classifier_config["max_iter"]),
        "seed": int(classifier_config["seed"]),
    }
    full_classifier = _classifier(
        fit_values[classifier_rows],
        fit_labels[classifier_rows],
        **classifier_kwargs,
    )
    hidden_classifier = _classifier(
        hidden_fit[classifier_rows],
        fit_labels[classifier_rows],
        **classifier_kwargs,
    )
    observed_classifier = _classifier(
        observed_fit[classifier_rows],
        fit_labels[classifier_rows],
        **classifier_kwargs,
    )

    marker_rows = []
    biology_by_specimen = {}
    for specimen_index, specimen in enumerate(test_specimens):
        cached = load_cached_specimen(
            config["data"]["cache_root"],
            modality,
            specimen,
        )
        if cached["markers"] != full_markers:
            raise ValueError(f"Test marker mismatch for {modality}/{specimen}")
        values = cached["values"]
        labels = cached["labels"]
        unknown_test = sorted(set(np.unique(labels)) - set(classes))
        if unknown_test:
            raise ValueError(f"Unexpected {modality} test labels: {unknown_test}")
        observed = values[:, observed_indices]
        hidden = values[:, hidden_indices]
        scaled_observed = _scale(
            observed,
            observed_location,
            observed_scale,
        )
        permutation = np.random.RandomState(
            int(config["evaluation"]["shuffle_seed"]) + specimen_index
        ).permutation(len(values))
        scaled_shuffled = scaled_observed[permutation]
        predictions = {
            "median": np.repeat(
                hidden_median[None, :],
                len(values),
                axis=0,
            ).astype(np.float32),
            "knn": knn.predict(scaled_observed),
            "mlp": mlp.predict(scaled_observed, device=device),
            "knn_shuffled_h": knn.predict(scaled_shuffled),
            "mlp_shuffled_h": mlp.predict(scaled_shuffled, device=device),
        }
        for representation, prediction in predictions.items():
            current_rows = marker_metrics(
                hidden,
                prediction,
                predictions["median"],
                hidden_scale,
                hidden_markers,
            )
            for row in current_rows:
                row.update(
                    {
                        "modality": modality,
                        "panel": panel_name,
                        "fold": int(fold_index),
                        "seed": int(seed),
                        "patient": patient_id_from_specimen(specimen),
                        "specimen": specimen,
                        "representation": representation,
                        "n_cells": int(len(values)),
                    }
                )
            marker_rows.extend(current_rows)
        biology_by_specimen[specimen] = {
            "patient": patient_id_from_specimen(specimen),
            "n_cells": int(len(values)),
            "metrics": _representation_biology(
                full_model=full_classifier,
                hidden_model=hidden_classifier,
                observed_model=observed_classifier,
                full_values=values,
                labels=labels,
                observed_indices=observed_indices,
                hidden_indices=hidden_indices,
                predictions=predictions,
                classes=classes,
            ),
        }

    output = (
        Path(config["output"]["root"])
        / str(config["experiment"]["name"])
        / modality
        / panel_name
        / f"fold_{int(fold_index)}"
        / f"seed_{int(seed)}"
    )
    output.mkdir(parents=True, exist_ok=True)
    marker_path = output / "marker_metrics.csv"
    marker_frame = pd.DataFrame(marker_rows)
    marker_frame.to_csv(marker_path, index=False)
    biology_path = output / "biology_metrics.json"
    _atomic_json(
        {
            "classes": list(classes),
            "classifier": {
                "type": "fixed_class_balanced_multinomial_logistic_regression",
                "trained_on": "outer_fit_true_markers_and_existing_labels",
                "representation_specific_refit": False,
                "training_cells": int(len(classifier_rows)),
            },
            "specimens": biology_by_specimen,
        },
        biology_path,
    )
    result = {
        "status": "ok",
        "experiment": str(config["experiment"]["name"]),
        "modality": modality,
        "panel": panel_name,
        "fold": int(fold_index),
        "seed": int(seed),
        "claim_scope": "processed_upstream_pregated_conditional_sensitivity",
        "split_manifest": str(split_path),
        "split_manifest_sha256": _digest(split_path),
        "fit_patients": int(len(np.unique(fit_patients))),
        "fit_cells": int(len(fit_values)),
        "test_patients": int(len(fold["test_patients"])),
        "test_specimens": int(len(test_specimens)),
        "observed_markers": list(observed_markers),
        "hidden_markers": list(hidden_markers),
        "knn_reference_cells": knn.reference_size,
        "mlp_best_validation_loss": float(mlp.best_validation_loss_),
        "hardware": hardware,
        "artifacts": {
            "marker_metrics": str(marker_path),
            "biology_metrics": str(biology_path),
        },
        "elapsed_seconds": time.time() - started,
    }
    result_path = output / "run_summary.json"
    result["artifacts"]["run_summary"] = str(result_path)
    _atomic_json(result, result_path)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        result["hardware"]["peak_gpu_memory_bytes"] = int(
            torch.cuda.max_memory_allocated(device)
        )
        _atomic_json(result, result_path)
        torch.cuda.empty_cache()
    return result
