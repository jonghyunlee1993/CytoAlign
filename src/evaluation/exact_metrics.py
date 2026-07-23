"""Patient-first exact-cell metrics for two-sided pseudo panels."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr


def _validated_scales(marker_scales: np.ndarray) -> np.ndarray:
    scales = np.asarray(marker_scales, dtype=np.float64)
    if scales.ndim != 1 or np.any(scales <= 0) or not np.isfinite(scales).all():
        raise ValueError("marker_scales must be a finite positive vector")
    return scales


def _aligned(
    predictions: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    scales: np.ndarray,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    specimens = sorted(set(predictions) & set(targets))
    if not specimens:
        raise ValueError("No aligned exact-truth specimens were supplied")
    result = []
    for specimen in specimens:
        prediction = np.asarray(predictions[specimen], dtype=np.float64)
        target = np.asarray(targets[specimen], dtype=np.float64)
        if prediction.shape != target.shape or prediction.ndim != 2:
            raise ValueError(f"Prediction/target shape mismatch for {specimen}")
        if prediction.shape[1] != scales.size:
            raise ValueError("Marker dimensions do not align with marker_scales")
        if not np.isfinite(prediction).all() or not np.isfinite(target).all():
            raise ValueError("Exact-truth arrays contain non-finite values")
        result.append((specimen, prediction, target))
    return result


def _patient_first(
    specimen_values: Mapping[str, float], patient_by_specimen: Mapping[str, str]
) -> tuple[float, dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for specimen, value in specimen_values.items():
        grouped[str(patient_by_specimen[specimen])].append(float(value))
    collapsed = {
        patient: float(np.mean(values)) for patient, values in grouped.items()
    }
    return float(np.mean(list(collapsed.values()))), collapsed


def patient_first_normalized_mae(
    predictions: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    patient_by_specimen: Mapping[str, str],
    marker_scales: np.ndarray,
) -> float:
    """Fast primary metric used for validation-time residual-gate selection."""

    scales = _validated_scales(marker_scales)
    specimen = {
        name: float(np.mean(np.abs(prediction - target) / scales[None, :]))
        for name, prediction, target in _aligned(predictions, targets, scales)
    }
    return _patient_first(specimen, patient_by_specimen)[0]


def evaluate_exact_cells(
    predictions: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    patient_by_specimen: Mapping[str, str],
    marker_scales: np.ndarray,
) -> dict:
    """Evaluate aligned pseudo-panel cells with patients as the top-level unit."""

    scales = _validated_scales(marker_scales)
    aligned = _aligned(predictions, targets, scales)
    specimen_mae: dict[str, float] = {}
    specimen_rmse: dict[str, float] = {}
    marker_patient_correlations: list[dict[str, list[float]]] = [
        defaultdict(list) for _ in range(scales.size)
    ]
    for specimen, prediction, target in aligned:
        difference = prediction - target
        specimen_mae[specimen] = float(
            np.mean(np.abs(difference) / scales[None, :])
        )
        specimen_rmse[specimen] = float(
            np.mean(np.sqrt(np.mean(difference**2, axis=0)) / scales)
        )
        patient = str(patient_by_specimen[specimen])
        for marker_index in range(scales.size):
            correlation = spearmanr(
                prediction[:, marker_index], target[:, marker_index]
            ).statistic
            if np.isfinite(correlation):
                marker_patient_correlations[marker_index][patient].append(
                    float(correlation)
                )
    mae, patient_mae = _patient_first(specimen_mae, patient_by_specimen)
    rmse, patient_rmse = _patient_first(specimen_rmse, patient_by_specimen)
    marker_correlations = []
    for grouped in marker_patient_correlations:
        patient_values = [
            float(np.mean(values)) for values in grouped.values() if values
        ]
        marker_correlations.append(
            float(np.mean(patient_values)) if patient_values else None
        )
    finite = [value for value in marker_correlations if value is not None]
    return {
        "patient_first_normalized_mae": mae,
        "patient_first_normalized_rmse": rmse,
        "macro_patient_marker_spearman": (
            float(np.mean(finite)) if finite else None
        ),
        "marker_patient_spearman": marker_correlations,
        "patient_normalized_mae": patient_mae,
        "patient_normalized_rmse": patient_rmse,
        "n_specimens": len(aligned),
        "n_patients": len(patient_mae),
        "n_cells": int(sum(prediction.shape[0] for _, prediction, _ in aligned)),
    }


def select_residual_alpha(
    baselines: Mapping[str, np.ndarray],
    residuals: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    patient_by_specimen: Mapping[str, str],
    marker_scales: np.ndarray,
    alphas: Sequence[float],
) -> tuple[float, dict[str, float]]:
    """Select an additive residual gate using only validation specimens."""

    candidates = tuple(float(value) for value in alphas)
    if not candidates or any(value < 0 or not np.isfinite(value) for value in candidates):
        raise ValueError("alphas must be a non-empty sequence of finite non-negative values")
    if set(baselines) != set(residuals):
        raise ValueError("baselines and residuals must contain the same specimens")
    scores = {}
    for alpha in candidates:
        prediction = {
            specimen: np.asarray(baselines[specimen])
            + alpha * np.asarray(residuals[specimen])
            for specimen in baselines
        }
        scores[str(alpha)] = patient_first_normalized_mae(
            prediction, targets, patient_by_specimen, marker_scales
        )
    selected = min(candidates, key=lambda value: (scores[str(value)], value))
    return float(selected), scores
