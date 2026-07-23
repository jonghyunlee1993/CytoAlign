"""Patient-first metrics for cell-unpaired matched-specimen populations."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr, wasserstein_distance


def evaluate_matched_populations(
    predictions: Mapping[str, np.ndarray],
    prediction_cell_types: Mapping[str, Sequence],
    targets: Mapping[str, np.ndarray],
    target_cell_types: Mapping[str, Sequence],
    patient_by_specimen: Mapping[str, str],
    marker_scales: np.ndarray,
    *,
    minimum_cells: int = 5,
) -> dict:
    """Evaluate distributions without constructing source-target cell pairs."""

    specimens = sorted(set(predictions) & set(targets))
    if not specimens:
        raise ValueError("No matched evaluation specimens were supplied")
    scales = np.asarray(marker_scales, dtype=np.float64)
    if scales.ndim != 1 or np.any(scales <= 0) or not np.isfinite(scales).all():
        raise ValueError("marker_scales must be a finite positive vector")
    if int(minimum_cells) < 1:
        raise ValueError("minimum_cells must be positive")

    specimen_wasserstein: dict[str, list[float]] = defaultdict(list)
    specimen_median_error: dict[str, list[float]] = defaultdict(list)
    predicted_medians: list[np.ndarray] = []
    target_medians: list[np.ndarray] = []
    n_strata = 0
    for specimen in specimens:
        prediction = np.asarray(predictions[specimen], dtype=np.float64)
        target = np.asarray(targets[specimen], dtype=np.float64)
        prediction_labels = np.asarray(prediction_cell_types[specimen]).astype(str)
        target_labels = np.asarray(target_cell_types[specimen]).astype(str)
        if prediction.ndim != 2 or target.ndim != 2:
            raise ValueError("Prediction and target populations must be matrices")
        if prediction.shape[1] != target.shape[1] or prediction.shape[1] != scales.size:
            raise ValueError("Population marker dimensions do not align")
        if (
            prediction.shape[0] != prediction_labels.size
            or target.shape[0] != target_labels.size
        ):
            raise ValueError("Population cell types do not align")
        common_types = sorted(set(prediction_labels) & set(target_labels))
        for label in common_types:
            predicted_rows = prediction_labels == label
            target_rows = target_labels == label
            if predicted_rows.sum() < int(minimum_cells) or target_rows.sum() < int(
                minimum_cells
            ):
                continue
            predicted_current = prediction[predicted_rows]
            target_current = target[target_rows]
            distances = [
                wasserstein_distance(
                    predicted_current[:, index], target_current[:, index]
                )
                / scales[index]
                for index in range(scales.size)
            ]
            predicted_median = np.median(predicted_current, axis=0)
            target_median = np.median(target_current, axis=0)
            specimen_wasserstein[specimen].append(float(np.mean(distances)))
            specimen_median_error[specimen].append(
                float(np.mean(np.abs(predicted_median - target_median) / scales))
            )
            predicted_medians.append(predicted_median)
            target_medians.append(target_median)
            n_strata += 1
    if not n_strata:
        raise ValueError("No cell-type strata met minimum_cells")

    def patient_first(
        values: Mapping[str, list[float]]
    ) -> tuple[float, dict[str, float]]:
        patient_values: dict[str, list[float]] = defaultdict(list)
        for specimen, current in values.items():
            if current:
                patient_values[str(patient_by_specimen[specimen])].append(
                    float(np.mean(current))
                )
        collapsed = {
            patient: float(np.mean(current))
            for patient, current in patient_values.items()
        }
        return float(np.mean(list(collapsed.values()))), collapsed

    wasserstein, patient_wasserstein = patient_first(specimen_wasserstein)
    median_error, patient_median_error = patient_first(specimen_median_error)
    predicted_summary = np.stack(predicted_medians)
    target_summary = np.stack(target_medians)
    marker_correlations = []
    for marker_index in range(scales.size):
        predicted_marker = predicted_summary[:, marker_index]
        target_marker = target_summary[:, marker_index]
        if np.ptp(predicted_marker) == 0 or np.ptp(target_marker) == 0:
            correlation = np.nan
        else:
            correlation = spearmanr(predicted_marker, target_marker).statistic
        marker_correlations.append(
            float(correlation) if np.isfinite(correlation) else None
        )
    finite = [value for value in marker_correlations if value is not None]
    return {
        "patient_first_normalized_wasserstein": wasserstein,
        "patient_first_normalized_median_error": median_error,
        "macro_marker_median_spearman": float(np.mean(finite)) if finite else None,
        "marker_median_spearman": marker_correlations,
        "n_specimens": len(specimen_wasserstein),
        "n_patients": len(patient_wasserstein),
        "n_cell_type_strata": n_strata,
        "patient_normalized_wasserstein": patient_wasserstein,
        "patient_normalized_median_error": patient_median_error,
    }
