"""Lightweight diagnostics for uncertainty-gated residual correction."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr, wasserstein_distance
from sklearn.metrics import roc_auc_score

from src.models.cytofmerge import CyTOFMergeDiagnostics


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    value = spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else None


def _auc(score: np.ndarray, outcome: np.ndarray) -> float | None:
    if np.unique(outcome).size < 2:
        return None
    value = roc_auc_score(outcome, score)
    return float(value) if np.isfinite(value) else None


def _quintiles(signal: np.ndarray, values: Mapping[str, np.ndarray]) -> list[dict]:
    order = np.argsort(signal)
    result = []
    for index, rows in enumerate(np.array_split(order, 5), start=1):
        if rows.size == 0:
            continue
        point = {
            "quintile": index,
            "n": int(rows.size),
            "uncertainty_mean": float(np.mean(signal[rows])),
        }
        point.update({name: float(np.mean(value[rows])) for name, value in values.items()})
        result.append(point)
    return result


def evaluate_uncertainty(
    baseline: Mapping[str, np.ndarray],
    residual: Mapping[str, np.ndarray],
    diagnostics: Mapping[str, CyTOFMergeDiagnostics],
    view: dict,
    marker_scales: np.ndarray,
    alphas: Sequence[float],
    selected_alpha: float,
    marker_names: Sequence[str],
    *,
    minimum_cells: int = 5,
) -> dict:
    """Relate kNN uncertainty to population error and correction headroom.

    A unit is one specimen, cell type, and target marker. The best alpha is an
    intentionally test-oracle diagnostic; it estimates whether a selective gate
    has useful headroom and is not deployable performance.
    """

    scales = np.asarray(marker_scales, dtype=float)
    alpha_grid = np.asarray(tuple(map(float, alphas)), dtype=float)
    if not np.any(alpha_grid == 0):
        raise ValueError("alphas must contain 0")
    records: dict[str, list] = defaultdict(list)
    for specimen in sorted(baseline):
        source_labels = np.asarray(view["source_labels"][specimen]).astype(str)
        target_labels = np.asarray(view["target_labels"][specimen]).astype(str)
        target = np.asarray(view["target_y"][specimen], dtype=float)
        current_baseline = np.asarray(baseline[specimen], dtype=float)
        current_residual = np.asarray(residual[specimen], dtype=float)
        current_diagnostics = diagnostics[specimen]
        for label in sorted(set(source_labels) & set(target_labels)):
            source_rows = source_labels == label
            target_rows = target_labels == label
            if source_rows.sum() < minimum_cells or target_rows.sum() < minimum_cells:
                continue
            support = float(
                np.mean(current_diagnostics.mean_neighbor_distance[source_rows])
            )
            for marker in range(scales.size):
                scale = scales[marker]
                target_values = target[target_rows, marker]
                baseline_values = current_baseline[source_rows, marker]
                residual_values = current_residual[source_rows, marker]
                errors = np.asarray(
                    [
                        wasserstein_distance(
                            baseline_values + alpha * residual_values, target_values
                        )
                        / scale
                        for alpha in alpha_grid
                    ]
                )
                baseline_error = float(errors[alpha_grid == 0.0][0])
                best_index = int(np.argmin(errors))
                selected_index = int(np.argmin(np.abs(alpha_grid - selected_alpha)))
                records["marker"].append(marker)
                records["support"].append(support)
                records["ambiguity"].append(
                    float(
                        np.mean(
                            current_diagnostics.median_neighbor_mad[
                                source_rows, marker
                            ]
                        )
                        / scale
                    )
                )
                records["baseline_error"].append(baseline_error)
                records["oracle_gain"].append(baseline_error - float(errors[best_index]))
                records["selected_gain"].append(
                    baseline_error - float(errors[selected_index])
                )
                records["oracle_active"].append(float(alpha_grid[best_index] > 0))
                records["oracle_alpha"].append(float(alpha_grid[best_index]))

    arrays = {name: np.asarray(values) for name, values in records.items()}
    active = arrays["oracle_active"].astype(bool)
    correlations = {}
    for uncertainty in ("support", "ambiguity"):
        correlations[uncertainty] = {
            outcome: _spearman(arrays[uncertainty], arrays[outcome])
            for outcome in ("baseline_error", "oracle_gain", "selected_gain")
        }
        correlations[uncertainty]["oracle_active_auc"] = _auc(
            arrays[uncertainty], active
        )

    marker_summary = {}
    for marker, name in enumerate(marker_names):
        rows = arrays["marker"] == marker
        marker_summary[str(name)] = {
            "n": int(rows.sum()),
            "baseline_error": float(np.mean(arrays["baseline_error"][rows])),
            "oracle_gain": float(np.mean(arrays["oracle_gain"][rows])),
            "selected_gain": float(np.mean(arrays["selected_gain"][rows])),
            "oracle_active_fraction": float(np.mean(active[rows])),
            "mean_oracle_alpha": float(np.mean(arrays["oracle_alpha"][rows])),
        }

    quintile_values = {
        "baseline_error": arrays["baseline_error"],
        "oracle_gain": arrays["oracle_gain"],
        "selected_gain": arrays["selected_gain"],
        "oracle_active_fraction": arrays["oracle_active"],
    }
    return {
        "unit": "specimen_cell_type_marker",
        "n_units": int(arrays["baseline_error"].size),
        "selected_alpha": float(selected_alpha),
        "mean_baseline_error": float(np.mean(arrays["baseline_error"])),
        "mean_selected_error": float(
            np.mean(arrays["baseline_error"] - arrays["selected_gain"])
        ),
        "mean_oracle_error": float(
            np.mean(arrays["baseline_error"] - arrays["oracle_gain"])
        ),
        "mean_selected_gain": float(np.mean(arrays["selected_gain"])),
        "mean_oracle_gain": float(np.mean(arrays["oracle_gain"])),
        "oracle_active_fraction": float(np.mean(active)),
        "correlations": correlations,
        "quintiles": {
            uncertainty: _quintiles(arrays[uncertainty], quintile_values)
            for uncertainty in ("support", "ambiguity")
        },
        "markers": marker_summary,
    }
