"""Aggregate fold/seed metrics for an end-to-end experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def summarize_experiment(path: str | Path) -> dict:
    root = Path(path)
    runs = [
        json.loads(file.read_text())
        for file in sorted(root.glob("fold_*/seed_*/metrics.json"))
    ]
    if not runs:
        raise ValueError(f"No completed runs found below {root}")
    method_names = tuple(runs[0]["methods"])
    summary = {}

    def standard_deviation(values: np.ndarray) -> float:
        return float(values.std(ddof=1)) if values.size > 1 else 0.0

    for method in method_names:
        wasserstein = np.asarray(
            [
                run["methods"][method]["test"]["patient_first_normalized_wasserstein"]
                for run in runs
            ]
        )
        median_error = np.asarray(
            [
                run["methods"][method]["test"]["patient_first_normalized_median_error"]
                for run in runs
            ]
        )
        summary[method] = {
            "wasserstein_mean": float(wasserstein.mean()),
            "wasserstein_std": standard_deviation(wasserstein),
            "median_error_mean": float(median_error.mean()),
            "median_error_std": standard_deviation(median_error),
        }

    comparisons = ("ridge_hl", "knn_hl", "mlp_hl", "ot_hl")
    wins = {
        baseline: sum(
            run["methods"]["cytoalign"]["test"]["patient_first_normalized_wasserstein"]
            < run["methods"][baseline]["test"]["patient_first_normalized_wasserstein"]
            for run in runs
        )
        for baseline in comparisons
    }
    result = {
        "status": "ok",
        "experiment": root.name,
        "n_runs": len(runs),
        "methods": summary,
        "cytoalign_wins": wins,
        "cytoalign_beats_all_baseline_means": all(
            summary["cytoalign"]["wasserstein_mean"]
            < summary[baseline]["wasserstein_mean"]
            for baseline in comparisons
        ),
        "source_x_adds_value_over_ot_hl": (
            summary["cytoalign"]["wasserstein_mean"]
            < summary["ot_hl"]["wasserstein_mean"]
        ),
    }
    (root / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result
