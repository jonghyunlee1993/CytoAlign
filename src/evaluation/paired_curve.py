"""Aggregate paired-specimen dose-response experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _mean_std(values) -> dict:
    values = np.asarray(list(values), dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "values": values.tolist(),
    }


def _linear_fit(x, y) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    denominator = np.sum((y - y.mean()) ** 2)
    r_squared = 1.0 - np.sum((y - fitted) ** 2) / denominator if denominator else 1.0
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
    }


def summarize_paired_curve(path: str | Path) -> dict:
    root = Path(path)
    runs = [
        json.loads(file.read_text())
        for file in sorted(root.glob("fold_0/seed_*/metrics.json"))
    ]
    if not runs:
        raise ValueError(f"No completed paired-curve runs found below {root}")
    counts = sorted(map(int, runs[0]["paired_curve"]))
    baseline_names = tuple(runs[0]["shared_methods"])
    baselines = {
        name: _mean_std(
            run["shared_methods"][name]["test"]["patient_first_normalized_wasserstein"]
            for run in runs
        )
        for name in baseline_names
    }
    curve = {}
    for count in counts:
        key = str(count)
        curve[key] = {
            method: _mean_std(
                run["paired_curve"][key]["methods"][method]["test"][
                    "patient_first_normalized_wasserstein"
                ]
                for run in runs
            )
            for method in ("ot_hl", "cytoalign")
        }
        curve[key]["selected_alpha"] = [
            run["paired_curve"][key]["methods"]["cytoalign"]["selected_alpha"]
            for run in runs
        ]
        curve[key]["cytoalign_minus_ot_hl"] = _mean_std(
            run["paired_curve"][key]["methods"]["cytoalign"]["test"][
                "patient_first_normalized_wasserstein"
            ]
            - run["paired_curve"][key]["methods"]["ot_hl"]["test"][
                "patient_first_normalized_wasserstein"
            ]
            for run in runs
        )

    mean_wasserstein = np.asarray(
        [curve[str(count)]["cytoalign"]["mean"] for count in counts]
    )
    improvement = mean_wasserstein[0] - mean_wasserstein
    positive_counts = np.asarray([count for count in counts if count > 0])
    positive_improvement = np.asarray(
        [improvement[index] for index, count in enumerate(counts) if count > 0]
    )
    nested = all(
        all(
            set(run["paired_curve"][str(left)]["paired_specimens"])
            <= set(run["paired_curve"][str(right)]["paired_specimens"])
            for left, right in zip(counts, counts[1:])
        )
        for run in runs
    )
    first_count_beating_baseline = {
        name: next(
            (
                count
                for count in counts
                if curve[str(count)]["cytoalign"]["mean"] < values["mean"]
            ),
            None,
        )
        for name, values in baselines.items()
    }
    first_count_showing_x_value = next(
        (
            count
            for count in counts
            if curve[str(count)]["cytoalign_minus_ot_hl"]["mean"] < 0
        ),
        None,
    )
    result = {
        "status": "ok",
        "experiment": root.name,
        "fold": 0,
        "n_seeds": len(runs),
        "residual_baseline": runs[0].get("residual_baseline", "ridge_hl"),
        "paired_counts": counts,
        "paired_sets_are_nested": nested,
        "baselines": baselines,
        "curve": curve,
        "first_count_beating_baseline": first_count_beating_baseline,
        "first_count_showing_x_value": first_count_showing_x_value,
        "trend": {
            "improvement_over_zero": improvement.tolist(),
            "raw_count_linear_fit": _linear_fit(counts, improvement),
            "doubling_linear_fit": _linear_fit(
                np.log2(positive_counts), positive_improvement
            ),
            "monotonic_improvement_steps": int(np.sum(np.diff(mean_wasserstein) <= 0)),
            "total_steps": len(counts) - 1,
        },
    }
    (root / "paired_curve_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result
