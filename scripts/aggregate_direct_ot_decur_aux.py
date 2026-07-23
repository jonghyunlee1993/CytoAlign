#!/usr/bin/env python3
"""Aggregate the paired direct-OT DeCUR auxiliary ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


VARIANTS = ("direct_ot", "paired_decur_aux", "uniform_decur_aux")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def metric(method: dict, split: str, family: str, name: str) -> float:
    return float(method[split][family][name])


def summary(values) -> dict:
    array = np.asarray(list(values), dtype=float)
    return {
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=0)),
        "values": array.tolist(),
    }


def main() -> None:
    args = arguments()
    loaded = [
        (path, json.loads(path.read_text())) for path in args.inputs
    ]
    loaded.sort(key=lambda item: int(item[1]["seed"]))
    paths = [item[0] for item in loaded]
    rows = [item[1] for item in loaded]
    if len({row["seed"] for row in rows}) != len(rows):
        raise ValueError("Duplicate seeds")
    if any(row["status"] != "ok" for row in rows):
        raise ValueError("All input runs must be successful")
    method_summary = {}
    for variant in VARIANTS:
        method_summary[variant] = {
            "selected_alpha": [
                row["methods"][variant]["selected_alpha"] for row in rows
            ],
            "validation_exact_mae": summary(
                metric(
                    row["methods"][variant],
                    "validation",
                    "exact_cell",
                    "patient_first_normalized_mae",
                )
                for row in rows
            ),
            "test_exact_mae": summary(
                metric(
                    row["methods"][variant],
                    "test",
                    "exact_cell",
                    "patient_first_normalized_mae",
                )
                for row in rows
            ),
            "test_exact_spearman": summary(
                metric(
                    row["methods"][variant],
                    "test",
                    "exact_cell",
                    "macro_patient_marker_spearman",
                )
                for row in rows
            ),
            "test_population_wasserstein": summary(
                metric(
                    row["methods"][variant],
                    "test",
                    "population",
                    "patient_first_normalized_wasserstein",
                )
                for row in rows
            ),
        }
    frozen_reference = {
        "test_exact_mae": summary(
            metric(
                row["direct_ot_reference"],
                "test",
                "exact_cell",
                "patient_first_normalized_mae",
            )
            for row in rows
        ),
        "test_population_wasserstein": summary(
            metric(
                row["direct_ot_reference"],
                "test",
                "population",
                "patient_first_normalized_wasserstein",
            )
            for row in rows
        ),
    }
    comparisons = []
    for row in rows:
        direct = row["methods"]["direct_ot"]
        paired = row["methods"]["paired_decur_aux"]
        uniform = row["methods"]["uniform_decur_aux"]
        comparisons.append(
            {
                "seed": row["seed"],
                "paired_beats_direct_validation": metric(
                    paired,
                    "validation",
                    "exact_cell",
                    "patient_first_normalized_mae",
                )
                < metric(
                    direct,
                    "validation",
                    "exact_cell",
                    "patient_first_normalized_mae",
                ),
                "paired_beats_uniform_validation": metric(
                    paired,
                    "validation",
                    "exact_cell",
                    "patient_first_normalized_mae",
                )
                < metric(
                    uniform,
                    "validation",
                    "exact_cell",
                    "patient_first_normalized_mae",
                ),
                "paired_beats_direct_test": metric(
                    paired,
                    "test",
                    "exact_cell",
                    "patient_first_normalized_mae",
                )
                < metric(
                    direct,
                    "test",
                    "exact_cell",
                    "patient_first_normalized_mae",
                ),
                "paired_beats_uniform_test": metric(
                    paired,
                    "test",
                    "exact_cell",
                    "patient_first_normalized_mae",
                )
                < metric(
                    uniform,
                    "test",
                    "exact_cell",
                    "patient_first_normalized_mae",
                ),
            }
        )
    counts = {
        key: int(sum(bool(row[key]) for row in comparisons))
        for key in comparisons[0]
        if key != "seed"
    }
    paired_mean = method_summary["paired_decur_aux"]["test_exact_mae"]["mean"]
    direct_mean = method_summary["direct_ot"]["test_exact_mae"]["mean"]
    uniform_mean = method_summary["uniform_decur_aux"]["test_exact_mae"]["mean"]
    paired_w = method_summary["paired_decur_aux"][
        "test_population_wasserstein"
    ]["mean"]
    direct_w = method_summary["direct_ot"]["test_population_wasserstein"][
        "mean"
    ]
    paired_spearman = method_summary["paired_decur_aux"][
        "test_exact_spearman"
    ]["mean"]
    direct_spearman = method_summary["direct_ot"]["test_exact_spearman"][
        "mean"
    ]
    patient_keys = sorted(
        rows[0]["methods"]["direct_ot"]["test"]["exact_cell"][
            "patient_normalized_mae"
        ]
    )
    patient_deltas = []
    for patient in patient_keys:
        patient_deltas.append(
            np.mean(
                [
                    row["methods"]["paired_decur_aux"]["test"]["exact_cell"][
                        "patient_normalized_mae"
                    ][patient]
                    - row["methods"]["direct_ot"]["test"]["exact_cell"][
                        "patient_normalized_mae"
                    ][patient]
                    for row in rows
                ]
            )
        )
    patient_deltas = np.asarray(patient_deltas)
    rng = np.random.RandomState(4207)
    bootstrap = np.asarray(
        [
            patient_deltas[
                rng.randint(0, patient_deltas.size, patient_deltas.size)
            ].mean()
            for _ in range(10000)
        ]
    )
    exact_delta = paired_mean - direct_mean
    passed = (
        counts["paired_beats_direct_validation"] >= 2
        and counts["paired_beats_uniform_validation"] >= 2
        and paired_mean < direct_mean
        and paired_mean < uniform_mean
    )
    result = {
        "status": "ok",
        "pre_specified_gate": {
            "passed": passed,
            "decision": (
                "promote_decur_auxiliary"
                if passed
                else "stop_decur_auxiliary"
            ),
            "rule": (
                "Paired-OT DeCUR must beat direct and uniform DeCUR on "
                "validation in at least two seeds and on mean test exact MAE."
            ),
        },
        "gate_counts": counts,
        "method_summary": method_summary,
        "frozen_direct_ot_reference": frozen_reference,
        "effect_size": {
            "paired_minus_direct_test_exact_mae": exact_delta,
            "relative_exact_mae_improvement_percent": (
                100.0 * (direct_mean - paired_mean) / direct_mean
            ),
            "paired_minus_direct_test_population_wasserstein": (
                paired_w - direct_w
            ),
            "paired_minus_direct_test_exact_spearman": (
                paired_spearman - direct_spearman
            ),
            "patient_cluster_bootstrap_exact_mae_delta_95_percent_ci": (
                np.quantile(bootstrap, (0.025, 0.975)).tolist()
            ),
            "patient_cluster_bootstrap_probability_of_improvement": float(
                np.mean(bootstrap < 0.0)
            ),
            "patients_improved_after_seed_average": int(
                np.sum(patient_deltas < 0.0)
            ),
            "n_patients": int(patient_deltas.size),
            "practical_interpretation": (
                "directional_signal_but_not_yet_material_or_precise"
                if exact_delta < 0
                and np.quantile(bootstrap, 0.975) >= 0
                else (
                    "material_supported_improvement"
                    if exact_delta < 0
                    else "no_directional_improvement"
                )
            ),
        },
        "seed_comparisons": comparisons,
        "runs": [
            {
                "seed": row["seed"],
                "artifact": str(path),
                "host": row["host"],
                "cuda_device": row["cuda_device"],
                "elapsed_seconds": row["elapsed_seconds"],
            }
            for path, row in zip(paths, rows)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
