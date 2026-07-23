#!/usr/bin/env python3
"""Aggregate the pre-specified three-seed OT new-specimen inference gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SEEDS = (4207, 4208, 4209)
ARCHITECTURES = ("ridge", "mlp")
TEACHERS = ("paired_ot", "pooled_ot", "pooled_uniform")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def exact(method: dict, split: str) -> float:
    return float(
        method[split]["exact_cell"]["patient_first_normalized_mae"]
    )


def wasserstein(method: dict, split: str) -> float:
    return float(
        method[split]["population"]["patient_first_normalized_wasserstein"]
    )


def mean_std(values) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)),
    }


def run_row(artifact: dict) -> dict:
    methods = artifact["methods"]
    rows = {}
    for name, method in methods.items():
        rows[name] = {
            "selected_alpha": (
                None
                if "selected_alpha" not in method
                else float(method["selected_alpha"])
            ),
            "validation_exact_mae": exact(method, "validation"),
            "test_exact_mae": exact(method, "test"),
            "validation_population_wasserstein": wasserstein(
                method, "validation"
            ),
            "test_population_wasserstein": wasserstein(method, "test"),
        }
    selected = {}
    for teacher in ("paired_ot", "pooled_ot"):
        candidates = [
            f"{architecture}_{teacher}_hxl"
            for architecture in ARCHITECTURES
        ]
        chosen = min(
            candidates,
            key=lambda name: (
                rows[name]["validation_exact_mae"],
                name,
            ),
        )
        architecture = chosen.split("_", 1)[0]
        selected[teacher] = {
            "method": chosen,
            "architecture": architecture,
            "h_control_method": f"{architecture}_{teacher}_hl",
        }
    return {
        "seed": int(artifact["seed"]),
        "host": str(artifact["host"]),
        "cuda_device": str(artifact["cuda_device"]),
        "cuda_visible_memory_bytes": int(
            artifact["cuda_visible_memory_bytes"]
        ),
        "elapsed_seconds": float(artifact["elapsed_seconds"]),
        "train": artifact["train"],
        "methods": rows,
        "validation_selected": selected,
    }


def main() -> None:
    args = arguments()
    paths = sorted(args.results.glob("seed*.json"))
    artifacts = [json.loads(path.read_text()) for path in paths]
    if len(artifacts) != 3 or any(item.get("status") != "ok" for item in artifacts):
        raise ValueError("Expected three complete OT new-sample artifacts")
    if tuple(sorted(int(item["seed"]) for item in artifacts)) != SEEDS:
        raise ValueError(f"Expected seeds {SEEDS}")
    for artifact in artifacts:
        contract = artifact["contract"]
        if (
            contract["teacher_generation_split"] != "train_only"
            or contract["test_target_reference_used_for_inference"]
            or contract["validation_target_reference_used_for_inference"]
            or not contract["validation_and_test_specimens_unseen_during_fit"]
            or contract["hidden_source_y_used_for_training"]
        ):
            raise ValueError("New-sample inference contract was violated")

    rows = sorted(
        (run_row(artifact) for artifact in artifacts),
        key=lambda row: row["seed"],
    )
    method_names = tuple(sorted(rows[0]["methods"]))
    if any(tuple(sorted(row["methods"])) != method_names for row in rows):
        raise ValueError("Method sets differ between seeds")
    summary = {
        method: {
            metric: mean_std(
                row["methods"][method][metric] for row in rows
            )
            for metric in (
                "selected_alpha",
                "validation_exact_mae",
                "test_exact_mae",
                "validation_population_wasserstein",
                "test_population_wasserstein",
            )
            if rows[0]["methods"][method][metric] is not None
        }
        for method in method_names
    }

    comparisons = []
    for row in rows:
        methods = row["methods"]
        current = {"seed": row["seed"]}
        for teacher in ("paired_ot", "pooled_ot"):
            selected = row["validation_selected"][teacher]
            method = methods[selected["method"]]
            h_control = methods[selected["h_control_method"]]
            current[f"{teacher}_selected_method"] = selected["method"]
            current[f"{teacher}_positive_alpha"] = (
                method["selected_alpha"] > 0
            )
            current[f"{teacher}_validation_beats_baseline_and_h_control"] = (
                method["validation_exact_mae"]
                < methods["baseline_hl"]["validation_exact_mae"]
                and method["validation_exact_mae"]
                < h_control["validation_exact_mae"]
            )
            current[f"{teacher}_test_beats_baseline_and_h_control"] = (
                method["test_exact_mae"]
                < methods["baseline_hl"]["test_exact_mae"]
                and method["test_exact_mae"] < h_control["test_exact_mae"]
            )
        for architecture in ARCHITECTURES:
            current[f"{architecture}_pooled_ot_beats_uniform_validation"] = (
                methods[f"{architecture}_pooled_ot_hxl"][
                    "validation_exact_mae"
                ]
                < methods[f"{architecture}_pooled_uniform_hxl"][
                    "validation_exact_mae"
                ]
            )
        comparisons.append(current)

    boolean_keys = [
        key
        for key, value in comparisons[0].items()
        if key != "seed" and isinstance(value, bool)
    ]
    counts = {
        key: sum(bool(row[key]) for row in comparisons)
        for key in boolean_keys
    }

    def selected_mean(teacher: str, metric: str) -> float:
        return float(
            np.mean(
                [
                    row["methods"][
                        row["validation_selected"][teacher]["method"]
                    ][metric]
                    for row in rows
                ]
            )
        )

    def selected_h_control_mean(teacher: str, metric: str) -> float:
        return float(
            np.mean(
                [
                    row["methods"][
                        row["validation_selected"][teacher][
                            "h_control_method"
                        ]
                    ][metric]
                    for row in rows
                ]
            )
        )

    baseline_test = summary["baseline_hl"]["test_exact_mae"]["mean"]
    pooled_test = selected_mean("pooled_ot", "test_exact_mae")
    pooled_h_test = selected_h_control_mean(
        "pooled_ot", "test_exact_mae"
    )
    pooled_gate = (
        counts["pooled_ot_positive_alpha"] >= 2
        and counts[
            "pooled_ot_validation_beats_baseline_and_h_control"
        ]
        >= 2
        and pooled_test < baseline_test
        and pooled_test < pooled_h_test
    )
    paired_test = selected_mean("paired_ot", "test_exact_mae")
    paired_h_test = selected_h_control_mean(
        "paired_ot", "test_exact_mae"
    )
    paired_gate = (
        counts["paired_ot_positive_alpha"] >= 2
        and counts[
            "paired_ot_validation_beats_baseline_and_h_control"
        ]
        >= 2
        and paired_test < baseline_test
        and paired_test < paired_h_test
    )
    result = {
        "status": "ok",
        "pre_specified_gate": {
            "pooled_unpaired_passed": pooled_gate,
            "paired_specimen_passed": paired_gate,
            "pooled_unpaired_decision": (
                "retain_ot_new_sample_candidate"
                if pooled_gate
                else "stop_pooled_ot_distillation"
            ),
            "paired_specimen_decision": (
                "retain_ot_new_sample_candidate"
                if paired_gate
                else "stop_paired_ot_distillation"
            ),
            "rule": (
                "Validation-selected H+X architecture must have positive alpha "
                "and beat Ridge-H+L plus its same-architecture H-only "
                "distiller in at least two seeds; mean test MAE must beat both."
            ),
        },
        "gate_counts_out_of_three": counts,
        "selected_method_means": {
            "baseline_hl_test_exact_mae": baseline_test,
            "pooled_ot_test_exact_mae": pooled_test,
            "pooled_ot_h_control_test_exact_mae": pooled_h_test,
            "paired_ot_test_exact_mae": paired_test,
            "paired_ot_h_control_test_exact_mae": paired_h_test,
        },
        "method_summary": summary,
        "seed_comparisons": comparisons,
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
