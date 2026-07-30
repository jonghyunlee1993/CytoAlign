"""Patient-first aggregation for same-cell recoverability experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


MARKER_METRICS = (
    "normalized_mae",
    "null_relative_skill",
    "spearman",
    "dynamic_range_retention",
    "normalized_median_error",
    "normalized_q90_error",
)

BIOLOGY_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "macro_auprc",
)


def _interval(
    values: np.ndarray,
    *,
    bootstrap_replicates: int,
    rng: np.random.RandomState,
) -> tuple[float | None, float | None]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return None, None
    boot = np.mean(
        rng.choice(
            finite,
            size=(int(bootstrap_replicates), len(finite)),
            replace=True,
        ),
        axis=1,
    )
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _summarize_numeric(
    patient_frame: pd.DataFrame,
    group_columns: Sequence[str],
    metrics: Sequence[str],
    *,
    bootstrap_replicates: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for group_key, current in patient_frame.groupby(
        list(group_columns), dropna=False, sort=True
    ):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = dict(zip(group_columns, group_key))
        row["n_patients"] = int(current["patient"].nunique())
        for metric_index, metric in enumerate(metrics):
            values = pd.to_numeric(current[metric], errors="coerce").to_numpy(
                dtype=float
            )
            finite = values[np.isfinite(values)]
            lower, upper = _interval(
                finite,
                bootstrap_replicates=bootstrap_replicates,
                rng=np.random.RandomState(
                    int(seed)
                    + 1009 * metric_index
                    + len(rows) * 104729
                ),
            )
            row[f"{metric}_mean"] = (
                float(np.mean(finite)) if len(finite) else None
            )
            row[f"{metric}_median"] = (
                float(np.median(finite)) if len(finite) else None
            )
            row[f"{metric}_ci_lower"] = lower
            row[f"{metric}_ci_upper"] = upper
        rows.append(row)
    return pd.DataFrame(rows)


def _biology_rows(paths: Sequence[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows = []
    class_rows = []
    for summary_path in paths:
        run = json.loads(summary_path.read_text(encoding="utf-8"))
        biology_path = Path(run["artifacts"]["biology_metrics"])
        biology = json.loads(biology_path.read_text(encoding="utf-8"))
        for specimen, specimen_payload in biology["specimens"].items():
            common = {
                "modality": run["modality"],
                "panel": run["panel"],
                "fold": int(run["fold"]),
                "seed": int(run["seed"]),
                "patient": specimen_payload["patient"],
                "specimen": specimen,
                "n_cells": int(specimen_payload["n_cells"]),
            }
            for representation, metrics in specimen_payload["metrics"].items():
                overall_rows.append(
                    {
                        **common,
                        "representation": representation,
                        **{name: metrics[name] for name in BIOLOGY_METRICS},
                    }
                )
                for label, class_metrics in metrics["per_class"].items():
                    class_rows.append(
                        {
                            **common,
                            "representation": representation,
                            "label": label,
                            **class_metrics,
                        }
                    )
    return pd.DataFrame(overall_rows), pd.DataFrame(class_rows)


def _retention_rows(patient_biology: pd.DataFrame) -> pd.DataFrame:
    rows = []
    methods = ("knn", "mlp", "knn_shuffled_h", "mlp_shuffled_h")
    for (modality, panel), current in patient_biology.groupby(
        ["modality", "panel"], sort=True
    ):
        indexed = current.set_index(["patient", "representation"])
        patients = sorted(current["patient"].unique())
        for probe in ("full", "hidden"):
            true_name = f"{probe}_true"
            null_name = (
                "full_hybrid_median" if probe == "full" else "hidden_median"
            )
            for method in methods:
                method_name = (
                    f"full_hybrid_{method}"
                    if probe == "full"
                    else f"hidden_{method}"
                )
                for metric in ("balanced_accuracy", "macro_f1", "macro_auprc"):
                    patient_values = []
                    for patient in patients:
                        keys = (
                            (patient, true_name),
                            (patient, null_name),
                            (patient, method_name),
                        )
                        if not all(key in indexed.index for key in keys):
                            continue
                        true = float(indexed.loc[keys[0], metric])
                        null = float(indexed.loc[keys[1], metric])
                        method_value = float(indexed.loc[keys[2], metric])
                        if all(np.isfinite([true, null, method_value])):
                            patient_values.append((true, null, method_value))
                    if not patient_values:
                        continue
                    array = np.asarray(patient_values)
                    true_mean, null_mean, method_mean = array.mean(axis=0)
                    denominator = true_mean - null_mean
                    rows.append(
                        {
                            "modality": modality,
                            "panel": panel,
                            "probe": probe,
                            "method": method,
                            "metric": metric,
                            "n_patients": len(array),
                            "true_mean": true_mean,
                            "null_mean": null_mean,
                            "method_mean": method_mean,
                            "biology_retention": (
                                (method_mean - null_mean) / denominator
                                if abs(denominator) > 1.0e-12
                                else None
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_self_recoverability(
    experiment_root: str | Path,
    *,
    bootstrap_replicates: int = 2000,
    seed: int = 4207,
) -> dict:
    root = Path(experiment_root)
    run_paths = sorted(root.glob("*/*/fold_*/seed_*/run_summary.json"))
    if not run_paths:
        raise FileNotFoundError(f"No completed runs found under {root}")
    marker_frames = []
    for path in run_paths:
        run = json.loads(path.read_text(encoding="utf-8"))
        if run.get("status") != "ok":
            raise ValueError(f"Incomplete run: {path}")
        marker_frames.append(pd.read_csv(run["artifacts"]["marker_metrics"]))
    marker = pd.concat(marker_frames, ignore_index=True)
    marker_patient = (
        marker.groupby(
            [
                "modality",
                "panel",
                "marker",
                "representation",
                "patient",
            ],
            as_index=False,
        )[list(MARKER_METRICS)]
        .mean(numeric_only=True)
    )
    marker_summary = _summarize_numeric(
        marker_patient,
        ("modality", "panel", "marker", "representation"),
        MARKER_METRICS,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )

    biology, biology_class = _biology_rows(run_paths)
    biology_patient = (
        biology.groupby(
            ["modality", "panel", "representation", "patient"],
            as_index=False,
        )[list(BIOLOGY_METRICS)]
        .mean(numeric_only=True)
    )
    biology_summary = _summarize_numeric(
        biology_patient,
        ("modality", "panel", "representation"),
        BIOLOGY_METRICS,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed + 1,
    )
    class_metric_names = (
        "recall",
        "precision",
        "auprc",
        "observed_prevalence",
        "predicted_prevalence",
        "prevalence_error",
    )
    biology_class_patient = (
        biology_class.groupby(
            ["modality", "panel", "representation", "label", "patient"],
            as_index=False,
        )[list(class_metric_names)]
        .mean(numeric_only=True)
    )
    biology_class_summary = _summarize_numeric(
        biology_class_patient,
        ("modality", "panel", "representation", "label"),
        class_metric_names,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed + 2,
    )
    retention = _retention_rows(biology_patient)

    output = root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "marker_patient": output / "marker_patient.csv",
        "marker_summary": output / "marker_summary.csv",
        "biology_patient": output / "biology_patient.csv",
        "biology_summary": output / "biology_summary.csv",
        "biology_class_patient": output / "biology_class_patient.csv",
        "biology_class_summary": output / "biology_class_summary.csv",
        "biology_retention": output / "biology_retention.csv",
    }
    marker_patient.to_csv(artifacts["marker_patient"], index=False)
    marker_summary.to_csv(artifacts["marker_summary"], index=False)
    biology_patient.to_csv(artifacts["biology_patient"], index=False)
    biology_summary.to_csv(artifacts["biology_summary"], index=False)
    biology_class_patient.to_csv(
        artifacts["biology_class_patient"], index=False
    )
    biology_class_summary.to_csv(
        artifacts["biology_class_summary"], index=False
    )
    retention.to_csv(artifacts["biology_retention"], index=False)
    payload = {
        "status": "ok",
        "completed_runs": len(run_paths),
        "modalities": sorted(marker["modality"].unique().tolist()),
        "panels": sorted(marker["panel"].unique().tolist()),
        "folds": sorted(map(int, marker["fold"].unique())),
        "seeds": sorted(map(int, marker["seed"].unique())),
        "patients": int(marker["patient"].nunique()),
        "bootstrap_unit": "patient",
        "seed_is_not_a_replicate": True,
        "artifacts": {key: str(value) for key, value in artifacts.items()},
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload["artifacts"]["summary"] = str(summary_path)
    return payload
