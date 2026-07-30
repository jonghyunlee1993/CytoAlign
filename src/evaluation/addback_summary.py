"""Patient-paired aggregation for clinical10-to-H19 add-back sweeps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


MARKER_METRICS = (
    "null_relative_skill",
    "spearman",
    "normalized_mae",
)
BIOLOGY_METRICS = (
    "balanced_accuracy",
    "macro_f1",
    "macro_auprc",
)
CLASS_METRICS = (
    "recall",
    "precision",
    "auprc",
    "observed_prevalence",
    "predicted_prevalence",
    "prevalence_error",
)


def _bootstrap_interval(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[float | None, float | None]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return None, None
    rng = np.random.RandomState(int(seed))
    boot = np.mean(
        rng.choice(
            finite,
            size=(int(replicates), len(finite)),
            replace=True,
        ),
        axis=1,
    )
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _biology_rows(
    run_paths: Sequence[Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows = []
    class_rows = []
    for run_path in run_paths:
        run = json.loads(run_path.read_text(encoding="utf-8"))
        biology = json.loads(
            Path(run["artifacts"]["biology_metrics"]).read_text(
                encoding="utf-8"
            )
        )
        for panel, panel_payload in biology["panels"].items():
            panel_common = {
                "modality": run["modality"],
                "panel": panel,
                "panel_kind": panel_payload["kind"],
                "added_marker": panel_payload["added_marker"],
                "fold": int(run["fold"]),
                "seed": int(run["seed"]),
            }
            for specimen, specimen_payload in panel_payload[
                "specimens"
            ].items():
                common = {
                    **panel_common,
                    "patient": specimen_payload["patient"],
                    "specimen": specimen,
                    "n_cells": int(specimen_payload["n_cells"]),
                }
                for representation, metrics in specimen_payload[
                    "metrics"
                ].items():
                    overall_rows.append(
                        {
                            **common,
                            "representation": representation,
                            **{
                                metric: metrics[metric]
                                for metric in BIOLOGY_METRICS
                            },
                        }
                    )
                    for label, class_metrics in metrics["per_class"].items():
                        class_rows.append(
                            {
                                **common,
                                "representation": representation,
                                "label": label,
                                **{
                                    metric: class_metrics[metric]
                                    for metric in CLASS_METRICS
                                },
                            }
                        )
    return pd.DataFrame(overall_rows), pd.DataFrame(class_rows)


def _paired_effect_rows(
    patient_frame: pd.DataFrame,
    *,
    base_panel: str,
    upper_panel: str,
    id_columns: Sequence[str],
    metrics: Sequence[str],
    bootstrap_replicates: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    panel_info = patient_frame[
        ["modality", "panel", "panel_kind", "added_marker"]
    ].drop_duplicates()
    for modality in sorted(patient_frame["modality"].unique()):
        current = patient_frame[patient_frame["modality"] == modality]
        base = current[current["panel"] == base_panel]
        upper = current[current["panel"] == upper_panel]
        for panel_index, panel_row in enumerate(
            panel_info[
                (panel_info["modality"] == modality)
                & (panel_info["panel"] != base_panel)
            ].itertuples(index=False)
        ):
            panel = current[current["panel"] == panel_row.panel]
            merge_keys = [*id_columns, "patient"]
            merged = panel.merge(
                base[[*merge_keys, *metrics]],
                on=merge_keys,
                how="inner",
                suffixes=("_panel", "_base"),
                validate="one_to_one",
            ).merge(
                upper[[*merge_keys, *metrics]],
                on=merge_keys,
                how="inner",
                validate="one_to_one",
            )
            metric_renames = {
                metric: f"{metric}_upper" for metric in metrics
            }
            merged = merged.rename(columns=metric_renames)
            group_columns = list(id_columns)
            grouped = (
                merged.groupby(group_columns, dropna=False, sort=True)
                if group_columns
                else [((), merged)]
            )
            for group_index, (group_key, group) in enumerate(grouped):
                if not isinstance(group_key, tuple):
                    group_key = (group_key,)
                row = {
                    "modality": modality,
                    "panel": panel_row.panel,
                    "panel_kind": panel_row.panel_kind,
                    "added_marker": panel_row.added_marker,
                    **dict(zip(group_columns, group_key)),
                    "n_patients": int(group["patient"].nunique()),
                }
                for metric_index, metric in enumerate(metrics):
                    panel_values = pd.to_numeric(
                        group[f"{metric}_panel"],
                        errors="coerce",
                    ).to_numpy(dtype=float)
                    base_values = pd.to_numeric(
                        group[f"{metric}_base"],
                        errors="coerce",
                    ).to_numpy(dtype=float)
                    upper_values = pd.to_numeric(
                        group[f"{metric}_upper"],
                        errors="coerce",
                    ).to_numpy(dtype=float)
                    finite = np.isfinite(
                        panel_values + base_values + upper_values
                    )
                    panel_values = panel_values[finite]
                    base_values = base_values[finite]
                    upper_values = upper_values[finite]
                    delta = panel_values - base_values
                    lower, upper_ci = _bootstrap_interval(
                        delta,
                        replicates=bootstrap_replicates,
                        seed=(
                            int(seed)
                            + 1009 * panel_index
                            + 104729 * group_index
                            + 9176 * metric_index
                        ),
                    )
                    panel_mean = (
                        float(np.mean(panel_values))
                        if len(panel_values)
                        else None
                    )
                    base_mean = (
                        float(np.mean(base_values))
                        if len(base_values)
                        else None
                    )
                    upper_mean = (
                        float(np.mean(upper_values))
                        if len(upper_values)
                        else None
                    )
                    delta_mean = (
                        float(np.mean(delta)) if len(delta) else None
                    )
                    gap = (
                        upper_mean - base_mean
                        if upper_mean is not None and base_mean is not None
                        else None
                    )
                    row.update(
                        {
                            f"{metric}_base_mean": base_mean,
                            f"{metric}_panel_mean": panel_mean,
                            f"{metric}_upper_mean": upper_mean,
                            f"{metric}_delta_mean": delta_mean,
                            f"{metric}_delta_ci_lower": lower,
                            f"{metric}_delta_ci_upper": upper_ci,
                            f"{metric}_gap_rescue_fraction": (
                                delta_mean / gap
                                if (
                                    delta_mean is not None
                                    and gap is not None
                                    and abs(gap) > 1.0e-12
                                )
                                else None
                            ),
                        }
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def _global_marker_effect(
    marker_patient: pd.DataFrame,
    *,
    base_panel: str,
    upper_panel: str,
    bootstrap_replicates: int,
    seed: int,
) -> pd.DataFrame:
    global_patient = (
        marker_patient.groupby(
            [
                "modality",
                "panel",
                "panel_kind",
                "added_marker",
                "representation",
                "patient",
            ],
            as_index=False,
            dropna=False,
        )[list(MARKER_METRICS)]
        .mean(numeric_only=True)
    )
    return _paired_effect_rows(
        global_patient,
        base_panel=base_panel,
        upper_panel=upper_panel,
        id_columns=("representation",),
        metrics=MARKER_METRICS,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )


def summarize_addback_sweep(
    experiment_root: str | Path,
    *,
    bootstrap_replicates: int = 2000,
    seed: int = 4207,
) -> dict:
    root = Path(experiment_root)
    run_paths = sorted(root.glob("*/fold_*/seed_*/run_summary.json"))
    if not run_paths:
        raise FileNotFoundError(f"No completed add-back runs under {root}")
    runs = [
        json.loads(path.read_text(encoding="utf-8")) for path in run_paths
    ]
    if any(run.get("status") != "ok" for run in runs):
        raise ValueError("At least one add-back run is incomplete")
    modalities = sorted({run["modality"] for run in runs})
    folds_by_modality = {
        modality: sorted(
            int(run["fold"])
            for run in runs
            if run["modality"] == modality
        )
        for modality in modalities
    }
    base_panel = str(
        next(
            name
            for name, payload in runs[0]["panels"].items()
            if payload["kind"] == "base"
        )
    )
    upper_panel = str(
        next(
            name
            for name, payload in runs[0]["panels"].items()
            if payload["kind"] == "upper"
        )
    )

    marker_frames = [
        pd.read_csv(run["artifacts"]["marker_metrics"]) for run in runs
    ]
    marker = pd.concat(marker_frames, ignore_index=True)
    marker_patient = (
        marker.groupby(
            [
                "modality",
                "panel",
                "panel_kind",
                "added_marker",
                "marker",
                "representation",
                "patient",
            ],
            as_index=False,
            dropna=False,
        )[list(MARKER_METRICS)]
        .mean(numeric_only=True)
    )
    marker_effect = _paired_effect_rows(
        marker_patient,
        base_panel=base_panel,
        upper_panel=upper_panel,
        id_columns=("marker", "representation"),
        metrics=MARKER_METRICS,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    marker_global = _global_marker_effect(
        marker_patient,
        base_panel=base_panel,
        upper_panel=upper_panel,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed + 1,
    )

    biology, biology_class = _biology_rows(run_paths)
    biology_patient = (
        biology.groupby(
            [
                "modality",
                "panel",
                "panel_kind",
                "added_marker",
                "representation",
                "patient",
            ],
            as_index=False,
            dropna=False,
        )[list(BIOLOGY_METRICS)]
        .mean(numeric_only=True)
    )
    biology_effect = _paired_effect_rows(
        biology_patient,
        base_panel=base_panel,
        upper_panel=upper_panel,
        id_columns=("representation",),
        metrics=BIOLOGY_METRICS,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed + 2,
    )
    biology_class_patient = (
        biology_class.groupby(
            [
                "modality",
                "panel",
                "panel_kind",
                "added_marker",
                "representation",
                "label",
                "patient",
            ],
            as_index=False,
            dropna=False,
        )[list(CLASS_METRICS)]
        .mean(numeric_only=True)
    )
    biology_class_effect = _paired_effect_rows(
        biology_class_patient,
        base_panel=base_panel,
        upper_panel=upper_panel,
        id_columns=("representation", "label"),
        metrics=CLASS_METRICS,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed + 3,
    )

    output = root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "marker_patient": output / "marker_patient.csv",
        "marker_effect": output / "marker_addback_effect.csv",
        "marker_global": output / "marker_addback_global.csv",
        "biology_patient": output / "biology_patient.csv",
        "biology_effect": output / "biology_addback_effect.csv",
        "biology_class_patient": output / "biology_class_patient.csv",
        "biology_class_effect": output / "biology_class_addback_effect.csv",
    }
    marker_patient.to_csv(artifacts["marker_patient"], index=False)
    marker_effect.to_csv(artifacts["marker_effect"], index=False)
    marker_global.to_csv(artifacts["marker_global"], index=False)
    biology_patient.to_csv(artifacts["biology_patient"], index=False)
    biology_effect.to_csv(artifacts["biology_effect"], index=False)
    biology_class_patient.to_csv(
        artifacts["biology_class_patient"],
        index=False,
    )
    biology_class_effect.to_csv(
        artifacts["biology_class_effect"],
        index=False,
    )
    payload = {
        "status": "ok",
        "completed_runs": len(run_paths),
        "modalities": modalities,
        "folds_by_modality": folds_by_modality,
        "base_panel": base_panel,
        "upper_panel": upper_panel,
        "panels": sorted(
            {
                panel
                for run in runs
                for panel in run["panels"]
            }
        ),
        "panels_by_modality": {
            modality: list(
                next(
                    run["panels"]
                    for run in runs
                    if run["modality"] == modality
                )
            )
            for modality in modalities
        },
        "patients": int(marker["patient"].nunique()),
        "primary_target_markers": {
            modality: next(
                run["primary_target_markers"]
                for run in runs
                if run["modality"] == modality
            )
            for modality in modalities
        },
        "bootstrap_unit": "patient",
        "artifacts": {key: str(value) for key, value in artifacts.items()},
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload["artifacts"]["summary"] = str(summary_path)
    return payload
