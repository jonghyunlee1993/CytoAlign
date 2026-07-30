"""Combine inductive same-cell baselines and transductive literature methods."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.self_recoverability_summary import (
    BIOLOGY_METRICS,
    MARKER_METRICS,
    _summarize_numeric,
)


BASE_REPRESENTATIONS = {
    "median": ("fit_marker_median", "inductive_fit_only"),
    "knn": ("cytofmerge_knn50_core", "inductive_fit_only"),
    "mlp": ("simple_mlp", "inductive_fit_only"),
}


def _completed_paths(root: Path, pattern: str) -> list[Path]:
    paths = []
    for path in sorted(root.glob(pattern)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "ok":
            paths.append(path)
    return paths


def _method_from_biology_representation(representation: str) -> str | None:
    for prefix in ("full_hybrid_", "hidden_"):
        if representation.startswith(prefix):
            return representation[len(prefix) :]
    return None


def summarize_literature_baselines(
    literature_root: str | Path,
    same_cell_root: str | Path,
    *,
    bootstrap_replicates: int = 2000,
    seed: int = 4207,
) -> dict:
    literature_root = Path(literature_root)
    same_cell_root = Path(same_cell_root)
    base_paths = _completed_paths(
        same_cell_root,
        "*/*/fold_*/seed_*/run_summary.json",
    )
    literature_paths = _completed_paths(
        literature_root,
        "*/*/*/fold_*/seed_*/run_summary.json",
    )
    if not base_paths:
        raise FileNotFoundError(f"No same-cell runs found under {same_cell_root}")
    if not literature_paths:
        raise FileNotFoundError(
            f"No literature baseline runs found under {literature_root}"
        )
    literature_keys = {
        (
            payload["modality"],
            payload["panel"],
            int(payload["fold"]),
            int(payload["seed"]),
        )
        for payload in (
            json.loads(path.read_text(encoding="utf-8"))
            for path in literature_paths
        )
    }
    matched_base_paths = []
    for path in base_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = (
            payload["modality"],
            payload["panel"],
            int(payload["fold"]),
            int(payload["seed"]),
        )
        if key in literature_keys:
            matched_base_paths.append(path)
    base_paths = matched_base_paths

    marker_frames = []
    biology_rows = []
    class_rows = []
    coverage_rows = []
    all_paths = [(path, False) for path in base_paths] + [
        (path, True) for path in literature_paths
    ]
    for run_path, is_literature in all_paths:
        run = json.loads(run_path.read_text(encoding="utf-8"))
        marker = pd.read_csv(run["artifacts"]["marker_metrics"])
        if is_literature:
            method = str(run["method"])
            marker = marker[marker["representation"] == method].copy()
            marker["method"] = method
            marker["information_access"] = "transductive_query_H"
            metadata = run.get("method_metadata", {})
            coverage_rows.append(
                {
                    "method": method,
                    "modality": run["modality"],
                    "panel": run["panel"],
                    "fold": int(run["fold"]),
                    "seed": int(run["seed"]),
                    "coverage_fraction": metadata.get("coverage_fraction", 1.0),
                    "fallback_cells": metadata.get("fallback_cells", 0),
                    "test_cells": int(run["test_cells"]),
                }
            )
        else:
            marker = marker[
                marker["representation"].isin(BASE_REPRESENTATIONS)
            ].copy()
            marker["method"] = marker["representation"].map(
                lambda name: BASE_REPRESENTATIONS[name][0]
            )
            marker["information_access"] = marker["representation"].map(
                lambda name: BASE_REPRESENTATIONS[name][1]
            )
        marker_frames.append(marker)

        biology = json.loads(
            Path(run["artifacts"]["biology_metrics"]).read_text(encoding="utf-8")
        )
        literature_method = str(run["method"]) if is_literature else None
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
                candidate = _method_from_biology_representation(representation)
                if is_literature:
                    if candidate != literature_method:
                        continue
                    method = literature_method
                    access = "transductive_query_H"
                else:
                    if candidate not in BASE_REPRESENTATIONS:
                        if representation not in (
                            "full_true",
                            "hidden_true",
                            "observed_true",
                        ):
                            continue
                        method = representation
                        access = "truth_reference"
                    else:
                        method, access = BASE_REPRESENTATIONS[candidate]
                biology_rows.append(
                    {
                        **common,
                        "representation": representation,
                        "method": method,
                        "information_access": access,
                        **{name: metrics[name] for name in BIOLOGY_METRICS},
                    }
                )
                for label, values in metrics["per_class"].items():
                    class_rows.append(
                        {
                            **common,
                            "representation": representation,
                            "method": method,
                            "information_access": access,
                            "label": label,
                            **values,
                        }
                    )

    marker = pd.concat(marker_frames, ignore_index=True)
    marker_patient = (
        marker.groupby(
            [
                "modality",
                "panel",
                "marker",
                "method",
                "information_access",
                "patient",
            ],
            as_index=False,
        )[list(MARKER_METRICS)]
        .mean(numeric_only=True)
    )
    marker_summary = _summarize_numeric(
        marker_patient,
        ("modality", "panel", "marker", "method", "information_access"),
        MARKER_METRICS,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    marker_panel_patient = (
        marker_patient.groupby(
            [
                "modality",
                "panel",
                "method",
                "information_access",
                "patient",
            ],
            as_index=False,
        )[list(MARKER_METRICS)]
        .mean(numeric_only=True)
    )
    marker_panel_summary = _summarize_numeric(
        marker_panel_patient,
        ("modality", "panel", "method", "information_access"),
        MARKER_METRICS,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed + 3,
    )
    biology = pd.DataFrame(biology_rows)
    biology_patient = (
        biology.groupby(
            [
                "modality",
                "panel",
                "representation",
                "method",
                "information_access",
                "patient",
            ],
            as_index=False,
        )[list(BIOLOGY_METRICS)]
        .mean(numeric_only=True)
    )
    biology_summary = _summarize_numeric(
        biology_patient,
        (
            "modality",
            "panel",
            "representation",
            "method",
            "information_access",
        ),
        BIOLOGY_METRICS,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed + 1,
    )
    biology_class = pd.DataFrame(class_rows)
    class_metrics = (
        "recall",
        "precision",
        "auprc",
        "observed_prevalence",
        "predicted_prevalence",
        "prevalence_error",
    )
    class_patient = (
        biology_class.groupby(
            [
                "modality",
                "panel",
                "representation",
                "method",
                "information_access",
                "label",
                "patient",
            ],
            as_index=False,
        )[list(class_metrics)]
        .mean(numeric_only=True)
    )
    class_summary = _summarize_numeric(
        class_patient,
        (
            "modality",
            "panel",
            "representation",
            "method",
            "information_access",
            "label",
        ),
        class_metrics,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed + 2,
    )
    coverage = pd.DataFrame(coverage_rows)

    output = literature_root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "marker_patient": output / "marker_patient.csv",
        "marker_summary": output / "marker_summary.csv",
        "marker_panel_patient": output / "marker_panel_patient.csv",
        "marker_panel_summary": output / "marker_panel_summary.csv",
        "biology_patient": output / "biology_patient.csv",
        "biology_summary": output / "biology_summary.csv",
        "biology_class_patient": output / "biology_class_patient.csv",
        "biology_class_summary": output / "biology_class_summary.csv",
        "coverage": output / "coverage.csv",
    }
    marker_patient.to_csv(artifacts["marker_patient"], index=False)
    marker_summary.to_csv(artifacts["marker_summary"], index=False)
    marker_panel_patient.to_csv(
        artifacts["marker_panel_patient"],
        index=False,
    )
    marker_panel_summary.to_csv(
        artifacts["marker_panel_summary"],
        index=False,
    )
    biology_patient.to_csv(artifacts["biology_patient"], index=False)
    biology_summary.to_csv(artifacts["biology_summary"], index=False)
    class_patient.to_csv(artifacts["biology_class_patient"], index=False)
    class_summary.to_csv(artifacts["biology_class_summary"], index=False)
    coverage.to_csv(artifacts["coverage"], index=False)
    payload = {
        "status": "ok",
        "same_cell_runs": len(base_paths),
        "literature_runs": len(literature_paths),
        "methods": sorted(marker["method"].unique().tolist()),
        "bootstrap_unit": "patient",
        "information_access_stratified": True,
        "artifacts": {key: str(value) for key, value in artifacts.items()},
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload["artifacts"]["summary"] = str(summary_path)
    return payload
