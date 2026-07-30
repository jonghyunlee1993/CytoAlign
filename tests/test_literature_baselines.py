import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.literature_imputation import (
    balanced_group_rows,
    marker_column_indices,
    predict_cycombine,
)
from src.training import literature_baseline
from src.training.self_recoverability import PLATFORM_FINE_CLASSES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
R_SCRIPT = Path(
    "/project/kimlab_tcga/JH_workspace/conda_envs/"
    "cytometry_r_baselines/bin/Rscript"
)


def test_marker_column_indices_do_not_assume_union_order():
    result = marker_column_indices(
        ("CCR7", "CD3", "CTLA-4", "TIGIT"),
        ("CTLA-4", "TIGIT"),
    )
    assert result.tolist() == [2, 3]


def test_balanced_group_rows_limits_dominant_specimens():
    groups = np.asarray(["large"] * 100 + ["small"] * 10)
    rows = balanced_group_rows(groups, 20, seed=7)
    selected = groups[rows]
    assert len(rows) == 20
    assert np.sum(selected == "large") == 10
    assert np.sum(selected == "small") == 10


@pytest.mark.skipif(not R_SCRIPT.is_file(), reason="isolated R baseline env absent")
def test_cycombine_adapter_preserves_order_and_full_coverage():
    rng = np.random.RandomState(3)
    observed_reference = rng.normal(size=(120, 2))
    hidden_reference = (2 * observed_reference[:, :1] + 0.1).astype(np.float32)
    reference = np.column_stack([observed_reference, hidden_reference])
    query = rng.normal(size=(11, 2)).astype(np.float32)
    prediction, metadata = predict_cycombine(
        reference_full=reference,
        query_observed=query,
        full_markers=("H1", "H2", "Y"),
        observed_markers=("H1", "H2"),
        hidden_markers=("Y",),
        fallback_hidden=np.asarray([0.0], dtype=np.float32),
        script=PROJECT_ROOT / "scripts/cycombine_panel_impute.R",
        rscript=R_SCRIPT,
        seed=19,
        xdim=1,
        ydim=1,
        rlen=1,
        minimum_reference_cells=50,
    )
    assert prediction.shape == (11, 1)
    assert np.all(np.isfinite(prediction))
    assert metadata["coverage_fraction"] == 1.0
    assert metadata["fallback_cells"] == 0


def _write_cache(path, specimen, markers, values, labels):
    target = path / "spectral_flow" / f"{specimen}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        target,
        values=np.asarray(values, dtype=np.float32),
        labels=np.asarray(labels, dtype=str),
        row_indices=np.arange(len(values), dtype=np.uint32),
        markers=np.asarray(markers, dtype=str),
    )


def test_literature_runner_uses_query_shared_markers_only(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    classes = PLATFORM_FINE_CLASSES["spectral_flow"]
    markers = ("H1", "H2", "Y")
    rng = np.random.RandomState(31)
    specimens = ("R0001_A", "R0002_A", "R0003_A")
    cache = tmp_path / "cache"
    for specimen_index, specimen in enumerate(specimens):
        labels = np.tile(np.asarray(classes), 12)
        class_index = np.tile(np.arange(len(classes)), 12).astype(np.float32)
        h1 = class_index + rng.normal(scale=0.1, size=len(labels))
        h2 = rng.normal(size=len(labels))
        hidden = h1 + specimen_index
        _write_cache(
            cache,
            specimen,
            markers,
            np.column_stack([h1, h2, hidden]),
            labels,
        )

    split = {
        "folds": [
            {
                "fold_index": 0,
                "train_specimens": ["R0001_A"],
                "validation_specimens": ["R0002_A"],
                "test_specimens": ["R0003_A"],
                "test_patients": ["R0003"],
            }
        ]
    }
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    seen = {}

    def fake_cycombine(**kwargs):
        seen["query_shape"] = kwargs["query_observed"].shape
        seen["reference_shape"] = kwargs["reference_full"].shape
        return (
            np.zeros((len(kwargs["query_observed"]), 1), dtype=np.float32),
            {
                "query_access": "transductive_shared_markers",
                "coverage_fraction": 1.0,
            },
        )

    monkeypatch.setattr(literature_baseline, "predict_cycombine", fake_cycombine)
    config = {
        "experiment": {"name": "tiny_literature"},
        "data": {
            "cache_root": str(cache),
            "split_manifest": str(split_path),
        },
        "panels": {"h2": {"markers": ["H1", "H2"]}},
        "training": {
            "device": "cpu",
            "fit_sample_seed": 7,
            "cells_per_fit_patient": 96,
            "reference_bank": {"max_reference_cells": 100, "seed": 7},
        },
        "methods": {
            "cycombine": {
                "script": "unused",
                "rscript": "unused",
                "xdim": 1,
                "ydim": 1,
                "rlen": 1,
                "minimum_reference_cells": 1,
            }
        },
        "classifier": {
            "c": 1.0,
            "max_iter": 200,
            "maximum_cells_per_class": 12,
            "sample_seed": 7,
            "seed": 7,
        },
        "output": {"root": str(tmp_path / "outputs")},
    }
    result = literature_baseline.run_literature_baseline(
        config,
        method="cycombine",
        modality="spectral_flow",
        panel_name="h2",
        fold_index=0,
        seed=13,
    )
    assert seen["query_shape"] == (96, 2)
    assert seen["reference_shape"] == (100, 3)
    assert result["information_access"]["query_hidden_markers"] is False
    marker = pd.read_csv(result["artifacts"]["marker_metrics"])
    assert set(marker["representation"]) == {"cycombine"}
