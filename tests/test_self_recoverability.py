import json

import numpy as np
import pandas as pd
import pytest

from src.models.gpu_knn import TorchKNNMedianRegressor
from src.training.self_recoverability import (
    PLATFORM_FINE_CLASSES,
    biology_metrics,
    marker_metrics,
    panel_indices,
    run_self_recoverability,
)


def test_panel_indices_resolve_canonical_aliases():
    observed, hidden, observed_names, hidden_names = panel_indices(
        ("CD3", "CD197", "CD279", "CD11c"),
        ("CD3", "CCR7", "PD-1"),
    )
    assert observed.tolist() == [0, 1, 2]
    assert hidden.tolist() == [3]
    assert observed_names == ("CD3", "CD197", "CD279")
    assert hidden_names == ("CD11c",)


def test_torch_knn_median_cpu_matches_manual_neighbors():
    pytest.importorskip("torch")
    reference = np.asarray([[0.0], [1.0], [2.0], [10.0]], dtype=np.float32)
    targets = np.asarray([[0.0], [2.0], [4.0], [100.0]], dtype=np.float32)
    model = TorchKNNMedianRegressor(
        k=3,
        device="cpu",
        query_chunk_size=1,
    ).fit(reference, targets)
    prediction = model.predict(np.asarray([[1.1], [9.0]], dtype=np.float32))
    np.testing.assert_allclose(prediction[:, 0], [2.0, 4.0])


def test_marker_and_biology_metrics_detect_signal_and_prevalence():
    truth = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    median = np.repeat([[1.5]], 4, axis=0)
    metrics = marker_metrics(
        truth,
        truth,
        median,
        np.asarray([2.0]),
        ("Y",),
    )[0]
    assert metrics["normalized_mae"] == 0.0
    assert metrics["null_relative_skill"] == 1.0
    assert metrics["spearman"] == 1.0
    assert metrics["dynamic_range_retention"] == 1.0

    labels = np.asarray(["A", "A", "B", "B"])
    probabilities = np.asarray(
        [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]]
    )
    result = biology_metrics(labels, labels, probabilities, ("A", "B"))
    assert result["balanced_accuracy"] == 1.0
    assert result["macro_auprc"] == 1.0
    assert result["per_class"]["A"]["prevalence_error"] == 0.0


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


def test_small_end_to_end_fixed_classifier_run(tmp_path):
    pytest.importorskip("torch")
    classes = PLATFORM_FINE_CLASSES["spectral_flow"]
    markers = ("CD3", "CD4", "Y_SIGNAL", "Y_NOISE")
    rng = np.random.RandomState(9)
    specimens = ("R0001_A", "R0002_A", "R0003_A")
    cache = tmp_path / "cache"
    for specimen_index, specimen in enumerate(specimens):
        labels = np.tile(np.asarray(classes), 20)
        class_index = np.tile(np.arange(len(classes)), 20).astype(np.float32)
        h1 = class_index + rng.normal(scale=0.05, size=len(labels))
        h2 = (class_index % 3) + rng.normal(scale=0.05, size=len(labels))
        y_signal = 2.0 * h1 + 0.5 * h2
        y_noise = rng.normal(size=len(labels)) + specimen_index * 0.01
        values = np.column_stack([h1, h2, y_signal, y_noise])
        _write_cache(cache, specimen, markers, values, labels)

    split = {
        "specimens": list(specimens),
        "folds": [
            {
                "fold_index": 0,
                "train_patients": ["R0001"],
                "validation_patients": ["R0002"],
                "test_patients": ["R0003"],
                "train_specimens": ["R0001_A"],
                "validation_specimens": ["R0002_A"],
                "test_specimens": ["R0003_A"],
            }
        ],
    }
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    output = tmp_path / "outputs"
    config = {
        "experiment": {"name": "tiny"},
        "data": {
            "cache_root": str(cache),
            "split_manifest": str(split_path),
        },
        "panels": {"tiny_h": {"markers": ["CD3", "CD4"]}},
        "training": {
            "device": "cpu",
            "fit_sample_seed": 7,
            "cells_per_fit_patient": 160,
            "knn": {
                "k": 3,
                "max_reference_cells": 100,
                "query_chunk_size": 32,
                "distance_memory_fraction": 0.08,
            },
            "mlp": {
                "hidden_dims": [16],
                "epochs": 3,
                "batch_size": 32,
                "learning_rate": 0.01,
                "patience": 2,
            },
        },
        "classifier": {
            "c": 1.0,
            "max_iter": 200,
            "maximum_cells_per_class": 20,
            "sample_seed": 7,
            "seed": 7,
        },
        "evaluation": {"shuffle_seed": 11},
        "output": {"root": str(output)},
    }
    result = run_self_recoverability(
        config,
        modality="spectral_flow",
        panel_name="tiny_h",
        fold_index=0,
        seed=13,
    )
    assert result["status"] == "ok"
    marker_frame = pd.read_csv(result["artifacts"]["marker_metrics"])
    signal = marker_frame[
        (marker_frame["marker"] == "Y_SIGNAL")
        & (marker_frame["representation"] == "knn")
    ]
    assert signal["null_relative_skill"].iloc[0] > 0.8
    biology = json.loads(
        (output / "tiny/spectral_flow/tiny_h/fold_0/seed_13/biology_metrics.json")
        .read_text()
    )
    representations = biology["specimens"]["R0003_A"]["metrics"]
    assert "full_hybrid_knn" in representations
    assert "hidden_knn" in representations
