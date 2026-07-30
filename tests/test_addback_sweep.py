import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation.addback_summary import summarize_addback_sweep
from src.training.addback_sweep import (
    addback_panel_definitions,
    run_addback_sweep,
)
from src.training.self_recoverability import PLATFORM_FINE_CLASSES


def test_addback_definitions_keep_upper_hidden_targets_fixed():
    config = {
        "panels": {
            "base": {"markers": ["CD3"]},
            "upper": {"markers": ["CD3", "CCR7", "PD-1"]},
        },
        "sweep": {
            "base_panel": "base",
            "upper_panel": "upper",
            "addback_panels": {
                "add_ccr7": "CCR7",
                "add_pd1": "PD-1",
            },
        },
    }
    definitions, target_indices, target_markers = addback_panel_definitions(
        config,
        ("CD3", "CD197", "CD279", "Y"),
    )
    assert tuple(definitions) == (
        "base",
        "add_ccr7",
        "add_pd1",
        "upper",
    )
    assert target_indices.tolist() == [3]
    assert target_markers == ("Y",)
    assert definitions["add_ccr7"]["observed_markers"] == (
        "CD3",
        "CD197",
    )


def test_targeted_pair_definitions_can_skip_single_addbacks():
    config = {
        "panels": {
            "base": {"markers": ["CD3"]},
            "upper": {"markers": ["CD3", "CCR7", "PD-1"]},
        },
        "sweep": {
            "base_panel": "base",
            "upper_panel": "upper",
            "include_single_addbacks": False,
            "addback_panels": {
                "add_ccr7": "CCR7",
                "add_pd1": "PD-1",
            },
            "combination_panels": {
                "cytof": {
                    "pair_ccr7_pd1": ["CCR7", "PD-1"],
                }
            },
        },
    }
    definitions, _, target_markers = addback_panel_definitions(
        config,
        ("CD3", "CD197", "CD279", "Y"),
        modality="cytof",
    )
    assert tuple(definitions) == ("base", "pair_ccr7_pd1", "upper")
    assert definitions["pair_ccr7_pd1"]["kind"] == "targeted_pair"
    assert definitions["pair_ccr7_pd1"]["added_marker"] == "CCR7+PD-1"
    assert target_markers == ("Y",)


def test_custom_compact_and_upper_removal_panels():
    config = {
        "panels": {
            "base": {"markers": ["CD3"]},
            "upper": {"markers": ["CD3", "CCR7", "PD-1"]},
        },
        "sweep": {
            "base_panel": "base",
            "upper_panel": "upper",
            "include_single_addbacks": False,
            "addback_panels": {
                "add_ccr7": "CCR7",
                "add_pd1": "PD-1",
            },
            "custom_panels": {
                "cytof": {
                    "compact": {
                        "operation": "add_to_base",
                        "kind": "compact_candidate",
                        "markers": ["CCR7", "PD-1"],
                    },
                    "minus_pd1": {
                        "operation": "remove_from_upper",
                        "kind": "selected_removal",
                        "markers": ["PD-1"],
                    },
                }
            },
        },
    }
    definitions, _, _ = addback_panel_definitions(
        config,
        ("CD3", "CD197", "CD279", "Y"),
        modality="cytof",
    )
    assert definitions["compact"]["observed_markers"] == (
        "CD3",
        "CD197",
        "CD279",
    )
    assert definitions["minus_pd1"]["observed_markers"] == (
        "CD3",
        "CD197",
    )
    assert definitions["minus_pd1"]["added_marker"] == "REMOVE:PD-1"


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


def test_small_addback_sweep_reuses_fixed_targets_and_classifier(tmp_path):
    pytest.importorskip("torch")
    classes = PLATFORM_FINE_CLASSES["spectral_flow"]
    markers = ("CD3", "CD4", "Y_SIGNAL", "Y_NOISE")
    rng = np.random.RandomState(31)
    specimens = ("R0001_A", "R0002_A", "R0003_A")
    cache = tmp_path / "cache"
    for specimen_index, specimen in enumerate(specimens):
        labels = np.tile(np.asarray(classes), 15)
        class_index = np.tile(np.arange(len(classes)), 15).astype(np.float32)
        cd3 = class_index + rng.normal(scale=0.05, size=len(labels))
        cd4 = (class_index % 3) + rng.normal(scale=0.05, size=len(labels))
        y_signal = 2.0 * cd3 + 0.5 * cd4
        y_noise = rng.normal(size=len(labels)) + specimen_index * 0.01
        _write_cache(
            cache,
            specimen,
            markers,
            np.column_stack([cd3, cd4, y_signal, y_noise]),
            labels,
        )

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
        "experiment": {"name": "tiny_addback"},
        "data": {
            "cache_root": str(cache),
            "split_manifest": str(split_path),
        },
        "panels": {
            "base": {"markers": ["CD3"]},
            "upper": {"markers": ["CD3", "CD4"]},
        },
        "sweep": {
            "base_panel": "base",
            "upper_panel": "upper",
            "addback_panels": {"add_cd4": "CD4"},
        },
        "training": {
            "device": "cpu",
            "fit_sample_seed": 7,
            "cells_per_fit_patient": 120,
            "knn": {
                "k": 3,
                "max_reference_cells": 100,
                "query_chunk_size": 32,
                "distance_memory_fraction": 0.08,
            },
        },
        "classifier": {
            "c": 1.0,
            "max_iter": 200,
            "maximum_cells_per_class": 20,
            "sample_seed": 7,
            "seed": 7,
        },
        "output": {"root": str(output)},
    }
    result = run_addback_sweep(
        config,
        modality="spectral_flow",
        fold_index=0,
        seed=13,
    )
    assert result["status"] == "ok"
    assert result["primary_target_markers"] == ["Y_SIGNAL", "Y_NOISE"]
    marker_frame = pd.read_csv(result["artifacts"]["marker_metrics"])
    assert set(marker_frame["panel"]) == {"base", "add_cd4", "upper"}
    assert set(marker_frame["marker"]) == {"Y_SIGNAL", "Y_NOISE"}
    assert set(marker_frame["representation"]) == {"median", "knn"}
    left = marker_frame[marker_frame["panel"] == "add_cd4"].reset_index(
        drop=True
    )
    right = marker_frame[marker_frame["panel"] == "upper"].reset_index(
        drop=True
    )
    np.testing.assert_allclose(
        left["null_relative_skill"],
        right["null_relative_skill"],
    )
    biology = json.loads(
        (
            output
            / "tiny_addback/spectral_flow/fold_0/seed_13/biology_metrics.json"
        ).read_text()
    )
    representations = biology["panels"]["base"]["specimens"]["R0003_A"][
        "metrics"
    ]
    assert set(representations) == {
        "full_true",
        "full_hybrid_median",
        "full_hybrid_knn",
    }
    summary = summarize_addback_sweep(
        output / "tiny_addback",
        bootstrap_replicates=20,
        seed=17,
    )
    assert summary["completed_runs"] == 1
    global_effect = pd.read_csv(summary["artifacts"]["marker_global"])
    add_effect = global_effect[
        (global_effect["panel"] == "add_cd4")
        & (global_effect["representation"] == "knn")
    ].iloc[0]
    assert add_effect["null_relative_skill_delta_mean"] > 0
