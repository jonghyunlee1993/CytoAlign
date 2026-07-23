import json
import sys

import numpy as np

from scripts.direct_ot_decur_aux_experiment import (
    main as direct_ot_decur_aux_main,
)
from scripts.ot_new_sample_experiment import main as ot_new_sample_main
from src.data.pseudo_panels import build_two_sided_pseudo_panel_manifest


def synthetic_cache(tmp_path):
    cache = tmp_path / "cache"
    specimen_dir = cache / "specimens"
    specimen_dir.mkdir(parents=True)
    markers = ("H1", "X1", "Y1", "H2", "X2", "Y2")
    pseudo = build_two_sided_pseudo_panel_manifest(
        markers,
        common_markers=("H1", "H2"),
        source_exclusive_markers=("X1", "X2"),
        target_exclusive_markers=("Y1", "Y2"),
        require_complete_partition=True,
    )
    specimens = tuple(f"R{index:04d}_A" for index in range(1, 7))
    labels = np.repeat(
        ["Blast", "Monocyte", "T cell", "B cell", "NK cell"],
        20,
    )
    rng = np.random.RandomState(43)
    records = {}
    for specimen in specimens:
        h = rng.normal(size=(100, 2)).astype(np.float32)
        x = (h + rng.normal(scale=0.1, size=(100, 2))).astype(np.float32)
        y = (x + rng.normal(scale=0.1, size=(100, 2))).astype(np.float32)
        values = np.column_stack(
            [h[:, 0], x[:, 0], y[:, 0], h[:, 1], x[:, 1], y[:, 1]]
        )
        path = specimen_dir / f"{specimen}.npz"
        np.savez(
            path,
            values=values.astype(np.float32),
            cell_types=labels.astype(str),
            original_row_indices=np.arange(100),
        )
        records[specimen] = {
            "cache_file": str(path),
            "n_cells": 100,
        }
    manifest = {
        "status": "ok",
        "sampling": {
            "kind": "synthetic",
            "cells_per_specimen": 100,
        },
        "pseudo_panel": pseudo.to_dict(),
        "specimens": records,
        "split_manifest": {
            "folds": [
                {
                    "train_specimens": list(specimens[:2]),
                    "validation_specimens": list(specimens[2:4]),
                    "test_specimens": list(specimens[4:]),
                }
            ]
        },
    }
    (cache / "manifest.json").write_text(json.dumps(manifest))
    return cache


def test_ot_new_sample_runner_one_epoch_smoke(tmp_path, monkeypatch):
    output = tmp_path / "ot_new_sample.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ot_new_sample_experiment.py",
            "--cache",
            str(synthetic_cache(tmp_path)),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--k-max",
            "8",
            "--k-min",
            "4",
            "--paired-epsilon-ratio",
            "0.1",
            "--pooled-epsilon-ratio",
            "0.2",
            "--sinkhorn-iterations",
            "100",
            "--mlp-epochs",
            "1",
            "--batch-size",
            "32",
            "--alphas",
            "0",
            "1",
        ],
    )
    ot_new_sample_main()
    result = json.loads(output.read_text())
    assert result["status"] == "ok"
    assert result["contract"]["teacher_generation_split"] == "train_only"
    assert not result["contract"]["test_target_reference_used_for_inference"]
    assert not result["contract"]["hidden_source_y_used_for_training"]
    assert (
        result["methods"]["baseline_hl"]["test"]["exact_cell"]["n_cells"]
        == 200
    )
    assert "mlp_paired_ot_hxl" in result["methods"]
    assert "mlp_pooled_ot_hxl" in result["methods"]


def test_direct_ot_decur_aux_runner_one_epoch_smoke(tmp_path, monkeypatch):
    output = tmp_path / "direct_ot_decur_aux.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "direct_ot_decur_aux_experiment.py",
            "--cache",
            str(synthetic_cache(tmp_path)),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--k-max",
            "8",
            "--k-min",
            "4",
            "--sinkhorn-iterations",
            "100",
            "--epochs",
            "1",
            "--batch-size",
            "32",
            "--patience",
            "1",
            "--hidden-dim",
            "8",
            "--projection-dim",
            "4",
            "--n-common",
            "2",
            "--aux-blocks-per-step",
            "2",
            "--alphas",
            "0",
            "1",
        ],
    )
    direct_ot_decur_aux_main()
    result = json.loads(output.read_text())
    assert result["status"] == "ok"
    assert not result["contract"]["target_encoder_used_at_inference"]
    assert not result["contract"]["prediction_decoder_receives_projection_output"]
    assert set(result["methods"]) == {
        "direct_ot",
        "paired_decur_aux",
        "uniform_decur_aux",
    }
    assert (
        result["methods"]["paired_decur_aux"]["test"]["exact_cell"]["n_cells"]
        == 200
    )
