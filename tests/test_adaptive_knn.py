import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.data.aml import SpecimenData
from src.data.cross_panel import CrossPanelDataset
from src.models.adaptive_knn import (
    AdaptiveMetricLearner,
    predict_with_candidate_reranking,
)
from src.training import adaptive_knn_experiment


def test_marker_specific_reranking_uses_different_neighbors():
    reference_common = np.asarray(
        [[0.0, 1.0], [0.1, 0.0], [1.0, 0.1], [1.0, 1.0]], dtype=np.float32
    )
    reference_targets = np.asarray(
        [[10.0, 100.0], [20.0, 200.0], [30.0, 300.0], [40.0, 400.0]],
        dtype=np.float32,
    )
    query = np.asarray([[0.0, 0.0]], dtype=np.float32)
    weights = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    prediction = predict_with_candidate_reranking(
        reference_common,
        reference_targets,
        query,
        {"adaptive": weights},
        np.asarray([True, True]),
        k=1,
        candidate_k=4,
        batch_size=2,
        device="cpu",
    )

    assert prediction["adaptive"].tolist() == [[10.0, 200.0]]
    assert prediction["plain_knn"].shape == (1, 2)


def test_panel_mask_renormalizes_weights():
    common = np.asarray([[0.0, 0.0], [1.0, 2.0], [2.0, 0.0]], dtype=np.float32)
    target = np.asarray([[0.0], [2.0], [4.0]], dtype=np.float32)
    query = np.asarray([[0.0, 1.9]], dtype=np.float32)
    weights = np.asarray([[0.99, 0.01]], dtype=np.float32)

    full = predict_with_candidate_reranking(
        common,
        target,
        query,
        {"adaptive": weights},
        np.asarray([True, True]),
        k=1,
        candidate_k=3,
        batch_size=2,
        device="cpu",
    )
    second_only = predict_with_candidate_reranking(
        common,
        target,
        query,
        {"adaptive": weights},
        np.asarray([False, True]),
        k=1,
        candidate_k=3,
        batch_size=2,
        device="cpu",
    )

    assert full["adaptive"].item() == 0.0
    assert second_only["adaptive"].item() == 2.0


def test_metric_learning_returns_simplex_weights():
    rng = np.random.RandomState(4)
    common = rng.normal(size=(80, 3)).astype(np.float32)
    targets = np.column_stack(
        [common[:, 0] + 0.05 * rng.normal(size=80), common[:, 2]]
    ).astype(np.float32)
    groups = np.repeat(["R1", "R2", "R3", "R4"], 20)
    learner = AdaptiveMetricLearner(
        mode="target",
        temperature=0.2,
        regularization=0.0,
        epochs=2,
        steps_per_epoch=3,
        query_batch_size=8,
        reference_batch_size=16,
        learning_rate=0.1,
        random_state=5,
    )

    fit = learner.fit(
        common,
        targets,
        groups,
        np.ones(2),
        panel_masks=[np.asarray([True, True, True])],
        mask_dropout=False,
        device="cpu",
    )

    assert fit.weights.shape == (2, 3)
    np.testing.assert_allclose(fit.weights.sum(axis=1), 1.0, atol=1e-6)
    assert len(fit.losses) == 2


def _specimen(name, source, rng):
    labels = np.repeat(["Blast", "T cell"], 6)
    common = rng.normal(size=(labels.size, 3))
    if source:
        values = np.column_stack([common, rng.normal(size=labels.size)])
        markers = ("H1", "H2", "H3", "X1")
    else:
        targets = np.column_stack([common[:, 0], common[:, 2]])
        values = np.column_stack([common, targets])
        markers = ("H1", "H2", "H3", "Y1", "Y2")
    return SpecimenData(
        modality="source" if source else "target",
        specimen_id=name,
        markers=markers,
        values=values.astype(np.float32),
        cell_types=labels,
        original_row_indices=np.arange(labels.size),
    )


def test_adaptive_experiment_writes_fold_metrics(tmp_path, monkeypatch):
    rng = np.random.RandomState(8)
    specimens = [f"R{index}_A" for index in range(5)]
    dataset = CrossPanelDataset(
        source_modality="source",
        target_modality="target",
        common_markers=("H1", "H2", "H3"),
        source_common_columns=("H1", "H2", "H3"),
        target_common_columns=("H1", "H2", "H3"),
        source_exclusive_columns=("X1",),
        target_exclusive_columns=("Y1", "Y2"),
        source={name: _specimen(name, True, rng) for name in specimens},
        target={name: _specimen(name, False, rng) for name in specimens},
        splits={
            "folds": [
                {
                    "train_specimens": specimens[:3],
                    "validation_specimens": specimens[3:4],
                    "test_specimens": specimens[4:],
                }
            ]
        },
    )
    monkeypatch.setattr(
        adaptive_knn_experiment, "load_cross_panel_dataset", lambda _: dataset
    )
    config = {
        "experiment": {
            "name": "adaptive_test",
            "runner": "adaptive_knn",
            "fold": 0,
            "seed": 9,
        },
        "data": {},
        "preprocessing": {"n_knots": 9, "max_fit_cells": 100},
        "training": {
            "device": "cpu",
            "adaptive_knn": {
                "k": 3,
                "candidate_k": 8,
                "max_fit_cells": 100,
                "max_reference_cells": 24,
                "epochs": 1,
                "steps_per_epoch": 1,
                "query_batch_size": 4,
                "reference_batch_size": 8,
                "inference_batch_size": 8,
                "learning_rate": 0.05,
                "temperatures": [0.1],
                "regularizations": [0.0],
                "panel_masks": {
                    "full": ["H1", "H2", "H3"],
                    "small": ["H1", "H3"],
                },
            },
        },
        "output": {"root": str(tmp_path)},
    }

    result = adaptive_knn_experiment.run_adaptive_knn_experiment(config)

    assert set(result["panel_results"]) == {"full", "small"}
    assert set(result["panel_results"]["full"]["methods"]) == {
        "plain_knn",
        "global_metric",
        "target_metric",
        "target_metric_panel_dropout",
    }
    assert result["hardware"]["device"] == "cpu"
    assert (tmp_path / "adaptive_test/fold_0/seed_9/adaptive_knn_metrics.json").is_file()
