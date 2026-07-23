import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.data.aml import COARSE_CELL_TYPES, SpecimenData
from src.data.cross_panel import CrossPanelDataset
from src.models.cytoalign import CytoAlign
from src.training import experiment


def test_pooled_selection_discards_specimen_identity():
    view = {
        "source_h": {"B": np.asarray([[2.0]]), "A": np.asarray([[1.0]])},
        "source_x": {"B": np.asarray([[4.0]]), "A": np.asarray([[3.0]])},
        "source_labels": {"B": np.asarray(["T"]), "A": np.asarray(["B"])},
        "target_y": {"B": np.asarray([[6.0]]), "A": np.asarray([[5.0]])},
        "target_labels": {"B": np.asarray(["T"]), "A": np.asarray(["B"])},
        "patients": {"A": "R1", "B": "R2"},
    }

    pooled = experiment._pooled_view(view)
    predictions = experiment._pooled_predictions(
        {"B": np.asarray([[8.0]]), "A": np.asarray([[7.0]])}
    )

    assert list(pooled["source_h"]) == ["pooled"]
    assert pooled["source_h"]["pooled"].ravel().tolist() == [1.0, 2.0]
    assert pooled["target_y"]["pooled"].ravel().tolist() == [5.0, 6.0]
    assert pooled["patients"] == {"pooled": "pooled"}
    assert predictions["pooled"].ravel().tolist() == [7.0, 8.0]


def _specimen(specimen_id, source, rng):
    labels = np.repeat(COARSE_CELL_TYPES, 8)
    h = rng.normal(size=(len(labels), 2))
    x = rng.normal(size=(len(labels), 2))
    if source:
        values = np.column_stack([h, x])
        markers = ("H1", "H2", "X1", "X2")
    else:
        y = 0.5 * h + rng.normal(scale=0.1, size=h.shape)
        values = np.column_stack([h, y])
        markers = ("H1", "H2", "Y1", "Y2")
    return SpecimenData(
        modality="source" if source else "target",
        specimen_id=specimen_id,
        markers=markers,
        values=values.astype(np.float32),
        cell_types=labels.astype(object),
        original_row_indices=np.arange(len(labels)),
    )


@pytest.mark.parametrize("residual_baseline", ["ridge_hl", "knn_hl"])
def test_end_to_end_training_writes_model_and_metrics(
    tmp_path, monkeypatch, residual_baseline
):
    rng = np.random.RandomState(3)
    specimens = [f"R{index:04d}_A" for index in range(5)]
    dataset = CrossPanelDataset(
        source_modality="source",
        target_modality="target",
        common_markers=("H1", "H2"),
        source_common_columns=("H1", "H2"),
        target_common_columns=("H1", "H2"),
        source_exclusive_columns=("X1", "X2"),
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
    monkeypatch.setattr(experiment, "load_cross_panel_dataset", lambda _: dataset)
    config = {
        "experiment": {
            "name": f"test_{residual_baseline}",
            "fold": 0,
            "seed": 7,
        },
        "data": {},
        "preprocessing": {"n_knots": 17, "max_fit_cells": 1000},
        "training": {
            "device": "cpu",
            "max_fit_cells": 1000,
            "residual_baseline": residual_baseline,
            "ot": {
                "k_max": 6,
                "k_min": 4,
                "epsilon_ratio": 0.1,
                "sinkhorn_iterations": 20,
            },
            "mlp": {
                "hidden_dims": [8],
                "epochs": 1,
                "batch_size": 32,
                "learning_rate": 0.001,
                "patience": 1,
            },
            "knn": {"k": 3, "max_reference_cells": 1000, "n_jobs": 1},
        },
        "evaluation": {"alphas": [0.0, 1.0]},
        "output": {"root": str(tmp_path)},
    }

    result = experiment.run_experiment(config)

    assert set(result["methods"]) == {
        "global_median",
        "cell_type_median",
        "ridge_hl",
        "knn_hl",
        "mlp_hl",
        "ot_hl",
        "cytoalign",
    }
    output = tmp_path / f"test_{residual_baseline}" / "fold_0" / "seed_7"
    assert (output / "model.pkl").exists()
    assert (output / "metrics.json").exists()
    assert result["residual_baseline"] == residual_baseline
    assert result["pairing"] == "matched"
    assert result["selection_pairing"] == "matched"
    model = CytoAlign.load(result["model"])
    source = dataset.source[specimens[-1]]
    prediction = model.predict(
        source.values[:, :2],
        source.values[:, 2:],
        source.cell_types,
        device="cpu",
    )
    assert model.source_modality == "source"
    assert prediction.shape == (40, 2)
