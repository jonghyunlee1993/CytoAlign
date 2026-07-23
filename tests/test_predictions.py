import numpy as np
import pytest

from src.evaluation.predictions import PredictionBundle


def _bundle():
    return PredictionBundle(
        predictions=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        patient_ids=np.asarray(["R0001", "R0001"]),
        specimen_ids=np.asarray(["R0001_A", "R0001_A"]),
        source_row_indices=np.asarray([10, 11]),
        cell_types=np.asarray(["Blast", "T cell"]),
        target_markers=("CD11c", "CD13"),
        metadata={
            "method": "cytofmerge_h_l",
            "direction": "sf_to_cytof",
            "fold": 0,
            "label_source": "oracle",
            "config_hash": "abc",
        },
        diagnostics={
            "mean_neighbor_distance": np.asarray([0.1, 0.2]),
            "used_fallback": np.asarray([False, True]),
        },
    )


def test_prediction_bundle_pickle_free_round_trip(tmp_path):
    path = tmp_path / "prediction.npz"
    _bundle().save(path)
    restored = PredictionBundle.load(path)
    np.testing.assert_array_equal(restored.predictions, _bundle().predictions)
    np.testing.assert_array_equal(restored.patient_ids, _bundle().patient_ids)
    np.testing.assert_array_equal(
        restored.diagnostics["used_fallback"], [False, True]
    )
    assert restored.target_markers == ("CD11c", "CD13")
    assert restored.metadata["method"] == "cytofmerge_h_l"


def test_prediction_bundle_rejects_misaligned_diagnostics():
    bundle = _bundle()
    bundle.diagnostics["bad"] = np.asarray([1])
    with pytest.raises(ValueError, match="does not align"):
        bundle.validate()

