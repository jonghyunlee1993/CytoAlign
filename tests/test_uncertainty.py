import numpy as np

from src.evaluation.uncertainty import evaluate_uncertainty
from src.models.cytofmerge import CyTOFMergeDiagnostics


def test_uncertainty_diagnostics_find_selective_correction_headroom():
    labels = np.asarray(["A"] * 10)
    baseline = {"sample": np.zeros((10, 1))}
    residual = {"sample": np.ones((10, 1))}
    view = {
        "source_labels": {"sample": labels},
        "target_labels": {"sample": labels},
        "target_y": {"sample": np.ones((10, 1))},
    }
    diagnostics = {
        "sample": CyTOFMergeDiagnostics(
            mean_neighbor_distance=np.ones(10),
            median_neighbor_mad=np.ones((10, 1)),
            effective_k=np.full(10, 5),
            used_fallback=np.zeros(10, dtype=bool),
        )
    }

    result = evaluate_uncertainty(
        baseline,
        residual,
        diagnostics,
        view,
        marker_scales=np.ones(1),
        alphas=[0.0, 1.0],
        selected_alpha=1.0,
        marker_names=["Y1"],
    )

    assert result["n_units"] == 1
    assert result["mean_baseline_error"] == 1.0
    assert result["mean_selected_error"] == 0.0
    assert result["mean_oracle_gain"] == 1.0
    assert result["markers"]["Y1"]["oracle_active_fraction"] == 1.0
