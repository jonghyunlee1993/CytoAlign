import numpy as np

from src.evaluation.exact_metrics import (
    evaluate_exact_cells,
    select_residual_alpha,
)


def test_exact_metrics_are_patient_first_and_alpha_zero_nests_baseline():
    target = {
        "P1_A": np.asarray([[0.0], [2.0]]),
        "P1_B": np.asarray([[2.0], [4.0]]),
        "P2_A": np.asarray([[10.0], [12.0]]),
    }
    baseline = {name: np.zeros_like(values) for name, values in target.items()}
    residual = {name: values.copy() for name, values in target.items()}
    patients = {"P1_A": "P1", "P1_B": "P1", "P2_A": "P2"}
    alpha, scores = select_residual_alpha(
        baseline,
        residual,
        target,
        patients,
        np.ones(1),
        (0.0, 0.5, 1.0),
    )
    assert alpha == 1.0
    assert scores["0.0"] > scores["0.5"] > scores["1.0"]
    metrics = evaluate_exact_cells(residual, target, patients, np.ones(1))
    assert metrics["patient_first_normalized_mae"] == 0.0
    assert metrics["patient_first_normalized_rmse"] == 0.0
    assert np.isclose(metrics["macro_patient_marker_spearman"], 1.0)
