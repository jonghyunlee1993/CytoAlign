import numpy as np

from src.evaluation.population_metrics import evaluate_matched_populations


def test_matched_population_metrics_reward_correct_unpaired_distributions():
    target = {
        "P1_A": np.asarray([[0.0], [1.0], [2.0], [3.0]]),
        "P2_A": np.asarray([[1.0], [2.0], [3.0], [4.0]]),
    }
    labels = {name: np.asarray(["T"] * 4) for name in target}
    patients = {"P1_A": "P1", "P2_A": "P2"}
    exact = evaluate_matched_populations(
        target,
        labels,
        target,
        labels,
        patients,
        np.asarray([1.0]),
        minimum_cells=2,
    )
    shifted = evaluate_matched_populations(
        {name: values + 2.0 for name, values in target.items()},
        labels,
        target,
        labels,
        patients,
        np.asarray([1.0]),
        minimum_cells=2,
    )
    assert exact["patient_first_normalized_wasserstein"] == 0.0
    assert shifted["patient_first_normalized_wasserstein"] > 1.0
    assert exact["n_patients"] == 2
