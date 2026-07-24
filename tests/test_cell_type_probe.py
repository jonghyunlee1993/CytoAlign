import numpy as np

from src.training.cell_type_probe import (
    FINE_CELL_TYPES,
    probe_metrics,
    shuffle_rows_within_groups,
)


def test_probe_metrics_reports_rare_class_auprc():
    labels = np.asarray(FINE_CELL_TYPES)
    predictions = labels.copy()
    probabilities = np.full((len(labels), len(labels)), 0.01)
    np.fill_diagonal(probabilities, 0.93)
    result = probe_metrics(labels, predictions, probabilities, FINE_CELL_TYPES)
    assert result["balanced_accuracy"] == 1.0
    assert result["rare_t_macro_auprc"] == 1.0
    assert result["per_class"]["T cell DP"]["recall"] == 1.0


def test_shuffle_rows_within_groups_preserves_group_marginals():
    values = np.arange(12).reshape(6, 2)
    groups = np.asarray(["a", "a", "a", "b", "b", "b"])
    shuffled = shuffle_rows_within_groups(values, groups, seed=7)
    assert sorted(map(tuple, shuffled[:3])) == sorted(map(tuple, values[:3]))
    assert sorted(map(tuple, shuffled[3:])) == sorted(map(tuple, values[3:]))
    assert not np.array_equal(shuffled, values)
