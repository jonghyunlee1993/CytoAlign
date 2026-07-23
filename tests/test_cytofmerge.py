import numpy as np

from src.models.cytofmerge import CyTOFMergeRegressor, patient_balanced_indices


def test_cell_type_condition_improves_ambiguous_common_space():
    reference_common = np.zeros((40, 2), dtype=np.float32)
    reference_labels = np.asarray(["Blast"] * 20 + ["T cell"] * 20)
    reference_target = np.where(
        (reference_labels == "Blast")[:, None],
        np.asarray([[-5.0, -2.0]]),
        np.asarray([[5.0, 2.0]]),
    )
    query_common = np.zeros((20, 2), dtype=np.float32)
    query_labels = np.asarray(["Blast"] * 10 + ["T cell"] * 10)
    truth = np.where(
        (query_labels == "Blast")[:, None],
        np.asarray([[-5.0, -2.0]]),
        np.asarray([[5.0, 2.0]]),
    )

    plain = CyTOFMergeRegressor(k=10, max_reference_cells=None).fit(
        reference_common, reference_target
    )
    conditional = CyTOFMergeRegressor(
        k=10, condition_on_cell_type=True, max_reference_cells=None
    ).fit(
        reference_common,
        reference_target,
        reference_cell_types=reference_labels,
    )
    plain_error = np.mean(np.abs(plain.predict(query_common) - truth))
    conditional_prediction, diagnostics = conditional.predict(
        query_common, cell_types=query_labels, return_diagnostics=True
    )
    conditional_error = np.mean(np.abs(conditional_prediction - truth))
    assert conditional_error == 0.0
    assert conditional_error < plain_error
    assert not diagnostics.used_fallback.any()
    assert np.all(diagnostics.effective_k == 10)


def test_unknown_query_type_uses_explicit_global_fallback():
    common = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    target = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    labels = np.asarray(["A", "A", "B", "B"])
    model = CyTOFMergeRegressor(
        k=2, condition_on_cell_type=True, max_reference_cells=None
    ).fit(common, target, reference_cell_types=labels)
    prediction, diagnostics = model.predict(
        [[1.5]], cell_types=["unknown"], return_diagnostics=True
    )
    assert prediction.shape == (1, 1)
    assert diagnostics.used_fallback.tolist() == [True]


def test_patient_balanced_reference_cap_is_deterministic_and_not_dominated():
    groups = np.asarray(["large"] * 90 + ["small"] * 10)
    first = patient_balanced_indices(groups, 100, maximum=20, seed=7)
    second = patient_balanced_indices(groups, 100, maximum=20, seed=7)
    np.testing.assert_array_equal(first, second)
    selected_groups = groups[first]
    assert len(first) == 20
    assert np.sum(selected_groups == "small") == 10
    assert np.sum(selected_groups == "large") == 10
