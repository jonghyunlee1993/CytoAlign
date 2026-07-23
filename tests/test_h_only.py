import numpy as np
import pytest

from src.models.h_only import (
    CellTypeMedianRegressor,
    GlobalMedianRegressor,
    HOnlyRegressor,
)


def _conditional_regression_data(seed=1):
    rng = np.random.RandomState(seed)
    common = rng.normal(size=(400, 2))
    cell_types = np.where(np.arange(400) % 2 == 0, "Blast", "T cell")
    offset = np.where(cell_types == "Blast", -6.0, 6.0)
    target = np.column_stack(
        [2.0 * common[:, 0] - common[:, 1] + offset, common[:, 1] + 0.5 * offset]
    )
    return common, target, cell_types


def test_cell_type_condition_has_measurable_value_for_ridge():
    common, target, labels = _conditional_regression_data()
    train = np.arange(300)
    test = np.arange(300, 400)
    plain = HOnlyRegressor(ridge_alpha=0.01).fit(common[train], target[train])
    conditional = HOnlyRegressor(condition_on_cell_type=True, ridge_alpha=0.01).fit(
        common[train], target[train], cell_types=labels[train]
    )

    plain_mae = np.mean(np.abs(plain.predict(common[test]) - target[test]))
    conditional_mae = np.mean(
        np.abs(
            conditional.predict(common[test], cell_types=labels[test]) - target[test]
        )
    )
    assert conditional_mae < 0.02 * plain_mae


def test_conditioned_model_rejects_unknown_cell_type():
    common, target, labels = _conditional_regression_data()
    model = HOnlyRegressor(condition_on_cell_type=True).fit(
        common[:100], target[:100], cell_types=labels[:100]
    )
    with pytest.raises(ValueError, match="Unknown cell types"):
        model.predict(common[:1], cell_types=["NK cell"])


def test_median_controls():
    _, target, labels = _conditional_regression_data()
    global_prediction = GlobalMedianRegressor().fit(target[:300]).predict(5)
    typed_prediction = (
        CellTypeMedianRegressor()
        .fit(target[:300], labels[:300])
        .predict(labels[300:305])
    )
    assert global_prediction.shape == typed_prediction.shape == (5, 2)
