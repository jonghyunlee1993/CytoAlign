import numpy as np
import pytest

from src.models.cell_type import CommonCellTypeClassifier
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
    plain = HOnlyRegressor(architecture="ridge", ridge_alpha=0.01).fit(
        common[train], target[train]
    )
    conditional = HOnlyRegressor(
        architecture="ridge", condition_on_cell_type=True, ridge_alpha=0.01
    ).fit(common[train], target[train], cell_types=labels[train])

    plain_mae = np.mean(np.abs(plain.predict(common[test]) - target[test]))
    conditional_mae = np.mean(
        np.abs(
            conditional.predict(common[test], cell_types=labels[test]) - target[test]
        )
    )
    assert conditional_mae < 0.02 * plain_mae


def test_h_only_mlp_supports_multioutput_and_cell_types():
    common, target, labels = _conditional_regression_data()
    model = HOnlyRegressor(
        architecture="mlp",
        condition_on_cell_type=True,
        hidden_dims=(16,),
        max_iter=300,
        random_state=3,
    ).fit(common[:300], target[:300], cell_types=labels[:300])
    prediction = model.predict(common[300:], cell_types=labels[300:])
    assert prediction.shape == (100, 2)
    assert np.isfinite(prediction).all()


def test_conditioned_model_rejects_unknown_cell_type():
    common, target, labels = _conditional_regression_data()
    model = HOnlyRegressor(condition_on_cell_type=True).fit(
        common[:100], target[:100], cell_types=labels[:100]
    )
    with pytest.raises(ValueError, match="Unknown cell types"):
        model.predict(common[:1], cell_types=["NK cell"])


def test_median_controls_and_common_cell_type_classifier():
    common, target, labels = _conditional_regression_data()
    global_prediction = GlobalMedianRegressor().fit(target[:300]).predict(5)
    typed_prediction = CellTypeMedianRegressor().fit(
        target[:300], labels[:300]
    ).predict(labels[300:305])
    assert global_prediction.shape == typed_prediction.shape == (5, 2)

    classifier_common = np.column_stack(
        [np.where(labels == "Blast", -3.0, 3.0), common[:, 0]]
    )
    classifier = CommonCellTypeClassifier().fit(classifier_common[:300], labels[:300])
    assert np.mean(classifier.predict(classifier_common[300:]) == labels[300:]) > 0.99

