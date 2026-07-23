import numpy as np
import pytest

from src.preprocessing.common_space import (
    CrossPanelCommonSpace,
    EmpiricalPercentileTransformer,
)


def test_empirical_percentile_transform_round_trip():
    values = np.column_stack([np.linspace(-2, 2, 101), np.linspace(10, 30, 101)])
    transformer = EmpiricalPercentileTransformer(n_knots=33).fit(values)
    percentiles = transformer.transform(values)
    restored = transformer.inverse_transform(percentiles)
    assert percentiles.min() >= 0
    assert percentiles.max() <= 1
    np.testing.assert_allclose(restored, values, atol=1.0e-5)


def test_constant_marker_is_supported():
    values = np.ones((20, 1)) * 7.5
    transformer = EmpiricalPercentileTransformer().fit(values)
    percentile = transformer.transform([[7.5]])
    restored = transformer.inverse_transform(percentile)
    np.testing.assert_allclose(percentile, [[0.5]], atol=1.0e-6)
    np.testing.assert_allclose(restored, [[7.5]], atol=1.0e-6)


def test_cross_panel_mapping_uses_only_fitted_support():
    source_train = np.linspace(-2, 2, 101)[:, None]
    target_train = np.linspace(80, 120, 101)[:, None]
    common = CrossPanelCommonSpace.fit(source_train, target_train, n_knots=33)

    mapped = common.source_to_target(np.asarray([[0.0], [1000.0]]))
    assert mapped[0, 0] == pytest.approx(100.0, abs=0.1)
    assert mapped[1, 0] == pytest.approx(120.0, abs=0.1)


def test_cross_panel_dimension_mismatch_is_rejected():
    with pytest.raises(ValueError, match="dimensions differ"):
        CrossPanelCommonSpace.fit(np.ones((5, 2)), np.ones((5, 3)))
