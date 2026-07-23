import numpy as np

from src.preprocessing.residualization import PanelResidualizer


def test_panel_residualizer_round_trip_and_exact_h_only_nesting():
    rng = np.random.RandomState(4)
    common = rng.normal(size=(200, 3))
    exclusive = np.column_stack(
        [
            2.0 * common[:, 0] - common[:, 1] + rng.normal(scale=0.1, size=200),
            -0.5 * common[:, 2] + rng.normal(scale=0.2, size=200),
        ]
    )
    model = PanelResidualizer(architecture="ridge", ridge_alpha=1.0e-6).fit(
        common, exclusive
    )
    residual = model.transform(common, exclusive)
    reconstructed = model.inverse_transform(common, residual)
    baseline = model.predict_baseline(common)
    nested = model.inverse_transform(common, residual, alpha=0.0)

    np.testing.assert_allclose(reconstructed, exclusive, atol=1.0e-5)
    np.testing.assert_array_equal(nested, baseline)
