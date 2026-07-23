import numpy as np

from src.losses.distribution import numpy_sliced_wasserstein


def test_numpy_sliced_wasserstein_handles_unequal_population_sizes():
    rng = np.random.RandomState(21)
    source = rng.normal(size=(80, 4))
    identical = source.copy()
    shifted = rng.normal(size=(113, 4)) + 2.0
    assert numpy_sliced_wasserstein(source, identical) < 1.0e-12
    assert numpy_sliced_wasserstein(source, shifted) > 0.5
