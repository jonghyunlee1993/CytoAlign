import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.concepts.prototype_alignment import (
    maximum_correlation_concept_order,
    permute_platform_concepts_,
)
from src.models.autoencoder import (
    DeterministicAutoencoder,
    PlatformConceptAutoencoder,
    TopKSparseAutoencoder,
)


def test_maximum_correlation_matching_recovers_permuted_axes():
    rng = np.random.RandomState(13)
    source = rng.normal(size=(1000, 5))
    target_order = np.asarray([3, 0, 4, 1, 2])
    target = source[:, target_order] + rng.normal(scale=0.01, size=source.shape)
    result = maximum_correlation_concept_order(source, target)
    aligned_source = source[:, result["source_order"]]
    aligned_target = target[:, result["target_order"]]
    diagonal = np.diag(np.corrcoef(aligned_source.T, aligned_target.T)[:5, 5:])
    assert np.min(diagonal) > 0.99


def test_concept_permutation_preserves_platform_reconstruction():
    torch.manual_seed(7)
    model = PlatformConceptAutoencoder(
        DeterministicAutoencoder(3, 2, hidden_dims=(4,)),
        TopKSparseAutoencoder(2, 5, 2),
    )
    values = torch.randn(20, 3)
    before = model(values)["reconstruction"].detach().clone()
    permute_platform_concepts_(model, np.asarray([2, 4, 1, 0, 3]))
    after = model(values)["reconstruction"].detach()
    assert torch.allclose(before, after, atol=1.0e-6)
