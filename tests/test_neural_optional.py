import numpy as np
import pytest

from src.models.autoencoder import (
    DeterministicAutoencoder,
    PlatformConceptAutoencoder,
    TopKSparseAutoencoder,
    torch_available,
)
from src.models.concept_bridge import ConceptResidualBridge


def test_neural_modules_have_an_import_safe_optional_dependency_boundary():
    if not torch_available():
        with pytest.raises(ImportError, match="PyTorch"):
            DeterministicAutoencoder(4, 2)
        return

    import torch

    autoencoder = DeterministicAutoencoder(4, 2, hidden_dims=(3,))
    sparse = TopKSparseAutoencoder(2, 5, 2)
    model = PlatformConceptAutoencoder(autoencoder, sparse)
    values = torch.randn(7, 4)
    output = model(values)
    assert output["reconstruction"].shape == (7, 4)
    assert torch.all((output["concepts"] > 0).sum(dim=1) <= 2)

    target_autoencoder = DeterministicAutoencoder(4, 2, hidden_dims=(3,))
    target_sparse = TopKSparseAutoencoder(2, 5, 2)
    target_model = PlatformConceptAutoencoder(target_autoencoder, target_sparse)
    bridge = ConceptResidualBridge(
        model,
        target_model,
        np.eye(5, dtype=np.float32),
        target_residual_indices=(2, 3),
    )
    bridged = bridge(values, alpha=0.0)
    assert torch.count_nonzero(bridged["residual"]) == 0
