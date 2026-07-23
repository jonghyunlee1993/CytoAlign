"""Initialize paired concept slots from population-prototype correlations."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def maximum_correlation_concept_order(
    source_prototypes: np.ndarray,
    target_prototypes: np.ndarray,
    *,
    epsilon: float = 1.0e-4,
) -> dict[str, np.ndarray]:
    """Match axes one-to-one, then rank pairs from most to least correlated."""

    source = np.asarray(source_prototypes, dtype=np.float64)
    target = np.asarray(target_prototypes, dtype=np.float64)
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("prototype arrays must be two-dimensional")
    if source.shape[0] != target.shape[0] or source.shape[0] < 2:
        raise ValueError("prototype arrays need aligned rows")
    if source.shape[1] != target.shape[1] or source.shape[1] < 1:
        raise ValueError("prototype arrays need the same non-zero concept dimension")
    source = source - source.mean(axis=0, keepdims=True)
    target = target - target.mean(axis=0, keepdims=True)
    source /= np.sqrt(source.var(axis=0, keepdims=True) + float(epsilon))
    target /= np.sqrt(target.var(axis=0, keepdims=True) + float(epsilon))
    correlation = source.T @ target / source.shape[0]
    source_indices, target_indices = linear_sum_assignment(-correlation)
    matched = correlation[source_indices, target_indices]
    ranking = np.argsort(-matched, kind="stable")
    return {
        "source_order": source_indices[ranking].astype(np.int64),
        "target_order": target_indices[ranking].astype(np.int64),
        "matched_correlation": matched[ranking].astype(np.float32),
        "correlation": correlation.astype(np.float32),
    }


def permute_platform_concepts_(model, order: np.ndarray) -> None:
    """Permute SAE encoder/decoder axes in place without changing its function."""

    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - optional dependency
        raise ImportError("PyTorch is required to permute SAE concepts") from error

    permutation = np.asarray(order, dtype=np.int64)
    n_concepts = int(model.n_concepts)
    if permutation.shape != (n_concepts,) or not np.array_equal(
        np.sort(permutation), np.arange(n_concepts)
    ):
        raise ValueError("order must be a permutation of every concept index")
    index = torch.as_tensor(
        permutation,
        device=model.sparse_autoencoder.concept_encoder.weight.device,
        dtype=torch.long,
    )
    encoder = model.sparse_autoencoder.concept_encoder
    decoder = model.sparse_autoencoder.latent_decoder
    with torch.no_grad():
        encoder.weight.copy_(encoder.weight[index].clone())
        if encoder.bias is not None:
            encoder.bias.copy_(encoder.bias[index].clone())
        decoder.weight.copy_(decoder.weight[:, index].clone())
