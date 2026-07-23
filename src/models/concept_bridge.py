"""Optional PyTorch cross-decoding through an explicit sparse concept graph."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.models.autoencoder import PlatformConceptAutoencoder, require_torch

try:  # pragma: no cover - exercised in the neural environment
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - current lightweight test env
    torch = None
    nn = None


if nn is not None:  # pragma: no branch

    class ConceptMappingLayer(nn.Module):
        """Fixed or support-constrained trainable source-to-target graph."""

        def __init__(self, initial_weights: np.ndarray, *, trainable: bool = False):
            super().__init__()
            weights = np.asarray(initial_weights, dtype=np.float32)
            if weights.ndim != 2 or np.any(weights < 0) or not np.isfinite(weights).all():
                raise ValueError("initial_weights must be a finite non-negative matrix")
            row_sums = weights.sum(axis=1)
            if np.any((row_sums > 1.0e-6) & (np.abs(row_sums - 1.0) > 1.0e-5)):
                raise ValueError("Matched concept-graph rows must sum to one")
            support = weights > 0
            self.trainable_mapping = bool(trainable)
            self.register_buffer("support", torch.as_tensor(support, dtype=torch.bool))
            self.register_buffer(
                "matched_rows", torch.as_tensor(row_sums > 0, dtype=torch.float32)
            )
            if trainable:
                logits = np.zeros_like(weights, dtype=np.float32)
                logits[support] = np.log(np.maximum(weights[support], 1.0e-8))
                self.logits = nn.Parameter(torch.as_tensor(logits))
            else:
                self.register_buffer("fixed_weights", torch.as_tensor(weights))

        def normalized_weights(self) -> torch.Tensor:
            if not self.trainable_mapping:
                return self.fixed_weights
            masked = self.logits.masked_fill(~self.support, -1.0e9)
            weights = torch.softmax(masked, dim=1)
            weights = weights * self.support.to(weights.dtype)
            denominator = weights.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
            weights = weights / denominator
            return weights * self.matched_rows[:, None]

        def forward(self, source_concepts: torch.Tensor) -> torch.Tensor:
            return source_concepts @ self.normalized_weights()


    class ConceptResidualBridge(nn.Module):
        """Decode full-source concept activation as a target residual prediction.

        The class intentionally returns only residuals.  The caller must use a
        fold-fitted :class:`PanelResidualizer` to add them to H-only, making the
        ``alpha=0`` evaluation path exact and easy to test.
        """

        def __init__(
            self,
            source_model: PlatformConceptAutoencoder,
            target_model: PlatformConceptAutoencoder,
            concept_weights: np.ndarray,
            target_residual_indices: Sequence[int],
            *,
            trainable_mapping: bool = False,
            freeze_dictionaries: bool = True,
            zero_initial_residual_scale: bool = True,
        ):
            super().__init__()
            weights = np.asarray(concept_weights)
            if weights.shape != (source_model.n_concepts, target_model.n_concepts):
                raise ValueError("Concept graph dimensions do not match platform models")
            indices = tuple(map(int, target_residual_indices))
            if not indices or min(indices) < 0:
                raise ValueError("target_residual_indices must be non-empty and non-negative")
            self.source_model = source_model
            self.target_model = target_model
            self.mapping = ConceptMappingLayer(
                weights, trainable=bool(trainable_mapping)
            )
            self.register_buffer(
                "target_residual_indices", torch.as_tensor(indices, dtype=torch.long)
            )
            initial_scale = torch.zeros(len(indices)) if zero_initial_residual_scale else torch.ones(len(indices))
            self.residual_scale = nn.Parameter(initial_scale)
            if freeze_dictionaries:
                self.source_model.requires_grad_(False)
                self.target_model.requires_grad_(False)

        def forward(
            self, source_values: torch.Tensor, *, alpha: float = 1.0
        ) -> dict[str, torch.Tensor]:
            source_concepts = self.source_model.encode_concepts(source_values)
            target_concepts = self.mapping(source_concepts)
            target_panel = self.target_model.decode_concepts(target_concepts)
            indices = self.target_residual_indices
            if torch.any(indices >= target_panel.shape[1]):
                raise ValueError("target_residual_indices exceed decoder output dimension")
            residual = target_panel[:, indices] * self.residual_scale
            return {
                "residual": float(alpha) * residual,
                "source_concepts": source_concepts,
                "target_concepts": target_concepts,
                "mapping_weights": self.mapping.normalized_weights(),
            }


else:

    class _TorchRequired:
        def __init__(self, *args, **kwargs):
            require_torch()


    class ConceptMappingLayer(_TorchRequired):
        pass


    class ConceptResidualBridge(_TorchRequired):
        pass
