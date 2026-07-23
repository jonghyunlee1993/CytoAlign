"""Optional PyTorch deterministic AE and Top-K sparse concept dictionary.

The classical baseline package remains usable without PyTorch.  Neural classes
are defined behind an import-safe boundary and raise a focused error only when
instantiated in an environment without the optional dependency.
"""

from __future__ import annotations

from typing import Sequence

try:  # pragma: no cover - exercised in the neural environment
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - current lightweight test env
    torch = None
    nn = None


def torch_available() -> bool:
    return torch is not None


def require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for CycleAlign neural concept models. "
            "Install the optional dependencies from requirements-neural.txt."
        )


if nn is not None:  # pragma: no branch

    def _activation(name: str) -> nn.Module:
        lookup = {
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "silu": nn.SiLU,
        }
        try:
            return lookup[str(name).lower()]()
        except KeyError as error:
            raise ValueError(f"Unsupported activation: {name}") from error


    def _mlp(
        dimensions: Sequence[int], *, activation: str, dropout: float
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        for index, (left, right) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            layers.append(nn.Linear(int(left), int(right)))
            if index < len(dimensions) - 2:
                layers.append(_activation(activation))
                if float(dropout) > 0:
                    layers.append(nn.Dropout(float(dropout)))
        return nn.Sequential(*layers)


    class DeterministicAutoencoder(nn.Module):
        """Small platform-specific autoencoder with an explicit latent API."""

        def __init__(
            self,
            input_dim: int,
            latent_dim: int,
            *,
            hidden_dims: Sequence[int] = (64, 32),
            activation: str = "gelu",
            dropout: float = 0.0,
        ):
            super().__init__()
            if int(input_dim) < 1 or int(latent_dim) < 1:
                raise ValueError("input_dim and latent_dim must be positive")
            hidden = tuple(map(int, hidden_dims))
            if any(value < 1 for value in hidden):
                raise ValueError("hidden dimensions must be positive")
            if not 0 <= float(dropout) < 1:
                raise ValueError("dropout must lie in [0, 1)")
            self.input_dim = int(input_dim)
            self.latent_dim = int(latent_dim)
            self.encoder = _mlp(
                (self.input_dim, *hidden, self.latent_dim),
                activation=activation,
                dropout=dropout,
            )
            self.decoder = _mlp(
                (self.latent_dim, *reversed(hidden), self.input_dim),
                activation=activation,
                dropout=dropout,
            )

        def encode(self, values: torch.Tensor) -> torch.Tensor:
            return self.encoder(values)

        def decode(self, latent: torch.Tensor) -> torch.Tensor:
            return self.decoder(latent)

        def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            latent = self.encode(values)
            return self.decode(latent), latent


    class TopKSparseAutoencoder(nn.Module):
        """Overcomplete non-negative Top-K dictionary over a frozen AE latent."""

        def __init__(
            self,
            latent_dim: int,
            n_concepts: int,
            top_k: int,
            *,
            encoder_bias: bool = True,
            decoder_bias: bool = True,
        ):
            super().__init__()
            if int(latent_dim) < 1 or int(n_concepts) < 1:
                raise ValueError("latent_dim and n_concepts must be positive")
            if not 1 <= int(top_k) <= int(n_concepts):
                raise ValueError("top_k must lie between one and n_concepts")
            self.latent_dim = int(latent_dim)
            self.n_concepts = int(n_concepts)
            self.top_k = int(top_k)
            self.concept_encoder = nn.Linear(
                self.latent_dim, self.n_concepts, bias=bool(encoder_bias)
            )
            self.latent_decoder = nn.Linear(
                self.n_concepts, self.latent_dim, bias=bool(decoder_bias)
            )

        def encode(self, latent: torch.Tensor) -> torch.Tensor:
            dense = torch.relu(self.concept_encoder(latent))
            values, indices = torch.topk(dense, self.top_k, dim=-1, sorted=False)
            sparse = torch.zeros_like(dense)
            return sparse.scatter(-1, indices, values)

        def decode(self, concepts: torch.Tensor) -> torch.Tensor:
            return self.latent_decoder(concepts)

        def forward(
            self, latent: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            concepts = self.encode(latent)
            return self.decode(concepts), concepts


    class PlatformConceptAutoencoder(nn.Module):
        """Composition of a platform AE and a post-hoc sparse dictionary."""

        def __init__(
            self,
            autoencoder: DeterministicAutoencoder,
            sparse_autoencoder: TopKSparseAutoencoder,
        ):
            super().__init__()
            if autoencoder.latent_dim != sparse_autoencoder.latent_dim:
                raise ValueError("AE and SAE latent dimensions differ")
            self.autoencoder = autoencoder
            self.sparse_autoencoder = sparse_autoencoder

        @property
        def n_concepts(self) -> int:
            return self.sparse_autoencoder.n_concepts

        def encode_concepts(self, values: torch.Tensor) -> torch.Tensor:
            latent = self.autoencoder.encode(values)
            return self.sparse_autoencoder.encode(latent)

        def decode_concepts(self, concepts: torch.Tensor) -> torch.Tensor:
            latent = self.sparse_autoencoder.decode(concepts)
            return self.autoencoder.decode(latent)

        def forward(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
            latent = self.autoencoder.encode(values)
            latent_reconstruction, concepts = self.sparse_autoencoder(latent)
            reconstruction = self.autoencoder.decode(latent_reconstruction)
            return {
                "reconstruction": reconstruction,
                "latent": latent,
                "latent_reconstruction": latent_reconstruction,
                "concepts": concepts,
            }

        @torch.no_grad()
        def common_intervention_effects(
            self,
            values: torch.Tensor,
            common_indices: Sequence[int],
            *,
            delta: float = 1.0,
        ) -> torch.Tensor:
            """Return ``cell x concept x common`` decoder intervention effects."""

            concepts = self.encode_concepts(values)
            baseline = self.decode_concepts(concepts)
            indices = torch.as_tensor(
                tuple(map(int, common_indices)), device=values.device, dtype=torch.long
            )
            if indices.numel() == 0 or torch.any(indices < 0) or torch.any(
                indices >= baseline.shape[1]
            ):
                raise ValueError("common_indices are empty or out of bounds")
            effects = []
            for concept_index in range(self.n_concepts):
                perturbed = concepts.clone()
                perturbed[:, concept_index] += float(delta)
                decoded = self.decode_concepts(perturbed)
                effects.append((decoded[:, indices] - baseline[:, indices])[:, None, :])
            return torch.cat(effects, dim=1)


else:

    class _TorchRequired:
        def __init__(self, *args, **kwargs):
            require_torch()


    class DeterministicAutoencoder(_TorchRequired):
        pass


    class TopKSparseAutoencoder(_TorchRequired):
        pass


    class PlatformConceptAutoencoder(_TorchRequired):
        pass
