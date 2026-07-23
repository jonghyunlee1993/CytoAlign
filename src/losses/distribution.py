"""Sliced Wasserstein losses for cell-unpaired specimen populations."""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - exercised in the neural environment
    import torch
except ModuleNotFoundError:  # pragma: no cover - current lightweight test env
    torch = None


def _matrix(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _quantile_grid(size: int) -> np.ndarray:
    return (np.arange(int(size), dtype=np.float64) + 0.5) / float(size)


def numpy_sliced_wasserstein(
    source: np.ndarray,
    target: np.ndarray,
    *,
    n_projections: int = 32,
    power: int = 1,
    random_state: int = 42,
) -> float:
    """Compare unequal-size populations using shared random 1-D projections."""

    left = _matrix(source, "source")
    right = _matrix(target, "target")
    if left.shape[1] != right.shape[1]:
        raise ValueError("source and target feature dimensions differ")
    if int(n_projections) < 1 or int(power) not in {1, 2}:
        raise ValueError("n_projections must be positive and power must be one or two")
    rng = np.random.RandomState(int(random_state))
    directions = rng.normal(size=(left.shape[1], int(n_projections)))
    directions /= np.linalg.norm(directions, axis=0, keepdims=True).clip(1.0e-12)
    left_projection = left @ directions
    right_projection = right @ directions
    quantile_count = max(left.shape[0], right.shape[0])
    probabilities = _quantile_grid(quantile_count)
    left_quantiles = np.quantile(left_projection, probabilities, axis=0)
    right_quantiles = np.quantile(right_projection, probabilities, axis=0)
    distance = np.mean(np.abs(left_quantiles - right_quantiles) ** int(power))
    return float(distance if int(power) == 1 else np.sqrt(distance))


def torch_sliced_wasserstein(
    source,
    target,
    *,
    n_projections: int = 32,
    power: int = 1,
    random_state: int = 42,
    maximum_quantiles: int = 512,
):
    """Differentiable torch counterpart used by specimen-level bridge training."""

    if torch is None:
        raise ImportError("PyTorch is required for torch_sliced_wasserstein")
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("source and target must be aligned two-dimensional tensors")
    if source.shape[0] == 0 or target.shape[0] == 0:
        raise ValueError("source and target populations must not be empty")
    if int(n_projections) < 1 or int(power) not in {1, 2}:
        raise ValueError("n_projections must be positive and power must be one or two")
    generator = torch.Generator(device=source.device)
    generator.manual_seed(int(random_state))
    directions = torch.randn(
        source.shape[1],
        int(n_projections),
        dtype=source.dtype,
        device=source.device,
        generator=generator,
    )
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1.0e-12)
    left_projection = source @ directions
    right_projection = target @ directions
    count = min(
        int(maximum_quantiles), max(int(source.shape[0]), int(target.shape[0]))
    )
    probabilities = (
        torch.arange(count, device=source.device, dtype=source.dtype) + 0.5
    ) / float(count)
    left_quantiles = torch.quantile(left_projection, probabilities, dim=0)
    right_quantiles = torch.quantile(right_projection, probabilities, dim=0)
    distance = torch.mean(torch.abs(left_quantiles - right_quantiles) ** int(power))
    return distance if int(power) == 1 else torch.sqrt(distance)
