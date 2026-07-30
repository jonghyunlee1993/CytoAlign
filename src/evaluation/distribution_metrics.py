"""Multivariate distribution metrics shared across benchmark methods."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import wasserstein_distance

from src.benchmark.artifacts import LockedFeatureScales


def make_sliced_wasserstein_projections(
    n_features: int,
    *,
    n_projections: int = 128,
    random_state: int = 4207,
) -> np.ndarray:
    """Create deterministic unit projections that can be frozen and shared."""

    if isinstance(n_features, bool) or not isinstance(n_features, int) or n_features < 1:
        raise ValueError("n_features must be a positive integer")
    if (
        isinstance(n_projections, bool)
        or not isinstance(n_projections, int)
        or n_projections < 1
    ):
        raise ValueError("n_projections must be a positive integer")
    rng = np.random.RandomState(int(random_state))
    projections = rng.normal(size=(n_projections, n_features))
    projections /= np.linalg.norm(projections, axis=1)[:, None]
    return projections


def _validate_rows(
    row_indices: Sequence[int] | np.ndarray | None,
    *,
    n_rows: int,
    maximum: int | None,
    random_state: int,
) -> np.ndarray:
    if row_indices is not None:
        rows = np.asarray(row_indices)
        if rows.ndim != 1 or rows.size == 0:
            raise ValueError("row_indices must be a non-empty vector")
        if not np.issubdtype(rows.dtype, np.integer):
            raise ValueError("row_indices must contain integers")
        rows = rows.astype(np.int64, copy=False)
        if np.any(rows < 0) or np.any(rows >= n_rows) or len(np.unique(rows)) != len(rows):
            raise ValueError("row_indices are out of range or duplicated")
        return rows
    if maximum is None or n_rows <= maximum:
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.RandomState(int(random_state))
    return np.sort(rng.choice(n_rows, maximum, replace=False))


def sliced_wasserstein_distance(
    left: np.ndarray,
    right: np.ndarray,
    feature_scales: LockedFeatureScales,
    *,
    marker_names: Sequence[str],
    projections: np.ndarray | None = None,
    left_row_indices: Sequence[int] | np.ndarray | None = None,
    right_row_indices: Sequence[int] | np.ndarray | None = None,
    n_projections: int = 128,
    max_rows: int | None = 5000,
    random_state: int = 4207,
) -> float:
    """Compute scaled sliced 1-Wasserstein using auditable shared artifacts."""

    if not isinstance(feature_scales, LockedFeatureScales):
        raise TypeError("feature_scales must be a LockedFeatureScales artifact")
    markers = tuple(marker_names)
    if markers != feature_scales.marker_names:
        raise ValueError("marker_names do not match the locked scale artifact")
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.ndim != 2 or right_array.ndim != 2:
        raise ValueError("left and right must be matrices")
    if left_array.shape[1] != right_array.shape[1] or left_array.shape[1] != len(markers):
        raise ValueError("Feature dimensions do not align")
    if left_array.shape[0] == 0 or right_array.shape[0] == 0:
        raise ValueError("left and right must be non-empty")
    if not np.isfinite(left_array).all() or not np.isfinite(right_array).all():
        raise ValueError("left and right must contain finite values")
    if max_rows is not None and (
        isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1
    ):
        raise ValueError("max_rows must be a positive integer when provided")

    if projections is None:
        projection_matrix = make_sliced_wasserstein_projections(
            len(markers),
            n_projections=n_projections,
            random_state=random_state,
        )
    else:
        projection_matrix = np.asarray(projections, dtype=np.float64)
        if (
            projection_matrix.ndim != 2
            or projection_matrix.shape[0] == 0
            or projection_matrix.shape[1] != len(markers)
            or not np.isfinite(projection_matrix).all()
        ):
            raise ValueError("projections must be a finite non-empty matrix")
        norms = np.linalg.norm(projection_matrix, axis=1)
        if np.any(norms <= 0):
            raise ValueError("projection rows must have positive norm")
        projection_matrix = projection_matrix / norms[:, None]

    left_rows = _validate_rows(
        left_row_indices,
        n_rows=left_array.shape[0],
        maximum=max_rows,
        random_state=int(random_state) + 104729,
    )
    if (
        right_row_indices is None
        and left_row_indices is None
        and left_array.shape == right_array.shape
        and np.array_equal(left_array, right_array)
    ):
        right_rows = left_rows
    else:
        right_rows = _validate_rows(
            right_row_indices,
            n_rows=right_array.shape[0],
            maximum=max_rows,
            random_state=int(random_state) + 130363,
        )

    scaled_left = left_array[left_rows] / feature_scales.values
    scaled_right = right_array[right_rows] / feature_scales.values
    distances = [
        wasserstein_distance(
            scaled_left @ projection,
            scaled_right @ projection,
        )
        for projection in projection_matrix
    ]
    return float(np.mean(distances))
