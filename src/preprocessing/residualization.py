"""Train-only H-conditioned residualization for concept models."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.models.h_only import HOnlyRegressor


def _matrix(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    return array


class PanelResidualizer:
    """Represent exclusive markers as standardized residuals beyond H-only.

    A separate instance is fitted for each platform and outer fold.  The
    baseline predictor and residual scaler are both learned from training
    patients only.  The inverse operation adds a predicted residual back to the
    same H-only baseline, which makes the final concept bridge explicitly
    nested in H-only.
    """

    def __init__(
        self,
        *,
        architecture: str = "ridge",
        condition_on_cell_type: bool = False,
        ridge_alpha: float = 1.0,
        hidden_dims: Sequence[int] = (128, 128),
        mlp_alpha: float = 1.0e-4,
        max_iter: int = 300,
        random_state: int = 42,
    ):
        self.architecture = str(architecture)
        self.condition_on_cell_type = bool(condition_on_cell_type)
        self.ridge_alpha = float(ridge_alpha)
        self.hidden_dims = tuple(map(int, hidden_dims))
        self.mlp_alpha = float(mlp_alpha)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)

    def fit(
        self,
        common: np.ndarray,
        exclusive: np.ndarray,
        *,
        cell_types: Sequence | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> "PanelResidualizer":
        common_array = _matrix(common, "common")
        exclusive_array = _matrix(exclusive, "exclusive")
        if common_array.shape[0] != exclusive_array.shape[0]:
            raise ValueError("common and exclusive row counts differ")
        self.predictor_ = HOnlyRegressor(
            architecture=self.architecture,
            condition_on_cell_type=self.condition_on_cell_type,
            ridge_alpha=self.ridge_alpha,
            hidden_dims=self.hidden_dims,
            mlp_alpha=self.mlp_alpha,
            max_iter=self.max_iter,
            random_state=self.random_state,
        ).fit(
            common_array,
            exclusive_array,
            cell_types=cell_types,
            sample_weight=sample_weight,
        )
        baseline = self.predictor_.predict(common_array, cell_types=cell_types)
        residual = exclusive_array - baseline
        self.residual_scaler_ = StandardScaler().fit(residual)
        self.n_common_markers_ = common_array.shape[1]
        self.n_exclusive_markers_ = exclusive_array.shape[1]
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "predictor_") or not hasattr(self, "residual_scaler_"):
            raise RuntimeError("Residualizer has not been fitted")

    def predict_baseline(
        self, common: np.ndarray, *, cell_types: Sequence | None = None
    ) -> np.ndarray:
        self._check_fitted()
        return self.predictor_.predict(common, cell_types=cell_types)

    def transform(
        self,
        common: np.ndarray,
        exclusive: np.ndarray,
        *,
        cell_types: Sequence | None = None,
        standardized: bool = True,
    ) -> np.ndarray:
        self._check_fitted()
        common_array = _matrix(common, "common")
        exclusive_array = _matrix(exclusive, "exclusive")
        if common_array.shape[0] != exclusive_array.shape[0]:
            raise ValueError("common and exclusive row counts differ")
        baseline = self.predict_baseline(common_array, cell_types=cell_types)
        residual = exclusive_array - baseline
        if standardized:
            residual = self.residual_scaler_.transform(residual)
        return np.asarray(residual, dtype=np.float32)

    def inverse_transform(
        self,
        common: np.ndarray,
        residual: np.ndarray,
        *,
        cell_types: Sequence | None = None,
        standardized: bool = True,
        alpha: float = 1.0,
    ) -> np.ndarray:
        """Add a residual prediction to H-only; ``alpha=0`` is exactly H-only."""

        self._check_fitted()
        common_array = _matrix(common, "common")
        residual_array = _matrix(residual, "residual")
        if common_array.shape[0] != residual_array.shape[0]:
            raise ValueError("common and residual row counts differ")
        if residual_array.shape[1] != self.n_exclusive_markers_:
            raise ValueError("Unexpected residual marker dimension")
        if standardized:
            residual_array = self.residual_scaler_.inverse_transform(residual_array)
        baseline = self.predict_baseline(common_array, cell_types=cell_types)
        return np.asarray(baseline + float(alpha) * residual_array, dtype=np.float32)
