"""Common-marker-only deterministic regression baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def _two_dimensional(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    return array


@dataclass
class _CellTypeFeatures:
    enabled: bool
    interactions: bool
    classes_: tuple | None = None

    def fit(self, labels: Sequence | None) -> None:
        if not self.enabled:
            self.classes_ = None
            return
        if labels is None:
            raise ValueError("cell_types are required when conditioning is enabled")
        values = np.asarray(labels)
        if values.ndim != 1:
            raise ValueError("cell_types must be one-dimensional")
        self.classes_ = tuple(np.unique(values).tolist())

    def transform(self, common: np.ndarray, labels: Sequence | None) -> np.ndarray:
        if not self.enabled:
            return common
        if labels is None or self.classes_ is None:
            raise ValueError("cell_types are required when conditioning is enabled")
        values = np.asarray(labels)
        if values.ndim != 1 or values.shape[0] != common.shape[0]:
            raise ValueError("cell_types do not align with common-marker rows")
        lookup = {label: index for index, label in enumerate(self.classes_)}
        unknown = sorted({value for value in np.unique(values) if value not in lookup})
        if unknown:
            raise ValueError(f"Unknown cell types at prediction: {unknown}")
        one_hot = np.zeros((values.size, len(self.classes_)), dtype=np.float64)
        for row, value in enumerate(values):
            one_hot[row, lookup[value]] = 1.0
        pieces = [common, one_hot]
        if self.interactions:
            interactions = (common[:, :, None] * one_hot[:, None, :]).reshape(
                common.shape[0], -1
            )
            pieces.append(interactions)
        return np.concatenate(pieces, axis=1)


class GlobalMedianRegressor:
    """Target-training global marker median."""

    def fit(self, target_exclusive: np.ndarray) -> "GlobalMedianRegressor":
        target = _two_dimensional(target_exclusive, "target_exclusive")
        self.median_ = np.median(target, axis=0)
        return self

    def predict(self, n_rows: int) -> np.ndarray:
        if not hasattr(self, "median_"):
            raise RuntimeError("Regressor has not been fitted")
        return np.repeat(self.median_[None, :], int(n_rows), axis=0).astype(np.float32)


class CellTypeMedianRegressor:
    """Target-training marker median within each coarse cell type."""

    def fit(
        self, target_exclusive: np.ndarray, cell_types: Sequence
    ) -> "CellTypeMedianRegressor":
        target = _two_dimensional(target_exclusive, "target_exclusive")
        labels = np.asarray(cell_types)
        if labels.ndim != 1 or labels.size != target.shape[0]:
            raise ValueError("cell_types do not align with target rows")
        self.global_median_ = np.median(target, axis=0)
        self.medians_ = {
            label: np.median(target[labels == label], axis=0)
            for label in np.unique(labels)
        }
        return self

    def predict(self, cell_types: Sequence) -> np.ndarray:
        if not hasattr(self, "medians_"):
            raise RuntimeError("Regressor has not been fitted")
        labels = np.asarray(cell_types)
        if labels.ndim != 1:
            raise ValueError("cell_types must be one-dimensional")
        output = np.empty((labels.size, self.global_median_.size), dtype=np.float32)
        for row, label in enumerate(labels):
            output[row] = self.medians_.get(label, self.global_median_)
        return output


class HOnlyRegressor:
    """Ridge or MLP prediction of target-exclusive markers from shared H."""

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
        if architecture not in {"ridge", "mlp"}:
            raise ValueError("architecture must be 'ridge' or 'mlp'")
        self.architecture = architecture
        self.condition_on_cell_type = bool(condition_on_cell_type)
        self.ridge_alpha = float(ridge_alpha)
        self.hidden_dims = tuple(map(int, hidden_dims))
        self.mlp_alpha = float(mlp_alpha)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)

    def fit(
        self,
        common: np.ndarray,
        target_exclusive: np.ndarray,
        *,
        cell_types: Sequence | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> "HOnlyRegressor":
        common_array = _two_dimensional(common, "common")
        target = _two_dimensional(target_exclusive, "target_exclusive")
        if common_array.shape[0] != target.shape[0]:
            raise ValueError("common and target_exclusive row counts differ")
        self.feature_builder_ = _CellTypeFeatures(
            enabled=self.condition_on_cell_type,
            interactions=self.architecture == "ridge",
        )
        self.feature_builder_.fit(cell_types)
        features = self.feature_builder_.transform(common_array, cell_types)
        self.feature_scaler_ = StandardScaler().fit(features)
        self.target_scaler_ = StandardScaler().fit(target)
        scaled_features = self.feature_scaler_.transform(features)
        scaled_target = self.target_scaler_.transform(target)
        if self.architecture == "ridge":
            self.model_ = Ridge(alpha=self.ridge_alpha)
            self.model_.fit(
                scaled_features,
                scaled_target,
                sample_weight=None if sample_weight is None else np.asarray(sample_weight),
            )
        else:
            if sample_weight is not None:
                raise ValueError("sample_weight is not supported by sklearn MLPRegressor")
            self.model_ = MLPRegressor(
                hidden_layer_sizes=self.hidden_dims,
                activation="relu",
                alpha=self.mlp_alpha,
                max_iter=self.max_iter,
                random_state=self.random_state,
                early_stopping=False,
            )
            self.model_.fit(scaled_features, scaled_target)
        self.n_common_markers_ = common_array.shape[1]
        self.n_target_markers_ = target.shape[1]
        return self

    def predict(
        self, common: np.ndarray, *, cell_types: Sequence | None = None
    ) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("Regressor has not been fitted")
        common_array = _two_dimensional(common, "common")
        if common_array.shape[1] != self.n_common_markers_:
            raise ValueError("Unexpected common-marker dimension")
        features = self.feature_builder_.transform(common_array, cell_types)
        scaled_prediction = self.model_.predict(self.feature_scaler_.transform(features))
        if scaled_prediction.ndim == 1:
            scaled_prediction = scaled_prediction[:, None]
        prediction = self.target_scaler_.inverse_transform(scaled_prediction)
        return prediction.astype(np.float32)

