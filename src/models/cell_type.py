"""Deployability sensitivity: predict coarse cell type from common markers."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class CommonCellTypeClassifier:
    def __init__(self, *, regularization: float = 1.0, max_iter: int = 500):
        self.regularization = float(regularization)
        self.max_iter = int(max_iter)

    def fit(
        self, common: np.ndarray, cell_types: Sequence
    ) -> "CommonCellTypeClassifier":
        values = np.asarray(common, dtype=np.float64)
        labels = np.asarray(cell_types)
        if values.ndim != 2 or labels.ndim != 1 or values.shape[0] != labels.size:
            raise ValueError("common and cell_types have incompatible shapes")
        self.scaler_ = StandardScaler().fit(values)
        self.model_ = LogisticRegression(
            C=self.regularization,
            max_iter=self.max_iter,
            class_weight="balanced",
            random_state=42,
        ).fit(self.scaler_.transform(values), labels)
        return self

    def predict(self, common: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("Classifier has not been fitted")
        values = np.asarray(common, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("common must be two-dimensional")
        return self.model_.predict(self.scaler_.transform(values))

    def predict_proba(self, common: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("Classifier has not been fitted")
        values = np.asarray(common, dtype=np.float64)
        return self.model_.predict_proba(self.scaler_.transform(values))

