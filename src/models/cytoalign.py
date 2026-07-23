"""Deployable CytoAlign baseline-plus-residual predictor."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np

from src.models.cytofmerge import CyTOFMergeRegressor
from src.models.h_only import HOnlyRegressor
from src.models.mlp import MLPRegressor
from src.preprocessing.common_space import CrossPanelCommonSpace


def encode_features(
    common: np.ndarray,
    exclusive: np.ndarray | None,
    cell_types: Sequence,
    classes: Sequence[str],
) -> np.ndarray:
    labels = np.asarray(cell_types).astype(str)
    lookup = {label: index for index, label in enumerate(classes)}
    one_hot = np.zeros((labels.size, len(classes)), dtype=np.float32)
    for index, label in enumerate(labels):
        one_hot[index, lookup[label]] = 1.0
    pieces = [np.asarray(common, dtype=np.float32)]
    if exclusive is not None:
        pieces.append(np.asarray(exclusive, dtype=np.float32))
    pieces.append(one_hot)
    return np.concatenate(pieces, axis=1)


class CytoAlign:
    def __init__(
        self,
        *,
        common_space: CrossPanelCommonSpace,
        baseline: HOnlyRegressor | CyTOFMergeRegressor,
        residual: MLPRegressor | None,
        classes: Sequence[str],
        alpha: float,
        source_modality: str,
        target_modality: str,
        source_common_columns: Sequence[str],
        source_exclusive_columns: Sequence[str],
        target_markers: Sequence[str],
    ):
        self.common_space = common_space
        self.baseline = baseline
        self.residual = residual
        self.classes = tuple(classes)
        self.alpha = float(alpha)
        self.source_modality = str(source_modality)
        self.target_modality = str(target_modality)
        self.source_common_columns = tuple(source_common_columns)
        self.source_exclusive_columns = tuple(source_exclusive_columns)
        self.target_markers = tuple(target_markers)

    def predict(
        self,
        source_common: np.ndarray,
        source_exclusive: np.ndarray,
        cell_types: Sequence,
        *,
        device: str = "cpu",
    ) -> np.ndarray:
        common = self.common_space.source_percentiles(source_common)
        baseline = self.baseline.predict(common, cell_types=cell_types)
        if self.residual is None or self.alpha == 0:
            return baseline
        features = encode_features(common, source_exclusive, cell_types, self.classes)
        residual = self.residual.predict(features, device=device)
        return baseline + self.alpha * residual

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if self.residual is not None:
            self.residual.cpu()
        with output.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "CytoAlign":
        with Path(path).open("rb") as handle:
            return pickle.load(handle)
