"""Leakage-safe adaptation of the CyTOFmerge kNN-median core rule."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _seed(seed: int, value: object) -> int:
    digest = hashlib.sha256(f"{seed}|{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def _matrix(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    return array


def patient_balanced_indices(
    groups: Sequence | None, n_rows: int, maximum: int | None, seed: int
) -> np.ndarray:
    """Deterministically cap a bank with approximately equal patient quotas."""
    indices = np.arange(int(n_rows), dtype=np.int64)
    if maximum is None or indices.size <= int(maximum):
        return indices
    rng = np.random.RandomState(int(seed))
    if groups is None:
        return np.sort(rng.choice(indices, int(maximum), replace=False))
    group_array = np.asarray(groups)
    if group_array.ndim != 1 or group_array.size != indices.size:
        raise ValueError("reference_groups do not align with reference rows")
    unique = np.unique(group_array)
    quota = max(1, int(np.ceil(int(maximum) / unique.size)))
    selected = []
    for group in unique:
        current = indices[group_array == group]
        if current.size > quota:
            current = rng.choice(current, quota, replace=False)
        selected.append(current)
    result = np.concatenate(selected)
    if result.size > int(maximum):
        result = rng.choice(result, int(maximum), replace=False)
    return np.sort(result)


@dataclass(frozen=True)
class CyTOFMergeDiagnostics:
    mean_neighbor_distance: np.ndarray
    effective_k: np.ndarray
    used_fallback: np.ndarray


@dataclass
class _ReferenceBank:
    common: np.ndarray
    target: np.ndarray
    index: NearestNeighbors


class CyTOFMergeRegressor:
    """Marker-wise median of common-space nearest target-training cells."""

    def __init__(
        self,
        *,
        k: int = 50,
        condition_on_cell_type: bool = False,
        max_reference_cells: int | None = 50_000,
        n_jobs: int = 1,
        query_chunk_size: int = 10_000,
        random_state: int = 4207,
    ):
        if int(k) < 1:
            raise ValueError("k must be positive")
        self.k = int(k)
        self.condition_on_cell_type = bool(condition_on_cell_type)
        self.max_reference_cells = (
            None if max_reference_cells is None else int(max_reference_cells)
        )
        self.n_jobs = int(n_jobs)
        self.query_chunk_size = int(query_chunk_size)
        self.random_state = int(random_state)

    def _bank(
        self,
        common: np.ndarray,
        target: np.ndarray,
        groups: Sequence | None,
        seed: int,
    ) -> _ReferenceBank:
        selected = patient_balanced_indices(
            groups, common.shape[0], self.max_reference_cells, seed
        )
        selected_common = common[selected]
        selected_target = target[selected]
        index = NearestNeighbors(
            n_neighbors=min(self.k, selected.size),
            metric="euclidean",
            algorithm="auto",
            n_jobs=self.n_jobs,
        ).fit(selected_common)
        return _ReferenceBank(selected_common, selected_target, index)

    def fit(
        self,
        reference_common: np.ndarray,
        reference_target_exclusive: np.ndarray,
        *,
        reference_cell_types: Sequence | None = None,
        reference_groups: Sequence | None = None,
    ) -> "CyTOFMergeRegressor":
        common = _matrix(reference_common, "reference_common")
        target = _matrix(reference_target_exclusive, "reference_target_exclusive")
        if common.shape[0] != target.shape[0] or common.shape[0] == 0:
            raise ValueError(
                "Reference common and target rows must be equal and non-empty"
            )
        if not np.isfinite(common).all():
            raise ValueError("reference_common contains non-finite values")
        groups = None if reference_groups is None else np.asarray(reference_groups)
        if groups is not None and (groups.ndim != 1 or groups.size != common.shape[0]):
            raise ValueError("reference_groups do not align with reference rows")
        self.global_bank_ = self._bank(common, target, groups, self.random_state)
        self.type_banks_: dict[object, _ReferenceBank] = {}
        if self.condition_on_cell_type:
            if reference_cell_types is None:
                raise ValueError(
                    "reference_cell_types are required when conditioning is enabled"
                )
            labels = np.asarray(reference_cell_types)
            if labels.ndim != 1 or labels.size != common.shape[0]:
                raise ValueError(
                    "reference_cell_types do not align with reference rows"
                )
            for label in np.unique(labels):
                rows = labels == label
                local_groups = None if groups is None else groups[rows]
                self.type_banks_[label] = self._bank(
                    common[rows],
                    target[rows],
                    local_groups,
                    _seed(self.random_state, label),
                )
        self.n_common_markers_ = common.shape[1]
        self.n_target_markers_ = target.shape[1]
        return self

    def _predict_bank(
        self, bank: _ReferenceBank, query: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        prediction = np.empty(
            (query.shape[0], self.n_target_markers_), dtype=np.float32
        )
        mean_distance = np.empty(query.shape[0], dtype=np.float32)
        effective_k = np.empty(query.shape[0], dtype=np.int32)
        for left in range(0, query.shape[0], self.query_chunk_size):
            right = min(query.shape[0], left + self.query_chunk_size)
            distances, neighbors = bank.index.kneighbors(query[left:right])
            prediction[left:right] = np.median(bank.target[neighbors], axis=1)
            mean_distance[left:right] = distances.mean(axis=1)
            effective_k[left:right] = neighbors.shape[1]
        return prediction, mean_distance, effective_k

    def predict(
        self,
        query_common: np.ndarray,
        *,
        cell_types: Sequence | None = None,
        return_diagnostics: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, CyTOFMergeDiagnostics]:
        if not hasattr(self, "global_bank_"):
            raise RuntimeError("Regressor has not been fitted")
        query = _matrix(query_common, "query_common")
        if query.shape[1] != self.n_common_markers_:
            raise ValueError("Unexpected common-marker dimension")
        if not np.isfinite(query).all():
            raise ValueError("query_common contains non-finite values")
        if not self.condition_on_cell_type:
            prediction, distance, effective_k = self._predict_bank(
                self.global_bank_, query
            )
            fallback = np.zeros(query.shape[0], dtype=bool)
        else:
            if cell_types is None:
                raise ValueError("cell_types are required when conditioning is enabled")
            labels = np.asarray(cell_types)
            if labels.ndim != 1 or labels.size != query.shape[0]:
                raise ValueError("cell_types do not align with query rows")
            prediction = np.empty(
                (query.shape[0], self.n_target_markers_), dtype=np.float32
            )
            distance = np.empty(query.shape[0], dtype=np.float32)
            effective_k = np.empty(query.shape[0], dtype=np.int32)
            fallback = np.zeros(query.shape[0], dtype=bool)
            for label in np.unique(labels):
                rows = np.flatnonzero(labels == label)
                bank = self.type_banks_.get(label)
                if bank is None:
                    bank = self.global_bank_
                    fallback[rows] = True
                current_prediction, current_distance, current_k = self._predict_bank(
                    bank, query[rows]
                )
                prediction[rows] = current_prediction
                distance[rows] = current_distance
                effective_k[rows] = current_k
        diagnostics = CyTOFMergeDiagnostics(distance, effective_k, fallback)
        return (prediction, diagnostics) if return_diagnostics else prediction
