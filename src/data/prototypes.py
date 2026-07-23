"""Leakage-safe specimen-by-cell-type pools for population prototypes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PrototypeGroupPool:
    specimen: str
    cell_type: str
    source_indices: np.ndarray
    target_indices: np.ndarray

    @property
    def key(self) -> str:
        return f"{self.specimen}|{self.cell_type}"


def build_disjoint_prototype_pools(
    specimens: Sequence,
    cell_types: Sequence,
    *,
    minimum_cells_per_side: int = 8,
    random_state: int = 42,
) -> tuple[PrototypeGroupPool, ...]:
    """Partition every specimen×type stratum into non-overlapping modalities."""

    specimen_array = np.asarray(specimens).astype(str)
    label_array = np.asarray(cell_types).astype(str)
    if specimen_array.ndim != 1 or label_array.shape != specimen_array.shape:
        raise ValueError("specimens and cell_types must be aligned vectors")
    if int(minimum_cells_per_side) < 1:
        raise ValueError("minimum_cells_per_side must be positive")
    rng = np.random.RandomState(int(random_state))
    result = []
    for specimen, label in sorted(set(zip(specimen_array, label_array))):
        rows = np.flatnonzero((specimen_array == specimen) & (label_array == label))
        if rows.size < 2 * int(minimum_cells_per_side):
            continue
        rows = rng.permutation(rows)
        midpoint = rows.size // 2
        source = np.sort(rows[:midpoint])
        target = np.sort(rows[midpoint:])
        result.append(
            PrototypeGroupPool(
                specimen=str(specimen),
                cell_type=str(label),
                source_indices=source,
                target_indices=target,
            )
        )
    if len(result) < 2:
        raise ValueError("Fewer than two specimen×cell-type prototype groups remain")
    return tuple(result)


def shuffled_target_group_order(
    groups: Sequence[PrototypeGroupPool], *, random_state: int
) -> np.ndarray:
    """Derange specimen identity within each cell type for a negative control."""

    group_tuple = tuple(groups)
    if not group_tuple:
        raise ValueError("groups must not be empty")
    rng = np.random.RandomState(int(random_state))
    order = np.arange(len(group_tuple), dtype=np.int64)
    labels = sorted({group.cell_type for group in group_tuple})
    for label in labels:
        current = np.asarray(
            [index for index, group in enumerate(group_tuple) if group.cell_type == label],
            dtype=np.int64,
        )
        if current.size < 2:
            raise ValueError(f"Cell type {label} has fewer than two groups")
        shuffled = rng.permutation(current)
        if np.any(shuffled == current):
            shift = int(rng.randint(1, current.size))
            shuffled = np.roll(current, shift)
        order[current] = shuffled
    return order
