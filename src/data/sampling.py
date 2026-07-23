"""Deterministic patient-by-cell-type balancing for neural and bridge models."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def patient_cell_type_balanced_indices(
    patient_ids: Sequence,
    cell_types: Sequence,
    *,
    maximum_per_stratum: int,
    random_state: int = 42,
) -> np.ndarray:
    """Cap each patient-by-cell-type stratum without duplicating cells."""

    patients = np.asarray(patient_ids)
    labels = np.asarray(cell_types)
    if patients.ndim != 1 or labels.ndim != 1 or patients.size != labels.size:
        raise ValueError("patient_ids and cell_types must be aligned one-dimensional arrays")
    if int(maximum_per_stratum) < 1:
        raise ValueError("maximum_per_stratum must be positive")
    rng = np.random.RandomState(int(random_state))
    selected: list[np.ndarray] = []
    strata = sorted({(str(patient), str(label)) for patient, label in zip(patients, labels)})
    for patient, label in strata:
        rows = np.flatnonzero(
            (patients.astype(str) == patient) & (labels.astype(str) == label)
        )
        if rows.size > int(maximum_per_stratum):
            rows = rng.choice(rows, int(maximum_per_stratum), replace=False)
        selected.append(np.sort(rows))
    if not selected:
        return np.empty(0, dtype=np.int64)
    return np.sort(np.concatenate(selected).astype(np.int64, copy=False))
