"""Source-blind null models for cross-panel distribution benchmarks."""

from __future__ import annotations

import hashlib

import numpy as np

from src.benchmark.artifacts import TrainingReference


def _stable_seed(base_seed: int, token: object) -> int:
    digest = hashlib.sha256(f"{int(base_seed)}|{token}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


class TargetPriorSampler:
    """Sample complete Y rows from target-training patients without using H.

    Patients and their specimens are sampled uniformly before a cell is sampled.
    This prevents patients or cell-rich visits from dominating the null and
    preserves the empirical Y--Y joint structure of each selected row.
    """

    def __init__(self, *, random_state: int = 4207):
        self.random_state = int(random_state)

    def fit(
        self,
        reference: TrainingReference,
    ) -> "TargetPriorSampler":
        if not isinstance(reference, TrainingReference):
            raise TypeError("reference must be a TrainingReference")
        if reference.bank_role != "null_prior_bank":
            raise ValueError("TargetPriorSampler requires a null_prior_bank")
        patients = tuple(sorted(set(reference.patient_ids)))
        self.target_ = np.asarray(reference.values)
        self.patients_ = patients
        patient_groups = np.asarray(reference.patient_ids, dtype=object)
        specimen_groups = np.asarray(reference.specimen_ids, dtype=object)
        self.rows_by_patient_ = {
            patient: np.flatnonzero(patient_groups == patient) for patient in patients
        }
        self.specimens_by_patient_ = {
            patient: tuple(
                sorted(set(specimen_groups[self.rows_by_patient_[patient]].tolist()))
            )
            for patient in patients
        }
        self.rows_by_specimen_ = {
            specimen: np.flatnonzero(specimen_groups == specimen)
            for specimen in sorted(set(reference.specimen_ids))
        }
        self.n_markers_ = reference.values.shape[1]
        self.marker_names_ = reference.marker_names
        self.reference_bank_id_ = reference.reference_bank_id
        self.reference_manifest_digest_ = reference.manifest_digest
        self.fold_ = reference.fold
        return self

    def predict(
        self,
        n_rows: int,
        *,
        token: object = "default",
    ) -> np.ndarray:
        """Return a deterministic source-blind draw for one query unit."""

        if not hasattr(self, "target_"):
            raise RuntimeError("Sampler has not been fitted")
        count = int(n_rows)
        if count < 1:
            raise ValueError("n_rows must be positive")
        rng = np.random.RandomState(_stable_seed(self.random_state, token))
        patient_indices = rng.randint(0, len(self.patients_), size=count)
        selected = np.empty(count, dtype=np.int64)
        for patient_index in range(len(self.patients_)):
            positions = np.flatnonzero(patient_indices == patient_index)
            if positions.size == 0:
                continue
            patient = self.patients_[patient_index]
            specimens = self.specimens_by_patient_[patient]
            specimen_indices = rng.randint(0, len(specimens), size=positions.size)
            for specimen_index in range(len(specimens)):
                specimen_positions = positions[
                    np.flatnonzero(specimen_indices == specimen_index)
                ]
                if specimen_positions.size == 0:
                    continue
                rows = self.rows_by_specimen_[specimens[specimen_index]]
                selected[specimen_positions] = rows[
                    rng.randint(0, rows.size, size=specimen_positions.size)
                ]
        return self.target_[selected].copy()
