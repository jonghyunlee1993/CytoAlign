"""Auditable, training-only artifacts used by benchmark metrics and nulls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _require_digest(value: str, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return digest


def _require_names(values: Any, name: str) -> tuple[str, ...]:
    names = tuple(values)
    if not names or any(not isinstance(value, str) or not value.strip() for value in names):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError(f"{name} must be unique")
    return names


@dataclass(frozen=True)
class LockedFeatureScales:
    """Marker-aligned scales fitted using outer-training target patients only."""

    values: np.ndarray
    marker_names: tuple[str, ...]
    fit_role: str
    fold: int
    patient_digest: str
    manifest_digest: str
    scale_id: str

    def __post_init__(self) -> None:
        scales = np.asarray(self.values, dtype=np.float64)
        markers = _require_names(self.marker_names, "marker_names")
        if scales.ndim != 1 or scales.size != len(markers):
            raise ValueError("Scale values and marker_names do not align")
        if scales.size == 0 or np.any(scales <= 0) or not np.isfinite(scales).all():
            raise ValueError("Scale values must be finite and positive")
        if self.fit_role != "outer_training_target":
            raise ValueError("Feature scales must be fitted on outer-training target data")
        if isinstance(self.fold, bool) or not isinstance(self.fold, int) or self.fold < 0:
            raise ValueError("fold must be a non-negative integer")
        if not isinstance(self.scale_id, str) or not self.scale_id.strip():
            raise ValueError("scale_id must be a non-empty string")
        _require_digest(self.patient_digest, "patient_digest")
        _require_digest(self.manifest_digest, "manifest_digest")
        immutable = scales.copy()
        immutable.setflags(write=False)
        object.__setattr__(self, "values", immutable)
        object.__setattr__(self, "marker_names", markers)


@dataclass(frozen=True)
class TrainingReference:
    """A marker-aligned outer-training reference bank with row provenance."""

    values: np.ndarray
    patient_ids: tuple[str, ...]
    specimen_ids: tuple[str, ...]
    marker_names: tuple[str, ...]
    split_role: str
    bank_role: str
    fold: int
    reference_bank_id: str
    manifest_digest: str

    def __post_init__(self) -> None:
        matrix = np.asarray(self.values, dtype=np.float64)
        patients = _require_names_allow_repeats(self.patient_ids, "patient_ids")
        specimens = _require_names_allow_repeats(self.specimen_ids, "specimen_ids")
        markers = _require_names(self.marker_names, "marker_names")
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("values must be a non-empty matrix")
        if not np.isfinite(matrix).all():
            raise ValueError("values contain non-finite entries")
        if len(patients) != matrix.shape[0] or len(specimens) != matrix.shape[0]:
            raise ValueError("Patient/specimen IDs must align with reference rows")
        if len(markers) != matrix.shape[1]:
            raise ValueError("marker_names do not align with reference columns")
        patient_by_specimen: dict[str, str] = {}
        for patient, specimen in zip(patients, specimens):
            previous = patient_by_specimen.setdefault(specimen, patient)
            if previous != patient:
                raise ValueError(f"Specimen {specimen!r} maps to multiple patients")
        if self.split_role != "outer_training":
            raise ValueError("Training references must have split_role='outer_training'")
        if self.bank_role not in {
            "target_predictor_bank",
            "calibration_bank",
            "null_prior_bank",
        }:
            raise ValueError("Unknown reference bank role")
        if isinstance(self.fold, bool) or not isinstance(self.fold, int) or self.fold < 0:
            raise ValueError("fold must be a non-negative integer")
        if not isinstance(self.reference_bank_id, str) or not self.reference_bank_id.strip():
            raise ValueError("reference_bank_id must be a non-empty string")
        _require_digest(self.manifest_digest, "manifest_digest")
        immutable = matrix.astype(np.float32, copy=True)
        immutable.setflags(write=False)
        object.__setattr__(self, "values", immutable)
        object.__setattr__(self, "patient_ids", patients)
        object.__setattr__(self, "specimen_ids", specimens)
        object.__setattr__(self, "marker_names", markers)


def _require_names_allow_repeats(values: Any, name: str) -> tuple[str, ...]:
    names = tuple(values)
    if not names or any(not isinstance(value, str) or not value.strip() for value in names):
        raise ValueError(f"{name} must contain non-empty strings")
    return names
