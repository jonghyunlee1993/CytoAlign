"""Patient-grouped folds for specimen-level AML data."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.model_selection import KFold


PATIENT_PATTERN = re.compile(r"^(R\d+)(?:_|$)", flags=re.IGNORECASE)


def patient_id_from_specimen(specimen_id: str) -> str:
    """Map ``R3161_8847`` to the longitudinal patient group ``R3161``."""
    stem = Path(str(specimen_id)).stem
    match = PATIENT_PATTERN.match(stem)
    if match:
        return match.group(1).upper()
    return stem.split("_", 1)[0]


def discover_exact_specimen_pairs(
    data_root: str | Path, modality_a: str, modality_b: str
) -> tuple[str, ...]:
    """Return specimen stems with cell and label CSVs in both modalities."""
    root = Path(data_root)

    def complete(modality: str) -> set[str]:
        cells = {path.stem for path in (root / modality / "cells").glob("*.csv")}
        labels = {path.stem for path in (root / modality / "labels").glob("*.csv")}
        return cells & labels

    return tuple(sorted(complete(modality_a) & complete(modality_b)))


def _inner_partition(
    patients: Sequence[str], validation_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    patients = np.asarray(sorted(patients), dtype=object)
    if patients.size < 2:
        raise ValueError("At least two outer-training patients are required")
    rng = np.random.RandomState(int(seed))
    shuffled = patients[rng.permutation(patients.size)]
    n_validation = max(1, int(round(float(validation_fraction) * patients.size)))
    n_validation = min(n_validation, patients.size - 1)
    validation = sorted(map(str, shuffled[:n_validation]))
    training = sorted(map(str, shuffled[n_validation:]))
    return training, validation


def build_patient_grouped_manifest(
    specimen_ids: Sequence[str],
    *,
    n_splits: int = 5,
    seed: int = 42,
    validation_fraction: float = 0.15,
    pair_name: str = "sf_cytof",
) -> dict:
    """Create outer folds grouped by the ``Rxxxx`` patient prefix."""
    specimens = tuple(sorted({Path(str(value)).stem for value in specimen_ids}))
    if not specimens:
        raise ValueError("No specimens were supplied")
    patient_to_specimens: dict[str, list[str]] = {}
    for specimen in specimens:
        patient_to_specimens.setdefault(patient_id_from_specimen(specimen), []).append(
            specimen
        )
    patients = np.asarray(sorted(patient_to_specimens), dtype=object)
    if patients.size < int(n_splits):
        raise ValueError("The number of patients is smaller than n_splits")

    splitter = KFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
    folds = []
    for fold_index, (outer_train_index, test_index) in enumerate(
        splitter.split(patients)
    ):
        outer_train = sorted(map(str, patients[outer_train_index]))
        test = sorted(map(str, patients[test_index]))
        training, validation = _inner_partition(
            outer_train, validation_fraction, int(seed) + 1009 * fold_index
        )

        def specimens_for(groups: Sequence[str]) -> list[str]:
            return sorted(
                specimen
                for patient in groups
                for specimen in patient_to_specimens[patient]
            )

        folds.append(
            {
                "fold_index": fold_index,
                "train_patients": training,
                "validation_patients": validation,
                "test_patients": test,
                "train_specimens": specimens_for(training),
                "validation_specimens": specimens_for(validation),
                "test_specimens": specimens_for(test),
            }
        )

    manifest = {
        "manifest_version": 1,
        "created_date": date.today().isoformat(),
        "pair": str(pair_name),
        "split_unit": "Rxxxx patient prefix",
        "fold_seed": int(seed),
        "validation_fraction": float(validation_fraction),
        "patients": sorted(patient_to_specimens),
        "specimens": list(specimens),
        "patient_to_specimens": {
            patient: sorted(values) for patient, values in patient_to_specimens.items()
        },
        "folds": folds,
    }
    validate_patient_grouped_manifest(manifest)
    return manifest


def validate_patient_grouped_manifest(manifest: dict) -> None:
    """Reject patient leakage and incomplete outer-test partitions."""
    required = {"patients", "specimens", "patient_to_specimens", "folds"}
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"Manifest is missing fields: {sorted(missing)}")
    expected_patients = set(map(str, manifest["patients"]))
    expected_specimens = set(map(str, manifest["specimens"]))
    observed_test_patients: list[str] = []
    observed_test_specimens: list[str] = []
    mapping = {
        str(patient): set(map(str, specimens))
        for patient, specimens in manifest["patient_to_specimens"].items()
    }
    for fold in manifest["folds"]:
        train = set(map(str, fold["train_patients"]))
        validation = set(map(str, fold["validation_patients"]))
        test = set(map(str, fold["test_patients"]))
        if train & validation or train & test or validation & test:
            raise ValueError(f"Fold {fold['fold_index']} contains patient leakage")
        if train | validation | test != expected_patients:
            raise ValueError(f"Fold {fold['fold_index']} does not cover all patients")
        for role, patients in (
            ("train", train),
            ("validation", validation),
            ("test", test),
        ):
            expected = set().union(*(mapping[patient] for patient in patients))
            actual = set(map(str, fold[f"{role}_specimens"]))
            if actual != expected:
                raise ValueError(
                    f"Fold {fold['fold_index']} has inconsistent {role} specimens"
                )
        observed_test_patients.extend(test)
        observed_test_specimens.extend(map(str, fold["test_specimens"]))
    if len(observed_test_patients) != len(set(observed_test_patients)):
        raise ValueError("A patient occurs in more than one outer test fold")
    if set(observed_test_patients) != expected_patients:
        raise ValueError("Outer test folds do not partition patients")
    if len(observed_test_specimens) != len(set(observed_test_specimens)):
        raise ValueError("A specimen occurs in more than one outer test fold")
    if set(observed_test_specimens) != expected_specimens:
        raise ValueError("Outer test folds do not partition specimens")


def save_manifest(manifest: dict, path: str | Path) -> None:
    validate_patient_grouped_manifest(manifest)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


def load_manifest(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_patient_grouped_manifest(manifest)
    return manifest
