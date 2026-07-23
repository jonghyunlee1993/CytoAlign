import copy

import pytest

from src.data.splits import (
    build_patient_grouped_manifest,
    discover_exact_specimen_pairs,
    load_manifest,
    patient_id_from_specimen,
    save_manifest,
    validate_patient_grouped_manifest,
)


def test_patient_prefix_parser_groups_longitudinal_specimens():
    assert patient_id_from_specimen("R3161_7139.csv") == "R3161"
    assert patient_id_from_specimen("R3161_8847") == "R3161"
    assert patient_id_from_specimen("sample_visit2") == "sample"


def test_grouped_folds_never_split_longitudinal_patient():
    specimens = [f"R{index:04d}_A" for index in range(1, 12)]
    specimens += ["R0002_B", "R0002_C", "R0007_B"]
    manifest = build_patient_grouped_manifest(
        specimens, n_splits=5, seed=42, validation_fraction=0.2
    )

    test_fold_by_patient = {}
    for fold in manifest["folds"]:
        for patient in fold["test_patients"]:
            test_fold_by_patient[patient] = fold["fold_index"]
        for role in ("train", "validation", "test"):
            patients = set(fold[f"{role}_patients"])
            for specimen in fold[f"{role}_specimens"]:
                assert patient_id_from_specimen(specimen) in patients
    assert len(test_fold_by_patient) == 11
    assert test_fold_by_patient["R0002"] in range(5)


def test_manifest_round_trip_and_leakage_validation(tmp_path):
    manifest = build_patient_grouped_manifest(
        [f"R{index:04d}_A" for index in range(10)], n_splits=5
    )
    path = tmp_path / "manifest.json"
    save_manifest(manifest, path)
    assert load_manifest(path) == manifest

    invalid = copy.deepcopy(manifest)
    leaked = invalid["folds"][0]["test_patients"][0]
    invalid["folds"][0]["train_patients"].append(leaked)
    with pytest.raises(ValueError, match="patient leakage"):
        validate_patient_grouped_manifest(invalid)


def test_exact_pair_discovery_requires_cells_and_labels_in_both_panels(tmp_path):
    for modality in ("spectral_flow", "cytof"):
        (tmp_path / modality / "cells").mkdir(parents=True)
        (tmp_path / modality / "labels").mkdir(parents=True)
        for stem in ("R0001_A", "R0002_A"):
            (tmp_path / modality / "cells" / f"{stem}.csv").write_text("CD3\n1\n")
            (tmp_path / modality / "labels" / f"{stem}.csv").write_text(
                "cell_type\nT cell\n"
            )
    (tmp_path / "cytof" / "labels" / "R0002_A.csv").unlink()
    assert discover_exact_specimen_pairs(tmp_path, "spectral_flow", "cytof") == (
        "R0001_A",
    )

