"""Validation for frozen benchmark manifest indexes and record tables."""

from __future__ import annotations

import csv
import hashlib
import math
from collections import defaultdict
from datetime import date
from collections.abc import Mapping
from pathlib import Path

import yaml

from src.benchmark.contract import BenchmarkContractError
from src.benchmark.reference_rows import (
    ReferenceRowError,
    validate_reference_rows,
)


REQUIRED_COLUMNS: dict[str, frozenset[str]] = {
    "data": frozenset(
        {
            "dataset",
            "pair",
            "modality",
            "specimen_id",
            "patient_id",
            "visit_id",
            "collection_id",
            "collection_date",
            "visit_order",
            "source_path",
            "source_sha256",
            "cells_path",
            "cells_sha256",
            "labels_path",
            "labels_sha256",
            "metadata_source_id",
            "metadata_sha256",
            "header_sha256",
            "total_events",
            "eligible_events",
            "event_selection_rule_id",
            "preprocessing_id",
            "upstream_label_selection",
            "runtime_label_selection",
            "claim_scope",
            "pair_status",
            "excluded",
            "exclusion_reason",
            "qc_flag",
        }
    ),
    "markers": frozenset(
        {
            "dataset",
            "direction",
            "modality",
            "canonical_marker",
            "original_channel",
            "role",
            "analysis_role",
            "channel_type",
            "marker_order",
            "panel_order",
            "analysis_order",
            "transformation",
            "transformation_confidence",
            "alias_rule",
            "alias_evidence",
        }
    ),
    "splits": frozenset(
        {
            "dataset",
            "pair",
            "evaluation_fold",
            "role",
            "patient_id",
            "specimen_id",
        }
    ),
    "endpoints": frozenset(
        {
            "dataset",
            "endpoint",
            "patient_id",
            "visit_id",
            "value",
            "provenance",
            "eligibility",
            "reason",
        }
    ),
    "banks": frozenset(
        {
            "reference_bank_id",
            "dataset",
            "pair",
            "direction",
            "modality",
            "evaluation_fold",
            "fit_stage",
            "included_split_roles",
            "seed",
            "bank_role",
            "patient_id",
            "specimen_id",
            "row_index_file",
            "row_index_sha256",
            "row_index_seed",
            "event_count",
            "patient_weight",
            "specimen_weight",
            "event_weight",
            "sampling_algorithm",
            "row_index_dtype",
            "status",
        }
    ),
    "stress": frozenset(
        {
            "condition_id",
            "dataset",
            "factor",
            "level",
            "seed",
            "content_file",
            "content_sha256",
        }
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BenchmarkContractError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _parse_integer(value: str, name: str, *, minimum: int = 0) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise BenchmarkContractError(f"{name} must be an integer") from error
    if str(parsed) != value or parsed < minimum:
        raise BenchmarkContractError(
            f"{name} must be an integer greater than or equal to {minimum}"
        )
    return parsed


def _parse_positive_float(value: str, name: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise BenchmarkContractError(f"{name} must be numeric") from error
    if not parsed > 0:
        raise BenchmarkContractError(f"{name} must be positive")
    return parsed
    return parsed


def _load_records(path: Path, manifest_type: str) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or ())
            missing = sorted(REQUIRED_COLUMNS[manifest_type] - columns)
            if missing:
                raise BenchmarkContractError(
                    f"{manifest_type} records are missing columns: {missing}"
                )
            records = list(reader)
    except UnicodeDecodeError as error:
        raise BenchmarkContractError(
            f"{manifest_type} records are not UTF-8 CSV"
        ) from error
    if not records:
        raise BenchmarkContractError(f"{manifest_type} records must not be empty")
    return records


def _validate_referenced_files(
    records: list[dict[str, str]],
    *,
    records_path: Path,
    path_field: str,
    digest_field: str,
) -> None:
    for row_number, record in enumerate(records, start=2):
        referenced = (records_path.parent / record[path_field]).resolve()
        if not referenced.is_file() or referenced.stat().st_size == 0:
            raise BenchmarkContractError(
                f"Missing or empty referenced file at row {row_number}: {referenced}"
            )
        expected = _require_sha256(
            record[digest_field],
            f"{digest_field} row {row_number}",
        )
        if sha256_file(referenced) != expected:
            raise BenchmarkContractError(
                f"Referenced-file digest mismatch at row {row_number}"
            )


def validate_manifest_index(
    index_path: Path,
    *,
    expected_type: str,
    expected_protocol_id: str,
    expected_index_digest: str,
) -> dict:
    """Validate one immutable index, its CSV table, and referenced content."""

    if expected_type not in REQUIRED_COLUMNS:
        raise BenchmarkContractError(f"Unknown manifest type: {expected_type}")
    if not index_path.is_file() or index_path.stat().st_size == 0:
        raise BenchmarkContractError(
            f"Manifest index is missing or empty: {index_path}"
        )
    expected_digest = _require_sha256(
        expected_index_digest,
        f"{expected_type} index digest",
    )
    if sha256_file(index_path) != expected_digest:
        raise BenchmarkContractError(
            f"{expected_type} manifest index digest mismatch"
        )
    try:
        index = yaml.safe_load(index_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise BenchmarkContractError(
            f"{expected_type} manifest index is not valid YAML"
        ) from error
    if not isinstance(index, Mapping):
        raise BenchmarkContractError(
            f"{expected_type} manifest index must be a mapping"
        )
    header = index.get("manifest")
    records_spec = index.get("records")
    if not isinstance(header, Mapping) or not isinstance(records_spec, Mapping):
        raise BenchmarkContractError(
            f"{expected_type} index requires manifest and records mappings"
        )
    required_header = {
        "type": expected_type,
        "protocol_id": expected_protocol_id,
        "status": "frozen",
    }
    for key, expected in required_header.items():
        if header.get(key) != expected:
            raise BenchmarkContractError(
                f"{expected_type} manifest.{key} must be {expected!r}"
            )
    version = header.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise BenchmarkContractError(
            f"{expected_type} manifest.version must be a positive integer"
        )
    records_relative = records_spec.get("path")
    if not isinstance(records_relative, str) or not records_relative.strip():
        raise BenchmarkContractError(
            f"{expected_type} records.path must be a path string"
        )
    records_path = (index_path.parent / records_relative).resolve()
    if not records_path.is_file() or records_path.stat().st_size == 0:
        raise BenchmarkContractError(
            f"{expected_type} records file is missing or empty: {records_path}"
        )
    records_digest = _require_sha256(
        records_spec.get("sha256"),
        f"{expected_type} records.sha256",
    )
    if sha256_file(records_path) != records_digest:
        raise BenchmarkContractError(
            f"{expected_type} records digest mismatch"
        )
    records = _load_records(records_path, expected_type)
    if expected_type == "banks":
        _validate_referenced_files(
            records,
            records_path=records_path,
            path_field="row_index_file",
            digest_field="row_index_sha256",
        )
    elif expected_type == "stress":
        _validate_referenced_files(
            records,
            records_path=records_path,
            path_field="content_file",
            digest_field="content_sha256",
        )
    return {
        "type": expected_type,
        "index_path": str(index_path),
        "index_digest": expected_digest,
        "records_path": str(records_path),
        "records_digest": records_digest,
        "records": records,
    }


def validate_manifest_bundle(
    manifests: Mapping[str, dict],
    *,
    n_splits: int,
) -> None:
    """Check identifier consistency across already validated manifest tables."""

    records = {name: manifest["records"] for name, manifest in manifests.items()}
    data_keys = set()
    data_patient_by_key = {}
    data_patients = set()
    included_data_keys = set()
    eligible_events_by_key = {}
    exclusion_by_specimen: dict[tuple[str, str], set[str]] = {}
    for row in records["data"]:
        key = (row["dataset"], row["modality"], row["specimen_id"])
        if key in data_keys:
            raise BenchmarkContractError(f"Duplicate data specimen key: {key}")
        data_keys.add(key)
        data_patient_by_key[key] = row["patient_id"]
        source_sha = _require_sha256(
            row["source_sha256"], "data.source_sha256"
        )
        cells_sha = _require_sha256(
            row["cells_sha256"], "data.cells_sha256"
        )
        _require_sha256(row["labels_sha256"], "data.labels_sha256")
        _require_sha256(row["metadata_sha256"], "data.metadata_sha256")
        _require_sha256(row["header_sha256"], "data.header_sha256")
        if source_sha != cells_sha or row["source_path"] != row["cells_path"]:
            raise BenchmarkContractError(
                "data source compatibility fields must match cells artifact"
            )
        for required_text_field in (
            "pair",
            "visit_id",
            "collection_id",
            "metadata_source_id",
            "event_selection_rule_id",
            "preprocessing_id",
            "claim_scope",
            "pair_status",
        ):
            if not row[required_text_field].strip():
                raise BenchmarkContractError(
                    f"data.{required_text_field} must be non-empty"
                )
        _parse_integer(row["visit_order"], "data.visit_order", minimum=1)
        try:
            date.fromisoformat(row["collection_date"])
        except ValueError as error:
            raise BenchmarkContractError(
                "data.collection_date must be ISO YYYY-MM-DD"
            ) from error
        for boolean_field in (
            "upstream_label_selection",
            "runtime_label_selection",
        ):
            if row[boolean_field] not in {"true", "false"}:
                raise BenchmarkContractError(
                    f"data.{boolean_field} must be true or false"
                )
        total_events = _parse_integer(
            row["total_events"],
            "data.total_events",
            minimum=1,
        )
        eligible_events = _parse_integer(
            row["eligible_events"],
            "data.eligible_events",
        )
        if eligible_events > total_events:
            raise BenchmarkContractError(
                "data.eligible_events cannot exceed total_events"
            )
        eligible_events_by_key[key] = eligible_events
        if row["excluded"] not in {"true", "false"}:
            raise BenchmarkContractError("data.excluded must be true or false")
        if row["excluded"] == "true" and not row["exclusion_reason"].strip():
            raise BenchmarkContractError(
                "Excluded data rows require an exclusion_reason"
            )
        if row["excluded"] == "false" and row["exclusion_reason"].strip():
            raise BenchmarkContractError(
                "Included data rows cannot have an exclusion_reason"
            )
        exclusion_by_specimen.setdefault(
            (row["dataset"], row["specimen_id"]),
            set(),
        ).add(row["excluded"])
        if row["excluded"] == "false":
            included_data_keys.add(key)
            data_patients.add((row["dataset"], row["patient_id"]))
    if any(len(states) != 1 for states in exclusion_by_specimen.values()):
        raise BenchmarkContractError(
            "A specimen has inconsistent exclusion state across modalities"
        )

    specimen_patient = {}
    specimens_by_dataset: dict[str, set[str]] = {}
    for dataset, _modality, specimen in data_keys:
        patient = data_patient_by_key[(dataset, _modality, specimen)]
        specimen_key = (dataset, specimen)
        previous = specimen_patient.setdefault(specimen_key, patient)
        if previous != patient:
            raise BenchmarkContractError(
                "A data specimen maps to different patients across modalities"
            )
        if (dataset, _modality, specimen) in included_data_keys:
            specimens_by_dataset.setdefault(dataset, set()).add(specimen)

    split_patients = set()
    split_row_keys = set()
    role_by_patient_fold = {}
    specimens_by_pair_fold: dict[tuple[str, str, int], set[str]] = {}
    folds_by_pair: dict[tuple[str, str], set[int]] = {}
    for row in records["splits"]:
        patient_key = (row["dataset"], row["patient_id"])
        split_patients.add(patient_key)
        fold = _parse_integer(row["evaluation_fold"], "splits.evaluation_fold")
        if fold >= n_splits:
            raise BenchmarkContractError(
                "splits.evaluation_fold exceeds protocol n_splits"
            )
        if row["role"] not in {"train", "validation", "test"}:
            raise BenchmarkContractError(
                "splits.role must be train, validation, or test"
            )
        specimen_key = (row["dataset"], row["specimen_id"])
        if specimen_key not in specimen_patient:
            raise BenchmarkContractError(
                "Split manifest contains a specimen absent from data manifest"
            )
        if row["specimen_id"] not in specimens_by_dataset.get(
            row["dataset"], set()
        ):
            raise BenchmarkContractError(
                "Split manifest contains an excluded data specimen"
            )
        if specimen_patient[specimen_key] != row["patient_id"]:
            raise BenchmarkContractError(
                "Split specimen and patient mapping disagree"
            )
        split_row_key = (
            row["dataset"],
            row["pair"],
            fold,
            row["specimen_id"],
        )
        if split_row_key in split_row_keys:
            raise BenchmarkContractError(f"Duplicate split row: {split_row_key}")
        split_row_keys.add(split_row_key)
        patient_fold_key = (
            row["dataset"],
            row["pair"],
            fold,
            row["patient_id"],
        )
        previous_role = role_by_patient_fold.setdefault(
            patient_fold_key,
            row["role"],
        )
        if previous_role != row["role"]:
            raise BenchmarkContractError(
                "A longitudinal patient has multiple roles in one fold"
            )
        pair_fold = (row["dataset"], row["pair"], fold)
        specimens_by_pair_fold.setdefault(pair_fold, set()).add(row["specimen_id"])
        folds_by_pair.setdefault((row["dataset"], row["pair"]), set()).add(fold)
    if split_patients != data_patients:
        raise BenchmarkContractError(
            "Data and split manifests have different dataset/patient keys"
        )
    expected_folds = set(range(n_splits))
    for (dataset, pair), folds in folds_by_pair.items():
        if folds != expected_folds:
            raise BenchmarkContractError(
                f"Split manifest for {(dataset, pair)!r} does not cover every fold"
            )
        for fold in expected_folds:
            observed = specimens_by_pair_fold[(dataset, pair, fold)]
            if observed != specimens_by_dataset[dataset]:
                raise BenchmarkContractError(
                    "Every split fold must assign every dataset specimen"
                )
        test_folds_by_patient: dict[str, set[int]] = {}
        for (row_dataset, row_pair, fold, patient), role in role_by_patient_fold.items():
            if row_dataset == dataset and row_pair == pair and role == "test":
                test_folds_by_patient.setdefault(patient, set()).add(fold)
        expected_patients = {
            patient for row_dataset, patient in data_patients if row_dataset == dataset
        }
        if set(test_folds_by_patient) != expected_patients or any(
            len(folds_for_patient) != 1
            for folds_for_patient in test_folds_by_patient.values()
        ):
            raise BenchmarkContractError(
                "Each patient must be test in exactly one evaluation fold"
            )

    marker_keys = set()
    marker_orders = set()
    h_order_by_direction_modality: dict[
        tuple[str, str, str], list[tuple[int, str]]
    ] = defaultdict(list)
    data_datasets = {row["dataset"] for row in records["data"]}
    for row in records["markers"]:
        if row["dataset"] not in data_datasets:
            raise BenchmarkContractError(
                "Marker manifest contains a dataset absent from data manifest"
            )
        if row["role"] not in {"H", "Y", "source_only", "technical"}:
            raise BenchmarkContractError(
                "markers.role must be H, Y, source_only, or technical"
            )
        expected_analysis_role = (
            "excluded" if row["role"] == "technical" else row["role"]
        )
        if row["analysis_role"] != expected_analysis_role:
            raise BenchmarkContractError(
                "markers.analysis_role disagrees with compatibility role"
            )
        if row["channel_type"] not in {"biological", "technical"}:
            raise BenchmarkContractError(
                "markers.channel_type must be biological or technical"
            )
        if (
            row["role"] == "technical"
            and row["channel_type"] != "technical"
        ):
            raise BenchmarkContractError(
                "Technical marker role requires technical channel_type"
            )
        key = (
            row["dataset"],
            row["direction"],
            row["modality"],
            row["canonical_marker"],
        )
        order_key = (
            row["dataset"],
            row["direction"],
            row["modality"],
            _parse_integer(row["marker_order"], "markers.marker_order"),
        )
        if key in marker_keys or order_key in marker_orders:
            raise BenchmarkContractError("Duplicate marker name or order")
        marker_keys.add(key)
        marker_orders.add(order_key)
        _parse_integer(row["panel_order"], "markers.panel_order")
        if row["analysis_role"] in {"H", "Y"}:
            analysis_order = _parse_integer(
                row["analysis_order"],
                "markers.analysis_order",
            )
        elif row["analysis_order"]:
            raise BenchmarkContractError(
                "Excluded/source-only marker analysis_order must be blank"
            )
        else:
            analysis_order = -1
        for text_field in (
            "transformation",
            "transformation_confidence",
            "alias_rule",
            "alias_evidence",
        ):
            if not row[text_field].strip():
                raise BenchmarkContractError(
                    f"markers.{text_field} must be non-empty"
                )
        if row["analysis_role"] == "H":
            h_order_by_direction_modality[
                (row["dataset"], row["direction"], row["modality"])
            ].append((analysis_order, row["canonical_marker"]))
    h_order_by_dataset_direction: dict[
        tuple[str, str], set[tuple[str, ...]]
    ] = defaultdict(set)
    for (dataset, direction, _modality), ordered in (
        h_order_by_direction_modality.items()
    ):
        sequence = tuple(marker for _, marker in sorted(ordered))
        if not sequence:
            raise BenchmarkContractError(
                "Every direction/modality marker contract requires H markers"
            )
        h_order_by_dataset_direction[(dataset, direction)].add(sequence)
    if any(
        len(sequences) != 1
        for sequences in h_order_by_dataset_direction.values()
    ):
        raise BenchmarkContractError(
            "H marker order differs across modalities within a direction"
        )

    for row in records["endpoints"]:
        if (row["dataset"], row["patient_id"]) not in data_patients:
            raise BenchmarkContractError(
                "Endpoint manifest contains a patient absent from data manifest"
            )
        if row["eligibility"] not in {
            "primary",
            "secondary",
            "exploratory",
            "not_evaluable",
        }:
            raise BenchmarkContractError("Unknown endpoint eligibility")
    shared_bank_artifacts = {}
    bank_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    validated_row_contracts = set()
    bank_records_path = Path(manifests["banks"]["records_path"])
    for row in records["banks"]:
        data_key = (row["dataset"], row["modality"], row["specimen_id"])
        if data_key not in data_keys:
            raise BenchmarkContractError(
                f"Reference bank contains an unknown specimen: {data_key}"
            )
        if data_patient_by_key[data_key] != row["patient_id"]:
            raise BenchmarkContractError(
                "Reference-bank specimen and patient mapping disagree"
            )
        if data_key not in included_data_keys:
            raise BenchmarkContractError(
                "Reference bank contains an excluded data specimen"
            )
        if (row["dataset"], row["patient_id"]) not in data_patients:
            raise BenchmarkContractError(
                "Reference bank contains a patient absent from data manifest"
            )
        evaluation_fold = _parse_integer(
            row["evaluation_fold"],
            "banks.evaluation_fold",
        )
        if evaluation_fold >= n_splits:
            raise BenchmarkContractError(
                "banks.evaluation_fold exceeds protocol n_splits"
            )
        patient_role = role_by_patient_fold.get(
            (
                row["dataset"],
                row["pair"],
                evaluation_fold,
                row["patient_id"],
            )
        )
        fit_stage_roles = {
            "inner_fit": ("train",),
            "outer_refit": ("train", "validation"),
        }
        fit_stage = row["fit_stage"]
        if fit_stage not in fit_stage_roles:
            raise BenchmarkContractError("Unknown reference-bank fit_stage")
        allowed_roles = fit_stage_roles[fit_stage]
        if row["included_split_roles"] != ";".join(allowed_roles):
            raise BenchmarkContractError(
                "Reference-bank included_split_roles disagrees with fit_stage"
            )
        if patient_role not in allowed_roles:
            raise BenchmarkContractError(
                "Reference bank contains a patient outside its fitting roles"
            )
        _parse_integer(row["seed"], "banks.seed")
        row_index_seed = _parse_integer(
            row["row_index_seed"], "banks.row_index_seed"
        )
        event_count = _parse_integer(
            row["event_count"], "banks.event_count", minimum=1
        )
        if event_count > eligible_events_by_key[data_key]:
            raise BenchmarkContractError(
                "Reference-bank event_count exceeds eligible specimen events"
            )
        row_index_path = (
            bank_records_path.parent / row["row_index_file"]
        ).resolve()
        row_contract = (
            row_index_path,
            eligible_events_by_key[data_key],
            event_count,
            row_index_seed,
            row["row_index_sha256"],
        )
        if row_contract not in validated_row_contracts:
            try:
                validate_reference_rows(
                    row_index_path,
                    eligible_events=eligible_events_by_key[data_key],
                    selected_events=event_count,
                    row_seed=row_index_seed,
                )
            except ReferenceRowError as error:
                raise BenchmarkContractError(str(error)) from error
            validated_row_contracts.add(row_contract)
        shared_key = (
            row["dataset"],
            row["modality"],
            row["specimen_id"],
            row["seed"],
        )
        shared_artifact = (
            row["row_index_file"],
            row["row_index_sha256"],
            event_count,
        )
        previous_artifact = shared_bank_artifacts.setdefault(
            shared_key,
            shared_artifact,
        )
        if previous_artifact != shared_artifact:
            raise BenchmarkContractError(
                "Bank roles do not reuse the same specimen/seed row artifact"
            )
        patient_weight = _parse_positive_float(
            row["patient_weight"], "banks.patient_weight"
        )
        specimen_weight = _parse_positive_float(
            row["specimen_weight"], "banks.specimen_weight"
        )
        event_weight = _parse_positive_float(
            row["event_weight"], "banks.event_weight"
        )
        if (
            row["sampling_algorithm"]
            != "numpy_pcg64_uniform_without_replacement_sorted_v1"
        ):
            raise BenchmarkContractError(
                "Unknown reference-bank sampling algorithm"
            )
        if row["row_index_dtype"] != "uint32":
            raise BenchmarkContractError(
                "Reference-bank row indices must use uint32"
            )
        if row["status"] != "materialized_validated":
            raise BenchmarkContractError(
                "Frozen reference-bank rows must be materialized_validated"
            )
        if row["bank_role"] not in {
            "target_predictor_bank",
            "calibration_bank",
            "null_prior_bank",
        }:
            raise BenchmarkContractError("Unknown reference-bank role")
        bank_groups[row["reference_bank_id"]].append(
            {
                "row": row,
                "event_count": event_count,
                "patient_weight": patient_weight,
                "specimen_weight": specimen_weight,
                "event_weight": event_weight,
                "allowed_roles": allowed_roles,
            }
        )

    for bank_id, group in bank_groups.items():
        first = group[0]["row"]
        dimension_fields = (
            "dataset",
            "pair",
            "direction",
            "modality",
            "evaluation_fold",
            "fit_stage",
            "seed",
            "bank_role",
        )
        if any(
            any(item["row"][field] != first[field] for field in dimension_fields)
            for item in group[1:]
        ):
            raise BenchmarkContractError(
                f"Reference bank ID spans inconsistent dimensions: {bank_id}"
            )
        observed = {
            (item["row"]["patient_id"], item["row"]["specimen_id"])
            for item in group
        }
        if len(observed) != len(group):
            raise BenchmarkContractError(
                f"Reference bank has duplicate memberships: {bank_id}"
            )
        fold = int(first["evaluation_fold"])
        allowed_roles = set(group[0]["allowed_roles"])
        expected = {
            (data_patient_by_key[key], key[2])
            for key in included_data_keys
            if key[0] == first["dataset"]
            and key[1] == first["modality"]
            and role_by_patient_fold.get(
                (
                    first["dataset"],
                    first["pair"],
                    fold,
                    data_patient_by_key[key],
                )
            )
            in allowed_roles
        }
        if observed != expected:
            raise BenchmarkContractError(
                f"Reference bank membership is incomplete: {bank_id}"
            )
        specimens_by_patient: dict[str, int] = defaultdict(int)
        for patient, _specimen in observed:
            specimens_by_patient[patient] += 1
        expected_patient_weight = 1.0 / len(specimens_by_patient)
        for item in group:
            row = item["row"]
            expected_specimen_weight = (
                expected_patient_weight
                / specimens_by_patient[row["patient_id"]]
            )
            expected_event_weight = (
                expected_specimen_weight / item["event_count"]
            )
            for observed_weight, expected_weight, label in (
                (
                    item["patient_weight"],
                    expected_patient_weight,
                    "patient_weight",
                ),
                (
                    item["specimen_weight"],
                    expected_specimen_weight,
                    "specimen_weight",
                ),
                (
                    item["event_weight"],
                    expected_event_weight,
                    "event_weight",
                ),
            ):
                if not math.isclose(
                    observed_weight,
                    expected_weight,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                ):
                    raise BenchmarkContractError(
                        f"Reference-bank {label} is inconsistent: {bank_id}"
                    )
