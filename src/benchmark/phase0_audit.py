"""Phase 0 inventory, event-selection, split, and endpoint audits."""

from __future__ import annotations

import csv
import hashlib
import json
from copy import deepcopy
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from src.benchmark.contract import protocol_digest, validate_primary_benchmark_config
from src.data.aml import coarsen_cell_types
from src.data.markers import (
    DEFAULT_TECHNICAL_MARKERS,
    build_pair_marker_manifest,
    canonical_marker_name,
)
from src.data.splits import build_patient_grouped_manifest, patient_id_from_specimen


DATASET_ID = "aml_sf_cytof"
PAIR_ID = "sf_cytof"
MODALITIES = ("spectral_flow", "cytof")
ENDPOINTS = ("FLT3", "TP53", "NPM", "IDH", "RAS")
COHORT_POLICIES = ("primary", "low_event_exclusion_sensitivity")
H19 = (
    "CD3",
    "CD4",
    "CD8",
    "CD14",
    "CD19",
    "CD20",
    "CD25",
    "CD27",
    "CD33",
    "CD34",
    "CD38",
    "CD45",
    "CD56",
    "CD117",
    "CD123",
    "CCR7",
    "PD-1",
    "CD45RA",
    "HLA-DR",
)


class Phase0AuditError(ValueError):
    """Raised when source inventory cannot support a deterministic draft."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _header(path: Path) -> tuple[str, ...]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        row = next(csv.reader(handle), None)
    if not row:
        raise Phase0AuditError(f"CSV has no header: {path}")
    return tuple(map(str, row))


def _header_sha256(header: Sequence[str]) -> str:
    payload = json.dumps(
        list(map(str, header)),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _count_csv_data_rows(
    path: Path,
    *,
    compute_sha256: bool,
) -> tuple[int, str]:
    """Count expression rows without consulting labels.

    AML expression CSVs contain one physical line per event. The binary pass
    also supports source hashing so the expensive files need not be opened
    twice.
    """

    digest = hashlib.sha256() if compute_sha256 else None
    newline_count = 0
    last_byte = b""
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            newline_count += block.count(b"\n")
            last_byte = block[-1:]
            if digest is not None:
                digest.update(block)
    physical_lines = newline_count + (1 if last_byte and last_byte != b"\n" else 0)
    data_rows = physical_lines - 1
    if data_rows < 1:
        raise Phase0AuditError(f"Expression CSV has no data rows: {path}")
    return data_rows, digest.hexdigest() if digest is not None else "pending"


def _label_column(columns: Sequence[str]) -> str:
    for candidate in ("cell_type", "label", "event_type"):
        if candidate in columns:
            return candidate
    if len(columns) == 1:
        return str(columns[0])
    raise Phase0AuditError(f"Cannot identify label column from {tuple(columns)}")


def _audit_label_file(path: Path, *, chunk_size: int = 250_000) -> dict:
    total = 0
    unmapped = 0
    fine_labels: set[str] = set()
    label_column = None
    for chunk in pd.read_csv(path, chunksize=int(chunk_size)):
        if label_column is None:
            label_column = _label_column(tuple(map(str, chunk.columns)))
        values = chunk[label_column].astype(str).to_numpy()
        coarse = coarsen_cell_types(values)
        total += len(values)
        unmapped += int((coarse == None).sum())  # noqa: E711
        fine_labels.update(map(str, values))
    if total < 1:
        raise Phase0AuditError(f"Label file has no data rows: {path}")
    return {
        "rows": total,
        "unmapped_rows": unmapped,
        "fine_labels": tuple(sorted(fine_labels)),
    }


def _complete_stems(root: Path, modality: str) -> tuple[set[str], set[str]]:
    cells = {path.stem for path in (root / modality / "cells").glob("*.csv")}
    labels = {path.stem for path in (root / modality / "labels").glob("*.csv")}
    return cells, labels


def _transformation(modality: str, marker: str) -> str:
    if modality == "cytof":
        return "upstream_asinh_x_over_5"
    if marker in {"SSC-A", "FSC-A", "SSC-B-A"}:
        return "upstream_linear_divide_600000"
    if marker == "AF-A":
        return "upstream_processed_autofluorescence"
    return "upstream_asinh_approximately_x_over_6000"


def _load_authoritative_metadata(
    path: Path,
    specimens: Sequence[str],
    contract: Mapping[str, object],
) -> tuple[dict[str, dict[str, str]], dict]:
    if not path.is_file():
        raise Phase0AuditError(f"Authoritative AML metadata is missing: {path}")
    digest = _sha256(path)
    if digest != contract["expected_sha256"]:
        raise Phase0AuditError(
            "Authoritative AML metadata digest differs from the protocol"
        )
    patient_column = str(contract["patient_column"])
    visit_column = str(contract["visit_column"])
    date_column = str(contract["collection_date_column"])
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    missing_columns = sorted(
        {patient_column, visit_column, date_column} - set(frame.columns)
    )
    if missing_columns:
        raise Phase0AuditError(
            f"Authoritative AML metadata is missing columns: {missing_columns}"
        )
    mapping = {}
    for _, row in frame.iterrows():
        patient = str(row[patient_column]).strip().upper()
        raw_visit = str(row[visit_column]).strip()
        try:
            visit = str(int(raw_visit))
        except ValueError as error:
            raise Phase0AuditError(
                f"Invalid collection ID in AML metadata: {raw_visit!r}"
            ) from error
        parsed_date = pd.to_datetime(
            str(row[date_column]).strip(),
            format=str(contract["date_format"]),
            errors="coerce",
        )
        if not patient or pd.isna(parsed_date):
            raise Phase0AuditError("AML metadata has a missing patient/date value")
        specimen = f"{patient}_{visit}"
        if specimen in mapping:
            raise Phase0AuditError(
                f"Duplicate authoritative AML specimen key: {specimen}"
            )
        mapping[specimen] = {
            "patient_id": patient,
            "visit_id": visit,
            "collection_date": parsed_date.date().isoformat(),
        }
    discovered = set(map(str, specimens))
    missing = sorted(discovered - set(mapping))
    if missing:
        raise Phase0AuditError(
            "Discovered specimens are absent from authoritative metadata: "
            f"{missing}"
        )
    discovered_mapping = {
        specimen: mapping[specimen] for specimen in sorted(discovered)
    }
    specimens_by_patient: dict[str, list[str]] = defaultdict(list)
    for specimen, values in discovered_mapping.items():
        if values["patient_id"] != patient_id_from_specimen(specimen):
            raise Phase0AuditError(
                f"Metadata patient mapping disagrees for {specimen}"
            )
        specimens_by_patient[values["patient_id"]].append(specimen)
    for patient_specimens in specimens_by_patient.values():
        ordered = sorted(
            patient_specimens,
            key=lambda specimen: (
                discovered_mapping[specimen]["collection_date"],
                int(discovered_mapping[specimen]["visit_id"]),
            ),
        )
        for visit_order, specimen in enumerate(ordered, start=1):
            discovered_mapping[specimen]["visit_order"] = str(visit_order)
    return discovered_mapping, {
        "artifact_id": str(contract["artifact_id"]),
        "path": path.as_posix(),
        "sha256": digest,
        "release_reference": str(contract["release_reference"]),
        "rows": len(frame),
        "discovered_specimens_mapped": len(discovered_mapping),
        "patient_column": patient_column,
        "visit_column": visit_column,
        "collection_date_column": date_column,
        "date_format": str(contract["date_format"]),
    }


def _canonical_lookup(header: Sequence[str]) -> dict[str, str]:
    lookup = {}
    for original in header:
        canonical = canonical_marker_name(original)
        if canonical in lookup:
            raise Phase0AuditError(
                f"Panel has duplicate canonical marker {canonical!r}"
            )
        lookup[canonical] = str(original)
    return lookup


def _source_bundle_digest(
    rows: Sequence[Mapping[str, object]],
    *,
    paired_only: bool,
) -> str:
    lines = []
    for row in rows:
        if paired_only and row["pair_status"] != "exact_pair":
            continue
        for path_field, digest_field in (
            ("cells_path", "cells_sha256"),
            ("labels_path", "labels_sha256"),
        ):
            digest = str(row[digest_field])
            if digest == "pending":
                return "pending"
            recorded_path = Path(str(row[path_field])).as_posix()
            lines.append(f"{digest}  {recorded_path}")
    payload = ("\n".join(sorted(lines)) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def audit_data_inventory(
    data_root: str | Path,
    *,
    hash_source_files: bool = False,
    authoritative_metadata_path: str | Path | None = None,
    authoritative_metadata_contract: Mapping[str, object] | None = None,
    cohort_policy: str = "primary",
    specimen_event_policy: Mapping[str, object] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict]:
    """Inventory all AML files and audit label-free processed-row eligibility."""

    root = Path(data_root)
    if cohort_policy not in COHORT_POLICIES:
        raise Phase0AuditError(
            f"cohort_policy must be one of {COHORT_POLICIES}, got {cohort_policy!r}"
        )
    if specimen_event_policy is None:
        raise Phase0AuditError("A protocol specimen-event policy is required")
    primary_minimum = int(
        specimen_event_policy["primary_min_events_per_modality"]
    )
    low_event_threshold = int(specimen_event_policy["low_event_flag_below"])
    sensitivity_threshold = int(
        specimen_event_policy["sensitivity_exclude_below"]
    )
    inventory: dict[str, tuple[set[str], set[str]]] = {
        modality: _complete_stems(root, modality) for modality in MODALITIES
    }
    incomplete = {
        modality: {
            "cells_without_labels": sorted(
                inventory[modality][0] - inventory[modality][1]
            ),
            "labels_without_cells": sorted(
                inventory[modality][1] - inventory[modality][0]
            ),
        }
        for modality in MODALITIES
    }
    if any(
        values["cells_without_labels"] or values["labels_without_cells"]
        for values in incomplete.values()
    ):
        raise Phase0AuditError(
            f"Cell/label file discovery is incomplete: {incomplete}"
        )
    complete_by_modality = {
        modality: inventory[modality][0] & inventory[modality][1]
        for modality in MODALITIES
    }
    paired = set.intersection(
        *(complete_by_modality[modality] for modality in MODALITIES)
    )
    if not paired:
        raise Phase0AuditError("No spectral-flow/CyTOF cell pairs were found")
    discovered = set.union(*complete_by_modality.values())
    if authoritative_metadata_path is not None:
        if authoritative_metadata_contract is None:
            raise Phase0AuditError(
                "An authoritative metadata path requires a metadata contract"
            )
        specimen_metadata, metadata_summary = _load_authoritative_metadata(
            Path(authoritative_metadata_path),
            sorted(discovered),
            authoritative_metadata_contract,
        )
        metadata_summary["paired_specimens_mapped"] = len(paired)
        metadata_status = "authoritative"
    else:
        specimen_metadata = {
            specimen: {
                "patient_id": patient_id_from_specimen(specimen),
                "visit_id": specimen,
                "visit_order": "unresolved",
                "collection_date": "",
            }
            for specimen in discovered
        }
        metadata_summary = {
            "artifact_id": "not_loaded",
            "status": "patient_prefix_and_specimen_proxy_only",
        }
        metadata_status = "fallback_unresolved"

    rows: list[dict[str, object]] = []
    modality_summaries: dict[str, dict[str, object]] = {}
    events_by_modality_specimen: dict[tuple[str, str], int] = {}
    for modality in MODALITIES:
        expected_header = None
        total_events = 0
        paired_events = 0
        total_unmapped = 0
        fine_labels: set[str] = set()
        for specimen in sorted(complete_by_modality[modality]):
            cell_path = root / modality / "cells" / f"{specimen}.csv"
            label_path = root / modality / "labels" / f"{specimen}.csv"
            header = _header(cell_path)
            if expected_header is None:
                expected_header = header
            elif header != expected_header:
                raise Phase0AuditError(
                    f"{modality}/{specimen} header differs from the panel contract"
                )
            expression_rows, cells_digest = _count_csv_data_rows(
                cell_path,
                compute_sha256=hash_source_files,
            )
            label_audit = _audit_label_file(label_path)
            if expression_rows != int(label_audit["rows"]):
                raise Phase0AuditError(
                    f"{modality}/{specimen} expression/label row mismatch: "
                    f"{expression_rows} != {label_audit['rows']}"
                )
            labels_digest = _sha256(label_path)
            total_events += expression_rows
            if specimen in paired:
                paired_events += expression_rows
            total_unmapped += int(label_audit["unmapped_rows"])
            fine_labels.update(label_audit["fine_labels"])
            events_by_modality_specimen[(modality, specimen)] = expression_rows
            metadata = specimen_metadata[specimen]
            if specimen in paired:
                pair_status = "exact_pair"
            else:
                pair_status = f"{modality}_only"
            rows.append(
                {
                    "dataset": DATASET_ID,
                    "pair": PAIR_ID,
                    "modality": modality,
                    "specimen_id": specimen,
                    "patient_id": metadata["patient_id"],
                    "visit_id": metadata["visit_id"],
                    "collection_id": metadata["visit_id"],
                    "visit_order": metadata["visit_order"],
                    "collection_date": metadata["collection_date"],
                    "visit_id_provenance": metadata_status,
                    "metadata_source_id": metadata_summary["artifact_id"],
                    "metadata_sha256": metadata_summary.get("sha256", "pending"),
                    "cells_path": cell_path.as_posix(),
                    "cells_sha256": cells_digest,
                    "cells_bytes": cell_path.stat().st_size,
                    "labels_path": label_path.as_posix(),
                    "labels_sha256": labels_digest,
                    "labels_bytes": label_path.stat().st_size,
                    "header_sha256": _header_sha256(header),
                    "source_path": cell_path.as_posix(),
                    "source_sha256": cells_digest,
                    "source_bytes": cell_path.stat().st_size,
                    "total_events": expression_rows,
                    "eligible_events": expression_rows,
                    "event_count_provenance": "expression_csv_physical_rows",
                    "event_selection_rule_id": (
                        "aml_processed_pregated_all_rows_v1"
                    ),
                    "local_unmapped_rows": int(label_audit["unmapped_rows"]),
                    "preprocessing_id": (
                        "processed_transformed_upstream_pregated"
                    ),
                    "upstream_label_selection": "true",
                    "runtime_label_selection": "false",
                    "claim_scope": (
                        "h_only_translation_within_modality_specific_"
                        "upstream_pregated_aml_compartments"
                    ),
                    "pair_status": pair_status,
                    "cohort_policy": cohort_policy,
                    "excluded": "pending",
                    "exclusion_reason": "pending",
                    "qc_flag": "",
                    "review_flag": "",
                }
            )
        modality_summaries[modality] = {
            "specimens": len(complete_by_modality[modality]),
            "paired_specimens": len(paired),
            "patients": len(
                {
                    patient_id_from_specimen(specimen)
                    for specimen in complete_by_modality[modality]
                }
            ),
            "events": total_events,
            "paired_events": paired_events,
            "local_unmapped_rows": total_unmapped,
            "headers_consistent": True,
            "cell_label_row_mismatches": 0,
            "markers": list(expected_header or ()),
            "fine_labels": sorted(fine_labels),
        }

    low_event_specimens = {
        specimen
        for specimen in paired
        if min(
            events_by_modality_specimen[(modality, specimen)]
            for modality in MODALITIES
        )
        < low_event_threshold
    }
    if cohort_policy == "primary":
        exclusion_threshold = primary_minimum
    else:
        exclusion_threshold = sensitivity_threshold
    excluded_paired = {
        specimen
        for specimen in paired
        if min(
            events_by_modality_specimen[(modality, specimen)]
            for modality in MODALITIES
        )
        < exclusion_threshold
    }
    eligible_paired = tuple(sorted(paired - excluded_paired))
    for row in rows:
        specimen = str(row["specimen_id"])
        if specimen not in paired:
            row["excluded"] = "true"
            row["exclusion_reason"] = "no_exact_cross_modality_pair"
            row["review_flag"] = "discovery_only_unpaired_specimen"
        elif specimen in excluded_paired:
            row["excluded"] = "true"
            row["exclusion_reason"] = (
                f"paired_modality_below_{exclusion_threshold}_events"
            )
            row["review_flag"] = "low_event_exclusion_sensitivity"
        else:
            row["excluded"] = "false"
            row["exclusion_reason"] = ""
            if specimen in low_event_specimens:
                row["qc_flag"] = (
                    f"paired_modality_below_{low_event_threshold}_events"
                )
                row["review_flag"] = "included_low_event_primary"

    for modality in MODALITIES:
        modality_summaries[modality]["eligible_paired_events"] = sum(
            events_by_modality_specimen[(modality, specimen)]
            for specimen in eligible_paired
        )
        modality_summaries[modality]["eligible_paired_specimens"] = len(
            eligible_paired
        )

    event_selection = [
        {
            "dataset": DATASET_ID,
            "modality": modality,
            "input_artifact": "processed_cells_csv",
            "upstream_selection": (
                "modality_specific_label_selected_population_gating_and_"
                "artifact_filtering"
            ),
            "local_selection": (
                "all_expression_csv_rows_without_runtime_label_filter"
            ),
            "local_label_filter_effect_rows": modality_summaries[modality][
                "local_unmapped_rows"
            ],
            "full_population_claim_allowed": "false",
            "allowed_estimand": (
                "h_only_translation_within_modality_specific_upstream_"
                "pregated_aml_compartments"
            ),
            "status": (
                "conditional_sensitivity_ready_raw_primary_rebuild_pending"
                if modality_summaries[modality]["local_unmapped_rows"] == 0
                else "blocked_unmapped_processed_rows"
            ),
            "blocker": (
                "current processed event sets use upstream biological labels and "
                "are not commensurate full-population cohorts"
            ),
        }
        for modality in MODALITIES
    ]
    event_selection.append(
        {
            "dataset": "aml_clinical_stress",
            "modality": "clinical_flow",
            "input_artifact": "processed_cells_csv",
            "upstream_selection": (
                "label_dependent_composite_channel_split_and_population_filter"
            ),
            "local_selection": "not_allowed",
            "local_label_filter_effect_rows": "not_evaluated",
            "full_population_claim_allowed": "false",
            "allowed_estimand": "none_until_raw_label_free_reprocessing",
            "status": "blocked",
            "blocker": (
                "label leakage plus repeated-patient label reuse in 12/46 specimens"
            ),
        }
    )
    summary = {
        "cohort_policy": cohort_policy,
        "paired_specimens": len(paired),
        "paired_patients": len(
            {patient_id_from_specimen(specimen) for specimen in paired}
        ),
        "eligible_paired_specimens": len(eligible_paired),
        "eligible_paired_patients": len(
            {patient_id_from_specimen(specimen) for specimen in eligible_paired}
        ),
        "specimens": list(eligible_paired),
        "all_discovered_specimens": sorted(discovered),
        "unpaired_specimens": sorted(discovered - paired),
        "excluded_paired_specimens": sorted(excluded_paired),
        "low_event_specimens": sorted(low_event_specimens),
        "specimen_event_policy": {
            "primary_min_events_per_modality": primary_minimum,
            "low_event_flag_below": low_event_threshold,
            "sensitivity_exclude_below": sensitivity_threshold,
        },
        "modalities": modality_summaries,
        "source_hashes_complete": bool(hash_source_files),
        "source_bundle_digests": {
            "all_complete_files": _source_bundle_digest(
                rows,
                paired_only=False,
            ),
            "exact_paired_files": _source_bundle_digest(
                rows,
                paired_only=True,
            ),
        },
        "authoritative_metadata": metadata_summary,
    }
    return rows, event_selection, summary


def build_marker_records(data_rows: Sequence[Mapping[str, object]]) -> list[dict]:
    """Build both-direction H/Y/source-only/technical marker contracts."""

    first_path_by_modality = {}
    for row in data_rows:
        first_path_by_modality.setdefault(
            str(row["modality"]),
            Path(str(row["source_path"])),
        )
    headers = {
        modality: _header(first_path_by_modality[modality])
        for modality in MODALITIES
    }
    panel = build_pair_marker_manifest(
        headers["spectral_flow"],
        headers["cytof"],
    )
    if set(panel.common_markers) != set(H19) or len(panel.common_markers) != len(H19):
        raise Phase0AuditError(
            "Observed shared markers do not match the predeclared H19 contract"
        )
    lookups = {
        modality: _canonical_lookup(header) for modality, header in headers.items()
    }
    technical = set(DEFAULT_TECHNICAL_MARKERS)
    records = []
    for source, target in (
        ("spectral_flow", "cytof"),
        ("cytof", "spectral_flow"),
    ):
        direction = f"{source}_to_{target}"
        for modality in (source, target):
            ordered: list[tuple[str, str]] = [(marker, "H") for marker in H19]
            for original in headers[modality]:
                canonical = canonical_marker_name(original)
                if canonical in H19:
                    continue
                if original in technical:
                    role = "technical"
                elif modality == target:
                    role = "Y"
                else:
                    role = "source_only"
                ordered.append((canonical, role))
            for marker_order, (canonical, role) in enumerate(ordered):
                original = lookups[modality][canonical]
                channel_type = (
                    "technical" if original in technical else "biological"
                )
                records.append(
                    {
                        "dataset": DATASET_ID,
                        "direction": direction,
                        "modality": modality,
                        "canonical_marker": canonical,
                        "original_channel": original,
                        "role": role,
                        "analysis_role": (
                            "excluded" if role == "technical" else role
                        ),
                        "channel_type": channel_type,
                        "marker_order": marker_order,
                        "analysis_order": (
                            marker_order if role in {"H", "Y"} else ""
                        ),
                        "panel_order": headers[modality].index(original),
                        "transformation": _transformation(modality, original),
                        "transformation_confidence": "upstream_code_supported",
                        "included_primary": (
                            "true" if role in {"H", "Y"} else "false"
                        ),
                        "alias_rule": (
                            f"{original}->{canonical}"
                            if original != canonical
                            else "identity"
                        ),
                        "alias_evidence": (
                            "canonical_protein_name_equivalence"
                            if original != canonical
                            else "exact_channel_name"
                        ),
                    }
                )
    return records


def build_split_records(
    specimens: Sequence[str],
    *,
    n_splits: int,
    fold_seed: int,
    validation_fraction: float,
    created_date: str,
    base_split_manifest_path: str | Path | None = None,
) -> tuple[list[dict], dict]:
    """Materialize every specimen role for every frozen candidate fold."""

    if base_split_manifest_path is None:
        manifest = build_patient_grouped_manifest(
            specimens,
            n_splits=n_splits,
            seed=fold_seed,
            validation_fraction=validation_fraction,
            pair_name=PAIR_ID,
        )
        manifest["created_date"] = created_date
        manifest["derivation"] = "generated_candidate"
    else:
        base_path = Path(base_split_manifest_path)
        try:
            base = json.loads(base_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise Phase0AuditError(
                f"Cannot load base split manifest: {base_path}"
            ) from error
        if (
            base.get("pair") != PAIR_ID
            or int(base.get("fold_seed", -1)) != fold_seed
            or len(base.get("folds", ())) != n_splits
        ):
            raise Phase0AuditError(
                "Base split manifest disagrees with the protocol"
            )
        eligible = set(map(str, specimens))
        missing = sorted(eligible - set(map(str, base.get("specimens", ()))))
        if missing:
            raise Phase0AuditError(
                f"Eligible specimens are absent from base split: {missing}"
            )
        manifest = deepcopy(base)
        patient_to_specimens: dict[str, list[str]] = defaultdict(list)
        for specimen in sorted(eligible):
            patient_to_specimens[patient_id_from_specimen(specimen)].append(
                specimen
            )
        manifest["specimens"] = sorted(eligible)
        manifest["patients"] = sorted(patient_to_specimens)
        manifest["patient_to_specimens"] = dict(
            sorted(patient_to_specimens.items())
        )
        for fold in manifest["folds"]:
            for role in ("train", "validation", "test"):
                role_specimens = sorted(
                    set(map(str, fold[f"{role}_specimens"])) & eligible
                )
                fold[f"{role}_specimens"] = role_specimens
                fold[f"{role}_patients"] = sorted(
                    {patient_id_from_specimen(item) for item in role_specimens}
                )
        manifest["created_date"] = created_date
        manifest["derivation"] = "filtered_from_primary_assignments"
        manifest["base_split_manifest"] = base_path.as_posix()
        manifest["base_split_manifest_sha256"] = _sha256(base_path)
    records = []
    for fold in manifest["folds"]:
        fold_index = int(fold["fold_index"])
        for role in ("train", "validation", "test"):
            patients = set(map(str, fold[f"{role}_patients"]))
            for specimen in map(str, fold[f"{role}_specimens"]):
                patient = patient_id_from_specimen(specimen)
                if patient not in patients:
                    raise Phase0AuditError(
                        "Generated split specimen/patient mapping is inconsistent"
                    )
                records.append(
                    {
                        "dataset": DATASET_ID,
                        "pair": PAIR_ID,
                        "evaluation_fold": fold_index,
                        "role": role,
                        "patient_id": patient,
                        "specimen_id": specimen,
                    }
                )
    records.sort(
        key=lambda row: (
            int(row["evaluation_fold"]),
            str(row["patient_id"]),
            str(row["specimen_id"]),
        )
    )
    expected_specimens = set(map(str, specimens))
    for fold_index in range(n_splits):
        observed = {
            str(row["specimen_id"])
            for row in records
            if int(row["evaluation_fold"]) == fold_index
        }
        if observed != expected_specimens:
            raise Phase0AuditError(
                f"Split fold {fold_index} does not assign every eligible specimen"
            )
    test_folds_by_patient: dict[str, set[int]] = defaultdict(set)
    for row in records:
        if row["role"] == "test":
            test_folds_by_patient[str(row["patient_id"])].add(
                int(row["evaluation_fold"])
            )
    expected_patients = {
        patient_id_from_specimen(specimen) for specimen in expected_specimens
    }
    if set(test_folds_by_patient) != expected_patients or any(
        len(folds) != 1 for folds in test_folds_by_patient.values()
    ):
        raise Phase0AuditError(
            "Every eligible patient must be test in exactly one fold"
        )
    return records, manifest


def _stable_integer_seed(*tokens: object) -> int:
    payload = "|".join(map(str, tokens)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def build_reference_bank_records(
    data_records: Sequence[Mapping[str, object]],
    split_records: Sequence[Mapping[str, object]],
    protocol: Mapping,
) -> list[dict]:
    """Draft inner-fit and outer-refit banks without test-patient access.

    Row-index files remain pending until the source-file checksums and minimum
    cell rule are frozen. The seed and expected selected event count are already
    fixed here so the eventual materialization is deterministic.
    """

    data_by_modality_specimen = {
        (str(row["modality"]), str(row["specimen_id"])): row
        for row in data_records
    }
    split_by_fold_role: dict[
        tuple[int, str], list[Mapping[str, object]]
    ] = defaultdict(list)
    for row in split_records:
        split_by_fold_role[
            (int(row["evaluation_fold"]), str(row["role"]))
        ].append(row)
    seeds = tuple(map(int, protocol["training"]["seeds"]))
    cell_cap = int(protocol["reference_bank"]["cells_per_specimen_cap"])
    fit_stages = {
        str(stage): tuple(map(str, roles))
        for stage, roles in protocol["reference_bank"]["fit_stages"].items()
    }
    folds = sorted({fold for fold, _ in split_by_fold_role})
    records = []
    for fold in folds:
        for fit_stage, included_roles in fit_stages.items():
            fit_rows = [
                row
                for role in included_roles
                for row in split_by_fold_role[(fold, role)]
            ]
            specimens_by_patient: dict[str, list[str]] = defaultdict(list)
            for row in fit_rows:
                specimens_by_patient[str(row["patient_id"])].append(
                    str(row["specimen_id"])
                )
            n_patients = len(specimens_by_patient)
            if n_patients < 1:
                raise Phase0AuditError(
                    f"Fold {fold}/{fit_stage} has no fitting patients"
                )
            for direction, source, target in (
                ("spectral_flow_to_cytof", "spectral_flow", "cytof"),
                ("cytof_to_spectral_flow", "cytof", "spectral_flow"),
            ):
                role_modalities = (
                    ("target_predictor_bank", (target,)),
                    ("null_prior_bank", (target,)),
                    ("calibration_bank", (source, target)),
                )
                for seed in seeds:
                    for bank_role, modalities in role_modalities:
                        for modality in modalities:
                            bank_id = (
                                f"{DATASET_ID}:{direction}:fold{fold}:"
                                f"{fit_stage}:seed{seed}:{bank_role}:{modality}"
                            )
                            for patient, patient_specimens in sorted(
                                specimens_by_patient.items()
                            ):
                                patient_weight = 1.0 / n_patients
                                specimen_weight = (
                                    patient_weight / len(patient_specimens)
                                )
                                for specimen in sorted(patient_specimens):
                                    data_row = data_by_modality_specimen.get(
                                        (modality, specimen)
                                    )
                                    if data_row is None:
                                        raise Phase0AuditError(
                                            "Reference bank lacks a modality/specimen "
                                            f"inventory row: {(modality, specimen)}"
                                        )
                                    row_seed = _stable_integer_seed(
                                        protocol["protocol"]["id"],
                                        DATASET_ID,
                                        modality,
                                        specimen,
                                        seed,
                                        data_row["event_selection_rule_id"],
                                    )
                                    selected_events = min(
                                        int(data_row["eligible_events"]),
                                        cell_cap,
                                    )
                                    event_weight = (
                                        specimen_weight / selected_events
                                    )
                                    records.append(
                                        {
                                            "reference_bank_id": bank_id,
                                            "dataset": DATASET_ID,
                                            "pair": PAIR_ID,
                                            "direction": direction,
                                            "modality": modality,
                                            "evaluation_fold": fold,
                                            "fit_stage": fit_stage,
                                            "included_split_roles": ";".join(
                                                included_roles
                                            ),
                                            "seed": seed,
                                            "bank_role": bank_role,
                                            "patient_id": patient,
                                            "specimen_id": specimen,
                                            "row_index_file": (
                                                "pending/"
                                                f"{modality}/{specimen}/"
                                                f"seed_{seed}.npy"
                                            ),
                                            "row_index_sha256": "pending",
                                            "row_index_seed": row_seed,
                                            "sampling_algorithm": (
                                                "numpy_pcg64_uniform_without_"
                                                "replacement_sorted_v1"
                                            ),
                                            "row_index_dtype": "uint32",
                                            "event_count": selected_events,
                                            "patient_weight": (
                                                f"{patient_weight:.17g}"
                                            ),
                                            "specimen_weight": (
                                                f"{specimen_weight:.17g}"
                                            ),
                                            "event_weight": (
                                                f"{event_weight:.17g}"
                                            ),
                                            "status": (
                                                "pending_source_hash_and_"
                                                "row_materialization"
                                            ),
                                        }
                                    )
    return records


def _load_endpoint_sources(
    data_root: Path,
) -> tuple[list[tuple[str, pd.DataFrame]], list[dict], dict]:
    paths = {
        "cytof": data_root
        / "downstream_labels"
        / "AML_cytof_downstream_labels.csv",
        "spectral_flow": data_root
        / "downstream_labels"
        / "AML_spectral-flow_downstream_labels.csv",
    }
    sources = []
    duplicate_issues = []
    source_summary = {}
    for source, path in paths.items():
        if not path.is_file():
            raise Phase0AuditError(f"Missing endpoint annotation file: {path}")
        frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
        if "file_name" not in frame:
            raise Phase0AuditError(f"Endpoint file lacks file_name: {path}")
        frame["file_name"] = frame["file_name"].map(lambda value: Path(value).stem)
        for specimen, group in frame.groupby("file_name", sort=True):
            if len(group) < 2:
                continue
            unique = group.drop_duplicates()
            issue_type = (
                "duplicate_exact_annotation"
                if len(unique) == 1
                else "duplicate_nonidentical_annotation"
            )
            duplicate_issues.append(
                {
                    "conflict_type": issue_type,
                    "patient_id": patient_id_from_specimen(specimen),
                    "specimen_id": specimen,
                    "endpoint": "ALL",
                    "values": str(len(group)),
                    "provenance": source,
                }
            )
        frame = frame.drop_duplicates()
        sources.append((source, frame))
        source_summary[source] = {
            "path": path.as_posix(),
            "sha256": _sha256(path),
            "rows_after_exact_deduplication": len(frame),
        }
    return sources, duplicate_issues, source_summary


def audit_endpoints(
    data_root: str | Path,
    specimens: Sequence[str],
    split_records: Sequence[Mapping[str, object]],
    endpoint_policy: Mapping[str, object],
    specimen_visit_ids: Mapping[str, str] | None = None,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Reconcile specimen annotations and summarize held-out patient support."""

    specimen_set = set(map(str, specimens))
    values: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    annotation_sources_by_specimen: dict[str, set[str]] = defaultdict(set)
    endpoint_sources, audit_issues, endpoint_source_summary = (
        _load_endpoint_sources(Path(data_root))
    )
    for source, frame in endpoint_sources:
        for _, row in frame.iterrows():
            specimen = str(row["file_name"])
            if specimen not in specimen_set:
                continue
            annotation_sources_by_specimen[specimen].add(source)
            for endpoint in ENDPOINTS:
                value = str(row[f"{endpoint}_Status"]).strip()
                if value:
                    values[(specimen, endpoint)].append((source, value))

    reconciled = {}
    modality_conflicts = []
    for specimen in sorted(specimen_set):
        for endpoint in ENDPOINTS:
            observations = values.get((specimen, endpoint), [])
            unique = sorted({value for _, value in observations})
            if len(unique) > 1:
                reconciled[(specimen, endpoint)] = "conflict"
                modality_conflicts.append(
                    {
                        "conflict_type": "cross_annotation_source",
                        "patient_id": patient_id_from_specimen(specimen),
                        "specimen_id": specimen,
                        "endpoint": endpoint,
                        "values": ";".join(unique),
                        "provenance": ";".join(
                            f"{source}:{value}" for source, value in observations
                        ),
                    }
                )
            elif (
                unique
                and len(annotation_sources_by_specimen.get(specimen, set())) == 1
                and endpoint_policy["single_source_annotation"]
                == "unknown_until_common_provenance"
            ):
                reconciled[(specimen, endpoint)] = "unknown"
            elif unique:
                reconciled[(specimen, endpoint)] = unique[0]
            else:
                reconciled[(specimen, endpoint)] = "unknown"

    specimens_by_patient: dict[str, list[str]] = defaultdict(list)
    for specimen in sorted(specimen_set):
        specimens_by_patient[patient_id_from_specimen(specimen)].append(specimen)
    patient_status = {}
    longitudinal_conflicts = []
    for patient, patient_specimens in sorted(specimens_by_patient.items()):
        for endpoint in ENDPOINTS:
            known = {
                reconciled[(specimen, endpoint)]
                for specimen in patient_specimens
                if reconciled[(specimen, endpoint)] not in {"unknown", "conflict"}
            }
            has_modality_conflict = any(
                reconciled[(specimen, endpoint)] == "conflict"
                for specimen in patient_specimens
            )
            if has_modality_conflict or len(known) > 1:
                status = "conflict"
                longitudinal_conflicts.append(
                    {
                        "conflict_type": "longitudinal_patient",
                        "patient_id": patient,
                        "specimen_id": ";".join(patient_specimens),
                        "endpoint": endpoint,
                        "values": ";".join(sorted(known)),
                        "provenance": "specimen_level_status_disagreement",
                    }
                )
            elif known:
                status = next(iter(known))
            else:
                status = "unknown"
            patient_status[(patient, endpoint)] = status

    for specimen in sorted(specimen_set):
        sources = annotation_sources_by_specimen.get(specimen, set())
        if len(sources) == 1:
            audit_issues.append(
                {
                    "conflict_type": "single_annotation_source",
                    "patient_id": patient_id_from_specimen(specimen),
                    "specimen_id": specimen,
                    "endpoint": "ALL",
                    "values": ";".join(sorted(sources)),
                    "provenance": "requires_common_clinical_provenance_review",
                }
            )
        elif not sources:
            audit_issues.append(
                {
                    "conflict_type": "missing_annotation_source",
                    "patient_id": patient_id_from_specimen(specimen),
                    "specimen_id": specimen,
                    "endpoint": "ALL",
                    "values": "",
                    "provenance": "no_endpoint_metadata_row",
                }
            )

    conflicts = sorted(
        audit_issues + modality_conflicts + longitudinal_conflicts,
        key=lambda row: (
            row["endpoint"],
            row["patient_id"],
            row["conflict_type"],
        ),
    )
    conflict_keys = {
        (row["patient_id"], row["endpoint"])
        for row in longitudinal_conflicts
    }
    endpoint_records = []
    for specimen in sorted(specimen_set):
        patient = patient_id_from_specimen(specimen)
        for endpoint in ENDPOINTS:
            value = reconciled[(specimen, endpoint)]
            if (patient, endpoint) in conflict_keys or value == "conflict":
                eligibility = "not_evaluable"
                reason = "longitudinal_or_annotation_source_conflict"
            elif value == "unknown":
                eligibility = "not_evaluable"
                reason = (
                    "single_source_annotation_unverified"
                    if len(annotation_sources_by_specimen.get(specimen, set()))
                    == 1
                    else "missing_annotation"
                )
            else:
                eligibility = "secondary"
                reason = ""
            provenance = ";".join(
                f"{source}:{observed}"
                for source, observed in values.get((specimen, endpoint), [])
            )
            endpoint_records.append(
                {
                    "dataset": DATASET_ID,
                    "endpoint": endpoint,
                    "patient_id": patient,
                    "visit_id": (
                        specimen_visit_ids.get(specimen, specimen)
                        if specimen_visit_ids is not None
                        else specimen
                    ),
                    "specimen_id": specimen,
                    "value": value,
                    "provenance": provenance or "no_annotation",
                    "eligibility": eligibility,
                    "reason": reason,
                }
            )

    patients_by_fold_role: dict[tuple[int, str], set[str]] = defaultdict(set)
    for row in split_records:
        patients_by_fold_role[
            (int(row["evaluation_fold"]), str(row["role"]))
        ].add(str(row["patient_id"]))
    support = []
    overall = {}
    for endpoint in ENDPOINTS:
        endpoint_rows = []
        for fold in sorted({key[0] for key in patients_by_fold_role}):
            for role in ("train", "validation", "test"):
                statuses = [
                    patient_status[(patient, endpoint)]
                    for patient in sorted(patients_by_fold_role[(fold, role)])
                ]
                row = {
                    "dataset": DATASET_ID,
                    "endpoint": endpoint,
                    "evaluation_fold": fold,
                    "role": role,
                    "mutant_patients": statuses.count("Mutant"),
                    "wildtype_patients": statuses.count("Wildtype"),
                    "unknown_patients": statuses.count("unknown"),
                    "conflict_patients": statuses.count("conflict"),
                    "patients": len(statuses),
                }
                endpoint_rows.append(row)
                support.append(row)
        test_rows = [row for row in endpoint_rows if row["role"] == "test"]
        total_statuses = [
            patient_status[(patient, endpoint)]
            for patient in sorted(specimens_by_patient)
        ]
        total_mutant = total_statuses.count("Mutant")
        total_wildtype = total_statuses.count("Wildtype")
        min_test_mutant = min(row["mutant_patients"] for row in test_rows)
        min_test_wildtype = min(row["wildtype_patients"] for row in test_rows)
        if min_test_mutant == 0 or min_test_wildtype == 0:
            eligibility = "descriptive_only_not_foldwise_evaluable"
        elif (
            total_mutant
            >= int(endpoint_policy["formal_support_min_total_per_class"])
            and total_wildtype
            >= int(endpoint_policy["formal_support_min_total_per_class"])
            and min_test_mutant
            >= int(endpoint_policy["formal_support_min_test_fold_per_class"])
            and min_test_wildtype
            >= int(endpoint_policy["formal_support_min_test_fold_per_class"])
        ):
            eligibility = "support_qualified_secondary"
        else:
            eligibility = "secondary_only_sparse_support"
        overall[endpoint] = {
            "eligibility": eligibility,
            "mutant_patients": total_mutant,
            "wildtype_patients": total_wildtype,
            "unknown_patients": total_statuses.count("unknown"),
            "conflict_patient_count": total_statuses.count("conflict"),
            "minimum_test_mutant_patients_per_fold": min_test_mutant,
            "minimum_test_wildtype_patients_per_fold": min_test_wildtype,
            "conflict_patients": sorted(
                patient
                for patient in specimens_by_patient
                if patient_status[(patient, endpoint)] == "conflict"
            ),
        }
        for row in endpoint_rows:
            row["endpoint_cv_eligibility"] = eligibility
    overall["_metadata_sources"] = endpoint_source_summary
    overall["_audit_issue_counts"] = {
        issue_type: sum(
            row["conflict_type"] == issue_type for row in conflicts
        )
        for issue_type in sorted({row["conflict_type"] for row in conflicts})
    }
    overall["_conflict_rule"] = {
        "concordant_nonmissing_visits": "collapse_to_patient_status",
        "all_missing_visits": "unknown",
        "discordant_nonmissing_visits": "conflict_and_endpoint_specific_exclusion",
        "ever_mutant_collapse": "forbidden",
        "specimen_suffix_temporal_order": "not_assumed",
        "single_source_annotation": str(
            endpoint_policy["single_source_annotation"]
        ),
    }
    return endpoint_records, support, conflicts, overall


def build_phase0_audit(
    data_root: str | Path,
    protocol: Mapping,
    *,
    hash_source_files: bool = False,
    authoritative_metadata_path: str | Path | None = None,
    cohort_policy: str = "primary",
    base_split_manifest_path: str | Path | None = None,
) -> dict:
    """Build all deterministic Phase 0 AML audit tables in memory."""

    validate_primary_benchmark_config(protocol)
    if (
        cohort_policy == "low_event_exclusion_sensitivity"
        and base_split_manifest_path is None
    ):
        raise Phase0AuditError(
            "Low-event sensitivity must preserve a supplied primary split"
        )
    data_records, event_selection, inventory_summary = audit_data_inventory(
        data_root,
        hash_source_files=hash_source_files,
        authoritative_metadata_path=authoritative_metadata_path,
        authoritative_metadata_contract=protocol["data"][
            "authoritative_metadata"
        ],
        cohort_policy=cohort_policy,
        specimen_event_policy=protocol["data"]["specimen_event_policy"],
    )
    marker_records = build_marker_records(data_records)
    split_records, split_manifest = build_split_records(
        inventory_summary["specimens"],
        n_splits=int(protocol["split"]["n_splits"]),
        fold_seed=int(protocol["split"]["fold_seed"]),
        validation_fraction=float(protocol["split"]["validation_fraction"]),
        created_date=str(protocol["protocol"]["created_date"]),
        base_split_manifest_path=base_split_manifest_path,
    )
    bank_records = build_reference_bank_records(
        data_records,
        split_records,
        protocol,
    )
    endpoint_records, endpoint_support, endpoint_conflicts, endpoint_summary = (
        audit_endpoints(
            data_root,
            inventory_summary["specimens"],
            split_records,
            protocol["endpoint_policy"],
            {
                str(row["specimen_id"]): str(row["visit_id"])
                for row in data_records
                if str(row["pair_status"]) == "exact_pair"
            },
        )
    )
    return {
        "summary": {
            "audit_id": (
                "aml_sf_cytof_phase0_draft_v1"
                if cohort_policy == "primary"
                else "aml_sf_cytof_low_event_exclusion_sensitivity_v1"
            ),
            "created_date": str(protocol["protocol"]["created_date"]),
            "protocol_id": str(protocol["protocol"]["id"]),
            "protocol_digest": protocol_digest(protocol),
            "status": "draft",
            "allowed_current_processed_estimand": (
                "h_only_translation_within_modality_specific_upstream_"
                "pregated_aml_compartments"
            ),
            "current_processed_analysis_role": "conditional_sensitivity_only",
            "raw_label_free_primary_ready": False,
            "full_population_claim_allowed": False,
            "cohort_policy": cohort_policy,
            "split_derivation": split_manifest["derivation"],
            "inventory": inventory_summary,
            "endpoint_summary": endpoint_summary,
            "unresolved_blockers": [
                blocker
                for blocker in (
                    (
                        "source file SHA-256 hashes pending"
                        if not hash_source_files
                        else ""
                    ),
                    (
                        "authoritative patient/visit mapping unavailable"
                        if authoritative_metadata_path is None
                        else ""
                    ),
                    "reference-bank row indices and digests pending",
                    "raw label-free AML SF/CyTOF rebuild not yet materialized",
                    (
                        "portable AML metadata snapshot/accession unresolved"
                        if protocol["data"]["authoritative_metadata"][
                            "release_reference"
                        ]
                        == "pending"
                        else ""
                    ),
                )
                if blocker
            ],
            "deferred_secondary_work": [
                "clinical detector-level reconstruction and provenance audit"
            ],
        },
        "event_selection": event_selection,
        "data_records": data_records,
        "marker_records": marker_records,
        "split_records": split_records,
        "bank_records": bank_records,
        "split_manifest": split_manifest,
        "endpoint_records": endpoint_records,
        "endpoint_support": endpoint_support,
        "endpoint_conflicts": endpoint_conflicts,
    }


def _write_csv(
    path: Path,
    records: Iterable[Mapping[str, object]],
    *,
    empty_fields: Sequence[str] | None = None,
) -> None:
    rows = list(records)
    if rows:
        fields = list(rows[0])
    elif empty_fields:
        fields = list(empty_fields)
    else:
        raise Phase0AuditError(f"Cannot infer fields for an empty table: {path}")
    if any(set(row) != set(fields) for row in rows):
        raise Phase0AuditError(f"Audit rows have inconsistent fields: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_phase0_audit(audit: Mapping, output_root: str | Path) -> dict:
    """Write deterministic draft tables plus content digests."""

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    table_names = (
        "event_selection",
        "data_records",
        "marker_records",
        "split_records",
        "bank_records",
        "endpoint_records",
        "endpoint_support",
        "endpoint_conflicts",
    )
    paths = {}
    for name in table_names:
        path = output / f"{name}.csv"
        _write_csv(
            path,
            audit[name],
            empty_fields=(
                (
                    "conflict_type",
                    "patient_id",
                    "specimen_id",
                    "endpoint",
                    "values",
                    "provenance",
                )
                if name == "endpoint_conflicts"
                else None
            ),
        )
        paths[name] = path
    split_path = output / "split_manifest.json"
    split_path.write_text(
        json.dumps(audit["split_manifest"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["split_manifest"] = split_path
    summary = dict(audit["summary"])
    summary["unresolved_blockers"] = [
        blocker for blocker in summary["unresolved_blockers"] if blocker
    ]
    summary_path = output / "audit_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["audit_summary"] = summary_path
    digests = {
        name: {
            "path": path.as_posix(),
            "sha256": _sha256(path),
        }
        for name, path in sorted(paths.items())
    }
    digest_path = output / "artifact_digests.json"
    digest_path.write_text(
        json.dumps(digests, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "output_root": output.as_posix(),
        "artifact_digests": digests,
        "artifact_digest_path": digest_path.as_posix(),
    }
