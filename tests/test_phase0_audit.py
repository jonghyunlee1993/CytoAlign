import copy
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from src.benchmark.phase0_audit import (
    ENDPOINTS,
    H19,
    Phase0AuditError,
    build_phase0_audit,
    write_phase0_audit,
)
from src.benchmark.reference_rows import materialize_reference_row_indices
from src.config import load_config


def _write_fixture(
    root: Path,
    *,
    longitudinal_conflict=False,
    header_mismatch=False,
    numeric_visits=False,
):
    spectral_header = [*H19, "FSC-A", "SF-Y"]
    cytof_header = [
        "CD197" if marker == "CCR7" else "CD279" if marker == "PD-1" else marker
        for marker in H19
    ] + ["CY-Y"]
    specimens = [
        f"R{index:04d}_{100 + index}" if numeric_visits else f"R{index:04d}_A"
        for index in range(1, 6)
    ]
    if longitudinal_conflict:
        specimens.append("R0001_B")
    for modality, header in (
        ("spectral_flow", spectral_header),
        ("cytof", cytof_header),
    ):
        (root / modality / "cells").mkdir(parents=True)
        (root / modality / "labels").mkdir(parents=True)
        for specimen_index, specimen in enumerate(specimens):
            current_header = list(header)
            if header_mismatch and specimen_index == 1 and modality == "cytof":
                current_header[-1] = "BROKEN"
            pd.DataFrame(
                [[float(column + row) for column in range(len(current_header))] for row in range(2)],
                columns=current_header,
            ).to_csv(root / modality / "cells" / f"{specimen}.csv", index=False)
            pd.DataFrame(
                {"cell_type": ["Blast", "T cell CD4"]}
            ).to_csv(root / modality / "labels" / f"{specimen}.csv", index=False)

    endpoint_dir = root / "downstream_labels"
    endpoint_dir.mkdir()
    endpoint_rows = []
    for index, specimen in enumerate(specimens):
        status = "Mutant" if index % 2 == 0 else "Wildtype"
        if longitudinal_conflict and specimen == "R0001_B":
            status = "Wildtype"
        row = {"file_name": specimen}
        for endpoint in ENDPOINTS:
            row[f"{endpoint}_Status"] = status
        row["IDH_Status_Fil"] = ""
        endpoint_rows.append(row)
    frame = pd.DataFrame(endpoint_rows)
    frame.to_csv(
        endpoint_dir / "AML_cytof_downstream_labels.csv",
        index=False,
    )
    frame.to_csv(
        endpoint_dir / "AML_spectral-flow_downstream_labels.csv",
        index=False,
    )
    return specimens


def test_phase0_audit_builds_patient_grouped_draft_artifacts(tmp_path):
    specimens = _write_fixture(tmp_path)
    audit = build_phase0_audit(
        tmp_path,
        load_config("configs/benchmark/protocol_v1.yaml"),
    )

    assert audit["summary"]["inventory"]["paired_specimens"] == len(specimens)
    assert audit["summary"]["inventory"]["paired_patients"] == 5
    assert audit["summary"]["full_population_claim_allowed"] is False
    assert len(audit["data_records"]) == len(specimens) * 2
    assert all(row["local_unmapped_rows"] == 0 for row in audit["data_records"])
    assert len(audit["split_records"]) == len(specimens) * 5
    assert audit["bank_records"]
    role_lookup = {
        (
            row["evaluation_fold"],
            row["patient_id"],
        ): row["role"]
        for row in audit["split_records"]
    }
    assert all(
        role_lookup[(row["evaluation_fold"], row["patient_id"])]
        in set(row["included_split_roles"].split(";"))
        for row in audit["bank_records"]
    )
    assert {
        row["fit_stage"] for row in audit["bank_records"]
    } == {"inner_fit", "outer_refit"}
    assert all(row["row_index_sha256"] == "pending" for row in audit["bank_records"])
    assert {
        row["role"]
        for row in audit["marker_records"]
    } == {"H", "Y", "source_only", "technical"}
    assert len(audit["endpoint_support"]) == len(ENDPOINTS) * 5 * 3

    result = write_phase0_audit(audit, tmp_path / "audit")
    assert Path(result["artifact_digest_path"]).is_file()
    assert len(result["artifact_digests"]) == 10


def test_phase0_audit_flags_longitudinal_endpoint_conflicts(tmp_path):
    _write_fixture(tmp_path, longitudinal_conflict=True)
    audit = build_phase0_audit(
        tmp_path,
        load_config("configs/benchmark/protocol_v1.yaml"),
    )
    conflicts = audit["endpoint_conflicts"]
    assert {
        (row["patient_id"], row["endpoint"])
        for row in conflicts
        if row["conflict_type"] == "longitudinal_patient"
    } == {("R0001", endpoint) for endpoint in ENDPOINTS}
    assert all(
        audit["summary"]["endpoint_summary"][endpoint]["conflict_patients"]
        == ["R0001"]
        for endpoint in ENDPOINTS
    )


def test_phase0_audit_rejects_inconsistent_panel_headers(tmp_path):
    _write_fixture(tmp_path, header_mismatch=True)
    with pytest.raises(Phase0AuditError, match="header differs"):
        build_phase0_audit(
            tmp_path,
            load_config("configs/benchmark/protocol_v1.yaml"),
        )


def test_phase0_audit_uses_digest_locked_authoritative_visit_metadata(tmp_path):
    specimens = _write_fixture(tmp_path, numeric_visits=True)
    metadata_path = tmp_path / "AML_meta_111224.csv"
    pd.DataFrame(
        {
            "Reg. ID": [specimen.split("_")[0] for specimen in specimens],
            "Coll. ID": [specimen.split("_")[1] for specimen in specimens],
            "Date": [f"1/{index}/24" for index in range(1, 6)],
        }
    ).to_csv(metadata_path, index=False)
    digest = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    protocol = copy.deepcopy(load_config("configs/benchmark/protocol_v1.yaml"))
    protocol["data"]["authoritative_metadata"]["expected_sha256"] = digest

    audit = build_phase0_audit(
        tmp_path,
        protocol,
        authoritative_metadata_path=metadata_path,
    )
    assert {
        row["visit_id_provenance"] for row in audit["data_records"]
    } == {"authoritative"}
    assert {
        row["collection_date"] for row in audit["data_records"]
    } == {f"2024-01-0{index}" for index in range(1, 6)}
    assert (
        audit["summary"]["inventory"]["authoritative_metadata"]["sha256"]
        == digest
    )


def test_phase0_audit_records_unpaired_discovery_rows_as_excluded(tmp_path):
    specimens = _write_fixture(tmp_path)
    unpaired = "R9999_999"
    first = specimens[0]
    cells = pd.read_csv(
        tmp_path / "spectral_flow" / "cells" / f"{first}.csv"
    )
    labels = pd.read_csv(
        tmp_path / "spectral_flow" / "labels" / f"{first}.csv"
    )
    cells.to_csv(
        tmp_path / "spectral_flow" / "cells" / f"{unpaired}.csv",
        index=False,
    )
    labels.to_csv(
        tmp_path / "spectral_flow" / "labels" / f"{unpaired}.csv",
        index=False,
    )

    audit = build_phase0_audit(
        tmp_path,
        load_config("configs/benchmark/protocol_v1.yaml"),
    )
    records = [
        row for row in audit["data_records"] if row["specimen_id"] == unpaired
    ]
    assert len(records) == 1
    assert records[0]["pair_status"] == "spectral_flow_only"
    assert records[0]["excluded"] == "true"
    assert records[0]["exclusion_reason"] == "no_exact_cross_modality_pair"
    assert audit["summary"]["inventory"]["unpaired_specimens"] == [unpaired]


def test_phase0_materializes_one_shared_label_free_reservoir_per_seed(tmp_path):
    specimens = _write_fixture(tmp_path)
    audit = build_phase0_audit(
        tmp_path,
        load_config("configs/benchmark/protocol_v1.yaml"),
        hash_source_files=True,
    )
    output = tmp_path / "audit"
    summary = materialize_reference_row_indices(audit, output)

    assert summary["unique_artifacts"] == len(specimens) * 2 * 3
    assert summary["created_artifacts"] == summary["unique_artifacts"]
    assert all(
        row["status"] == "materialized_validated"
        for row in audit["bank_records"]
    )
    artifacts_by_contract = {}
    for row in audit["bank_records"]:
        key = (row["modality"], row["specimen_id"], row["seed"])
        artifact = (row["row_index_file"], row["row_index_sha256"])
        assert artifacts_by_contract.setdefault(key, artifact) == artifact


def test_low_event_sensitivity_preserves_primary_patient_fold_assignments(
    tmp_path,
):
    specimens = _write_fixture(tmp_path)
    low_specimen = specimens[0]
    for modality in ("spectral_flow", "cytof"):
        cells_path = tmp_path / modality / "cells" / f"{low_specimen}.csv"
        labels_path = tmp_path / modality / "labels" / f"{low_specimen}.csv"
        pd.read_csv(cells_path).head(1).to_csv(cells_path, index=False)
        pd.read_csv(labels_path).head(1).to_csv(labels_path, index=False)
    protocol = copy.deepcopy(load_config("configs/benchmark/protocol_v1.yaml"))
    protocol["data"]["specimen_event_policy"]["low_event_flag_below"] = 2
    protocol["data"]["specimen_event_policy"]["sensitivity_exclude_below"] = 2
    primary = build_phase0_audit(tmp_path, protocol)
    primary_output = tmp_path / "primary"
    write_phase0_audit(primary, primary_output)

    sensitivity = build_phase0_audit(
        tmp_path,
        protocol,
        cohort_policy="low_event_exclusion_sensitivity",
        base_split_manifest_path=primary_output / "split_manifest.json",
    )
    assert sensitivity["summary"]["split_derivation"] == (
        "filtered_from_primary_assignments"
    )
    assert sensitivity["summary"]["inventory"]["excluded_paired_specimens"] == [
        low_specimen
    ]
    primary_roles = {
        (
            row["evaluation_fold"],
            row["specimen_id"],
        ): row["role"]
        for row in primary["split_records"]
        if row["specimen_id"] != low_specimen
    }
    sensitivity_roles = {
        (
            row["evaluation_fold"],
            row["specimen_id"],
        ): row["role"]
        for row in sensitivity["split_records"]
    }
    assert sensitivity_roles == primary_roles


def test_low_event_sensitivity_rejects_fold_redraw(tmp_path):
    _write_fixture(tmp_path)
    with pytest.raises(Phase0AuditError, match="preserve a supplied primary split"):
        build_phase0_audit(
            tmp_path,
            load_config("configs/benchmark/protocol_v1.yaml"),
            cohort_policy="low_event_exclusion_sensitivity",
        )


def test_single_source_endpoint_annotation_is_conservatively_unknown(tmp_path):
    specimens = _write_fixture(tmp_path)
    cytof_endpoint = (
        tmp_path / "downstream_labels" / "AML_cytof_downstream_labels.csv"
    )
    frame = pd.read_csv(cytof_endpoint)
    frame.loc[frame["file_name"] != specimens[0]].to_csv(
        cytof_endpoint,
        index=False,
    )

    audit = build_phase0_audit(
        tmp_path,
        load_config("configs/benchmark/protocol_v1.yaml"),
    )
    records = [
        row
        for row in audit["endpoint_records"]
        if row["specimen_id"] == specimens[0]
    ]
    assert {row["value"] for row in records} == {"unknown"}
    assert {
        row["reason"] for row in records
    } == {"single_source_annotation_unverified"}
