import copy
import csv
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.benchmark.contract import BenchmarkContractError
from src.benchmark.manifest_validation import REQUIRED_COLUMNS, sha256_file
from src.config import load_config
from src.wrappers.benchmark_preflight import run_preflight


def _records(manifest_type, *, split_mismatch=False, bank_leak=False):
    common = {
        "data": [
            {
                "dataset": "fixture",
                "pair": "sf_cytof",
                "modality": "target",
                "specimen_id": f"S{index}",
                "patient_id": f"P{index}",
                "visit_id": "1",
                "collection_id": "1",
                "collection_date": f"2024-01-0{index}",
                "visit_order": "1",
                "source_path": f"cells/S{index}.csv",
                "source_sha256": str(index) * 64,
                "cells_path": f"cells/S{index}.csv",
                "cells_sha256": str(index) * 64,
                "labels_path": f"labels/S{index}.csv",
                "labels_sha256": str(index + 1) * 64,
                "metadata_source_id": "fixture_metadata",
                "metadata_sha256": "a" * 64,
                "header_sha256": "b" * 64,
                "total_events": "10",
                "eligible_events": "10",
                "event_selection_rule_id": "label_free_v1",
                "preprocessing_id": "fixture_v1",
                "upstream_label_selection": "false",
                "runtime_label_selection": "false",
                "claim_scope": "fixture",
                "pair_status": "exact_pair",
                "excluded": "false",
                "exclusion_reason": "",
                "qc_flag": "",
            }
            for index in range(1, 6)
        ],
        "markers": [
            {
                "dataset": "fixture",
                "direction": "source_to_target",
                "modality": "target",
                "canonical_marker": "Y1",
                "original_channel": "Y1",
                "role": "H",
                "analysis_role": "H",
                "channel_type": "biological",
                "marker_order": "0",
                "panel_order": "0",
                "analysis_order": "0",
                "transformation": "none",
                "transformation_confidence": "fixture",
                "alias_rule": "identity",
                "alias_evidence": "exact_channel_name",
            }
        ],
        "splits": [
            {
                "dataset": "fixture",
                "pair": "sf_cytof",
                "evaluation_fold": str(fold),
                "role": (
                    "test"
                    if patient_index - 1 == fold
                    else "validation"
                    if patient_index - 1 == (fold + 1) % 5
                    else "train"
                ),
                "patient_id": (
                    "P6"
                    if split_mismatch and patient_index == 2
                    else f"P{patient_index}"
                ),
                "specimen_id": f"S{patient_index}",
            }
            for fold in range(5)
            for patient_index in range(1, 6)
        ],
        "endpoints": [
            {
                "dataset": "fixture",
                "endpoint": "phenotype",
                "patient_id": "P1",
                "visit_id": "V1",
                "value": "positive",
                "provenance": "fixture",
                "eligibility": "secondary",
                "reason": "",
            }
        ],
        "banks": [
            {
                "reference_bank_id": "bank-1",
                "dataset": "fixture",
                "pair": "sf_cytof",
                "direction": "source_to_target",
                "modality": "target",
                "evaluation_fold": "0",
                "fit_stage": "inner_fit",
                "included_split_roles": "train",
                "seed": "4207",
                "bank_role": "null_prior_bank",
                "patient_id": (
                    "P1" if bank_leak and index == 3 else f"P{index}"
                ),
                "specimen_id": (
                    "S1" if bank_leak and index == 3 else f"S{index}"
                ),
                "row_index_file": f"rows_S{index}.npy",
                "row_index_sha256": "",
                "row_index_seed": str(1000 + index),
                "event_count": "10",
                "patient_weight": f"{1 / 3:.17g}",
                "specimen_weight": f"{1 / 3:.17g}",
                "event_weight": f"{1 / 30:.17g}",
                "sampling_algorithm": (
                    "numpy_pcg64_uniform_without_replacement_sorted_v1"
                ),
                "row_index_dtype": "uint32",
                "status": "materialized_validated",
            }
            for index in range(3, 6)
        ],
        "stress": [
            {
                "condition_id": "control",
                "dataset": "fixture",
                "factor": "none",
                "level": "0",
                "seed": "4207",
                "content_file": "condition.yaml",
                "content_sha256": "",
            }
        ],
    }
    return common[manifest_type]


def _write_manifest_bundle(tmp_path, *, split_mismatch=False, bank_leak=False):
    protocol = copy.deepcopy(load_config("configs/benchmark/protocol_v1.yaml"))
    protocol["protocol"]["status"] = "frozen"
    protocol["data"]["authoritative_metadata"]["release_reference"] = (
        "fixture://metadata"
    )
    protocol["methods"]["primary"] = [
        "global_target_median",
        "target_prior_sampler",
        "ridge",
        "knn50",
        "simple_mlp",
    ]
    for manifest_type in protocol["manifests"]:
        manifest_dir = tmp_path / manifest_type
        manifest_dir.mkdir()
        records = _records(
            manifest_type,
            split_mismatch=split_mismatch,
            bank_leak=bank_leak,
        )
        if manifest_type == "banks":
            for record in records:
                referenced = manifest_dir / record["row_index_file"]
                np.save(
                    referenced,
                    np.arange(10, dtype=np.uint32),
                    allow_pickle=False,
                )
                record["row_index_sha256"] = sha256_file(referenced)
        elif manifest_type == "stress":
            referenced = manifest_dir / "condition.yaml"
            referenced.write_text("condition: control\n")
            records[0]["content_sha256"] = sha256_file(referenced)

        records_path = manifest_dir / "records.csv"
        with records_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=sorted(REQUIRED_COLUMNS[manifest_type]),
            )
            writer.writeheader()
            writer.writerows(records)
        index_path = manifest_dir / "index.yaml"
        index = {
            "manifest": {
                "type": manifest_type,
                "protocol_id": protocol["protocol"]["id"],
                "version": 1,
                "status": "frozen",
            },
            "records": {
                "path": records_path.name,
                "sha256": sha256_file(records_path),
            },
        }
        index_path.write_text(yaml.safe_dump(index, sort_keys=False))
        protocol["manifests"][manifest_type] = str(index_path)
        protocol["manifest_digests"][manifest_type] = sha256_file(index_path)

    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False))
    return protocol_path


def test_draft_preflight_reports_missing_manifests_without_claiming_ready():
    result = run_preflight("configs/benchmark/protocol_v1.yaml")
    assert result["status"] == "ok"
    assert result["protocol_status"] == "draft"
    assert result["ready_for_full_run"] is False
    assert result["missing_manifest_paths"]
    assert result["pending_manifest_digests"]
    repository_root = Path(result["repository_root"])
    assert repository_root == Path.cwd().resolve()
    assert Path(result["manifest_paths"]["data"]) == (
        repository_root / "benchmark/manifests/data/index.yaml"
    )


def test_full_run_preflight_rejects_draft_protocol():
    with pytest.raises(BenchmarkContractError, match="must be frozen"):
        run_preflight("configs/benchmark/protocol_v1.yaml", mode="full")


def test_full_preflight_rejects_empty_manifest_directories(tmp_path):
    protocol = copy.deepcopy(load_config("configs/benchmark/protocol_v1.yaml"))
    protocol["protocol"]["status"] = "frozen"
    protocol["data"]["authoritative_metadata"]["release_reference"] = (
        "fixture://metadata"
    )
    protocol["methods"]["primary"] = [
        "global_target_median",
        "target_prior_sampler",
        "ridge",
        "knn50",
        "simple_mlp",
    ]
    for name in protocol["manifests"]:
        path = tmp_path / name
        path.mkdir()
        protocol["manifests"][name] = str(path)
        protocol["manifest_digests"][name] = "0" * 64
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False))

    with pytest.raises(BenchmarkContractError, match="missing="):
        run_preflight(protocol_path, mode="full")


def test_full_preflight_validates_content_and_cross_manifest_ids(tmp_path):
    protocol_path = _write_manifest_bundle(tmp_path)
    result = run_preflight(protocol_path, mode="full")
    assert result["ready_for_full_run"] is True
    assert result["manifest_bundle_validated"] is True
    assert result["validated_manifest_types"] == [
        "banks",
        "data",
        "endpoints",
        "markers",
        "splits",
        "stress",
    ]


def test_full_preflight_rejects_manifest_digest_mismatch(tmp_path):
    protocol_path = _write_manifest_bundle(tmp_path)
    protocol = load_config(protocol_path)
    protocol["manifest_digests"]["data"] = "0" * 64
    protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False))
    with pytest.raises(BenchmarkContractError, match="index digest mismatch"):
        run_preflight(protocol_path, mode="full")


def test_full_preflight_rejects_cross_manifest_patient_mismatch(tmp_path):
    protocol_path = _write_manifest_bundle(tmp_path, split_mismatch=True)
    with pytest.raises(BenchmarkContractError, match="mapping disagree"):
        run_preflight(protocol_path, mode="full")


def test_full_preflight_rejects_test_patient_in_reference_bank(tmp_path):
    protocol_path = _write_manifest_bundle(tmp_path, bank_leak=True)
    with pytest.raises(BenchmarkContractError, match="outside its fitting roles"):
        run_preflight(protocol_path, mode="full")
