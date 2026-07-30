"""Validation for machine-readable benchmark protocols.

The legacy experiment configuration remains intentionally permissive for
reproduction.  New benchmark runs must pass this stricter, fail-closed
contract before loading patient data or fitting a model.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import date

from src.benchmark.method_registry import primary_method_violations


class BenchmarkContractError(ValueError):
    """Raised when a proposed primary benchmark run violates the protocol."""


def _mapping(parent: Mapping, key: str) -> Mapping:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise BenchmarkContractError(f"{key!r} must be a mapping")
    return value


def _sequence(parent: Mapping, key: str) -> tuple:
    value = parent.get(key)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise BenchmarkContractError(f"{key!r} must be a sequence")
    return tuple(value)


def _require_literal(parent: Mapping, key: str, expected: object) -> None:
    if key not in parent:
        raise BenchmarkContractError(f"Missing required key: {key}")
    if type(parent[key]) is not type(expected) or parent[key] != expected:
        raise BenchmarkContractError(
            f"{key!r} must be {expected!r}, got {parent[key]!r}"
        )


def _strict_int(
    parent: Mapping,
    key: str,
    *,
    minimum: int,
) -> int:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BenchmarkContractError(
            f"{key!r} must be an integer greater than or equal to {minimum}"
        )
    return value


def _strict_fraction(parent: Mapping, key: str) -> float:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkContractError(f"{key!r} must be numeric")
    result = float(value)
    if not 0.0 < result < 1.0:
        raise BenchmarkContractError(f"{key!r} must be strictly between 0 and 1")
    return result


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_primary_benchmark_config(config: Mapping) -> None:
    """Reject leakage-prone or non-auditable primary benchmark configs."""

    if not isinstance(config, Mapping):
        raise BenchmarkContractError("Benchmark config must be a mapping")

    protocol = _mapping(config, "protocol")
    for key in ("id", "version", "status"):
        if key not in protocol:
            raise BenchmarkContractError(f"protocol.{key} is required")
    if not isinstance(protocol["id"], str) or not protocol["id"].strip():
        raise BenchmarkContractError("protocol.id must be non-empty")
    _strict_int(protocol, "version", minimum=1)
    if protocol["status"] not in {"draft", "frozen"}:
        raise BenchmarkContractError("protocol.status must be draft or frozen")
    try:
        date.fromisoformat(str(protocol["created_date"]))
    except (KeyError, ValueError) as error:
        raise BenchmarkContractError(
            "protocol.created_date must be ISO YYYY-MM-DD"
        ) from error

    access = _mapping(config, "information_access")
    _require_literal(access, "translator_cell_labels", "forbidden")
    _require_literal(access, "translator_endpoint_labels", "forbidden")
    _require_literal(access, "downstream_outcome_labels", "outer_train_only")
    _require_literal(access, "test_target_statistics", "forbidden")
    _require_literal(access, "label_informed_track", "oracle_supplement_only")

    data = _mapping(config, "data")
    _require_literal(data, "sampling", "reservoir")
    _require_literal(data, "inference_unit", "patient")
    _require_literal(
        data,
        "longitudinal_aggregation",
        "patient_first_equal_specimen_weight",
    )
    max_cells = _strict_int(data, "max_cells_per_specimen", minimum=1)
    specimen_policy = _mapping(data, "specimen_event_policy")
    primary_min_events = _strict_int(
        specimen_policy,
        "primary_min_events_per_modality",
        minimum=1,
    )
    low_event_flag = _strict_int(
        specimen_policy,
        "low_event_flag_below",
        minimum=primary_min_events,
    )
    _require_literal(
        specimen_policy,
        "primary_low_event_action",
        "include_with_flag",
    )
    sensitivity_exclusion = _strict_int(
        specimen_policy,
        "sensitivity_exclude_below",
        minimum=primary_min_events,
    )
    if sensitivity_exclusion != low_event_flag:
        raise BenchmarkContractError(
            "The low-event flag and exclusion-sensitivity thresholds must match"
        )
    metadata = _mapping(data, "authoritative_metadata")
    _require_literal(metadata, "artifact_id", "AML_meta_111224")
    if not _is_sha256(metadata.get("expected_sha256")):
        raise BenchmarkContractError(
            "data.authoritative_metadata.expected_sha256 must be a SHA-256 digest"
        )
    release_reference = metadata.get("release_reference")
    if (
        not isinstance(release_reference, str)
        or not release_reference.strip()
        or (
            protocol["status"] == "frozen"
            and release_reference == "pending"
        )
    ):
        raise BenchmarkContractError(
            "data.authoritative_metadata.release_reference must be portable "
            "before protocol freeze"
        )
    _require_literal(metadata, "patient_column", "Reg. ID")
    _require_literal(metadata, "visit_column", "Coll. ID")
    _require_literal(metadata, "collection_date_column", "Date")
    _require_literal(metadata, "date_format", "%m/%d/%y")
    event_selection = _mapping(data, "event_selection")
    _require_literal(event_selection, "primary_requirement", "label_free")
    _require_literal(
        event_selection,
        "conditional_pregated_fallback_requires_claim_restriction",
        True,
    )
    _require_literal(event_selection, "aml_primary_source_level", "raw_fcs")
    _require_literal(event_selection, "aml_primary_rule", "technical_qc_only")
    _require_literal(
        event_selection,
        "aml_processed_role",
        "conditional_sensitivity_only",
    )
    _require_literal(
        event_selection,
        "nunez_local_row_selection",
        "all_public_pregated_rows",
    )
    _require_literal(
        event_selection,
        "evaluation_labels_for_event_selection",
        "forbidden",
    )

    split = _mapping(config, "split")
    _require_literal(split, "frozen_manifest_required", True)
    _require_literal(split, "dynamic_rebuild_forbidden", True)
    _require_literal(
        split,
        "sensitivity_patient_fold_policy",
        "preserve_primary_assignments",
    )
    _strict_int(split, "n_splits", minimum=2)
    _strict_int(split, "fold_seed", minimum=0)
    _strict_fraction(split, "validation_fraction")

    training = _mapping(config, "training")
    _require_literal(training, "label_conditioning", False)
    seeds = _sequence(training, "seeds")
    if (
        not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise BenchmarkContractError(
            "training.seeds must be non-empty, unique integers"
        )

    reference_bank = _mapping(config, "reference_bank")
    _require_literal(reference_bank, "primary_mode", "data_matched")
    _require_literal(reference_bank, "patient_set", "outer_training_target")
    _require_literal(reference_bank, "specimen_set", "all_eligible")
    _require_literal(
        reference_bank,
        "patient_weighting",
        "patient_then_equal_specimen",
    )
    _require_literal(reference_bank, "sampled_rows_shared_across_methods", True)
    _require_literal(reference_bank, "pairing_curve_holds_bank_fixed", True)
    _require_literal(reference_bank, "native_bank_deviation", "secondary_only")
    _require_literal(reference_bank, "record_actual_budget", True)
    fit_stages = _mapping(reference_bank, "fit_stages")
    if tuple(map(str, _sequence(fit_stages, "inner_fit"))) != ("train",):
        raise BenchmarkContractError(
            "reference_bank.fit_stages.inner_fit must be [train]"
        )
    if tuple(map(str, _sequence(fit_stages, "outer_refit"))) != (
        "train",
        "validation",
    ):
        raise BenchmarkContractError(
            "reference_bank.fit_stages.outer_refit must be [train, validation]"
        )
    bank_cap = _strict_int(
        reference_bank,
        "cells_per_specimen_cap",
        minimum=1,
    )
    if bank_cap != max_cells:
        raise BenchmarkContractError(
            "Reference-bank and sampling cell caps must match"
        )

    evaluation = _mapping(config, "evaluation")
    _require_literal(evaluation, "label_stratified_selection", False)
    _require_literal(evaluation, "require_exact_specimen_keys", True)
    _require_literal(
        evaluation,
        "primary_scale",
        "target_training_patient_balanced_iqr",
    )
    fallback = _sequence(evaluation, "zero_iqr_fallback")
    if fallback != (
        "target_training_patient_balanced_mad",
        "locked_minimum_scale",
    ):
        raise BenchmarkContractError("evaluation.zero_iqr_fallback is not locked")
    _strict_fraction(evaluation, "upper_quantile")
    _require_literal(
        evaluation, "split_half_interpretation", "finite_cell_sampling_floor"
    )
    required_tracks = set(map(str, _sequence(evaluation, "required_tracks")))
    expected_tracks = {"marginal", "joint", "individualization"}
    if not expected_tracks.issubset(required_tracks):
        missing = sorted(expected_tracks - required_tracks)
        raise BenchmarkContractError(
            f"evaluation.required_tracks is missing {missing}"
        )
    required_nulls = set(map(str, _sequence(evaluation, "required_nulls")))
    expected_nulls = {
        "global_target_median",
        "target_prior_sampler",
        "wrong_patient_reference",
    }
    if not expected_nulls.issubset(required_nulls):
        missing = sorted(expected_nulls - required_nulls)
        raise BenchmarkContractError(
            f"evaluation.required_nulls is missing {missing}"
        )

    endpoint_policy = _mapping(config, "endpoint_policy")
    _require_literal(
        endpoint_policy,
        "role",
        "secondary_clinical_illustration",
    )
    _require_literal(
        endpoint_policy,
        "exact_duplicate",
        "deduplicate_with_audit",
    )
    _require_literal(
        endpoint_policy,
        "single_source_annotation",
        "unknown_until_common_provenance",
    )
    _require_literal(
        endpoint_policy,
        "cross_source_conflict",
        "not_evaluable",
    )
    _require_literal(
        endpoint_policy,
        "longitudinal_conflict",
        "endpoint_specific_not_evaluable",
    )
    _require_literal(
        endpoint_policy,
        "ever_mutant_collapse",
        "forbidden",
    )
    _strict_int(
        endpoint_policy,
        "formal_support_min_total_per_class",
        minimum=1,
    )
    _strict_int(
        endpoint_policy,
        "formal_support_min_test_fold_per_class",
        minimum=1,
    )

    methods = _mapping(config, "methods")
    primary_values = _sequence(methods, "primary")
    if any(not isinstance(value, str) or not value for value in primary_values):
        raise BenchmarkContractError("methods.primary must contain method names")
    if len(set(primary_values)) != len(primary_values):
        raise BenchmarkContractError("methods.primary must be unique")
    primary = set(primary_values)
    forbidden = set(map(str, _sequence(methods, "forbidden_primary")))
    overlap = sorted(primary & forbidden)
    if overlap:
        raise BenchmarkContractError(
            f"Primary method list contains forbidden methods: {overlap}"
        )
    if "target_prior_sampler" not in primary:
        raise BenchmarkContractError(
            "Primary methods must include target_prior_sampler"
        )
    capability_violations = primary_method_violations(
        primary,
        require_implemented=protocol["status"] == "frozen",
    )
    if capability_violations:
        raise BenchmarkContractError(
            "Primary method registry violations: "
            f"{capability_violations}"
        )

    manifests = _mapping(config, "manifests")
    manifest_types = ("data", "markers", "splits", "endpoints", "banks", "stress")
    for key in manifest_types:
        value = manifests.get(key)
        if not isinstance(value, str) or not value.strip():
            raise BenchmarkContractError(f"manifests.{key} must be a path string")
    manifest_digests = _mapping(config, "manifest_digests")
    for key in manifest_types:
        value = manifest_digests.get(key)
        if protocol["status"] == "draft" and value == "pending":
            continue
        if not _is_sha256(value):
            raise BenchmarkContractError(
                f"manifest_digests.{key} must be a SHA-256 digest"
            )

    reporting = _mapping(config, "reporting")
    for key in (
        "record_git_state",
        "record_protocol_digest",
        "record_manifest_digests",
        "record_accessed_information",
        "record_sampled_row_indices",
        "record_method_failures",
    ):
        _require_literal(reporting, key, True)


def protocol_digest(config: Mapping) -> str:
    """Return a stable SHA-256 digest after validating the protocol."""

    validate_primary_benchmark_config(config)
    canonical = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
