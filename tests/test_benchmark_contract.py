import copy

import pytest

from src.benchmark.contract import (
    BenchmarkContractError,
    protocol_digest,
    validate_primary_benchmark_config,
)
from src.config import load_config


@pytest.fixture
def protocol():
    return load_config("configs/benchmark/protocol_v1.yaml")


def test_repository_benchmark_protocol_is_valid_and_stable(protocol):
    validate_primary_benchmark_config(protocol)
    digest = protocol_digest(protocol)
    assert len(digest) == 64
    assert digest == protocol_digest(copy.deepcopy(protocol))


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("training", "label_conditioning"), True, "label_conditioning"),
        (
            ("evaluation", "label_stratified_selection"),
            True,
            "label_stratified_selection",
        ),
        (("data", "sampling"), "head", "sampling"),
        (
            ("information_access", "test_target_statistics"),
            "allowed",
            "test_target_statistics",
        ),
        (("split", "dynamic_rebuild_forbidden"), False, "dynamic_rebuild"),
        (
            ("reference_bank", "sampled_rows_shared_across_methods"),
            False,
            "sampled_rows_shared",
        ),
        (
            ("evaluation", "primary_scale"),
            "test_target_iqr",
            "primary_scale",
        ),
    ],
)
def test_primary_contract_fails_closed_on_leakage_prone_settings(
    protocol, path, value, match
):
    invalid = copy.deepcopy(protocol)
    invalid[path[0]][path[1]] = value
    with pytest.raises(BenchmarkContractError, match=match):
        validate_primary_benchmark_config(invalid)


def test_primary_contract_rejects_label_informed_method(protocol):
    invalid = copy.deepcopy(protocol)
    invalid["methods"]["primary"].append("cell_type_median")
    with pytest.raises(BenchmarkContractError, match="forbidden"):
        validate_primary_benchmark_config(invalid)


def test_registry_rejects_label_method_even_if_yaml_forbidden_list_is_removed(
    protocol,
):
    invalid = copy.deepcopy(protocol)
    invalid["methods"]["forbidden_primary"] = []
    invalid["methods"]["primary"].append("cell_type_median")
    with pytest.raises(BenchmarkContractError, match="registry violations"):
        validate_primary_benchmark_config(invalid)


def test_primary_contract_rejects_unknown_method_and_numeric_boolean(protocol):
    unknown = copy.deepcopy(protocol)
    unknown["methods"]["primary"].append("renamed_label_model")
    with pytest.raises(BenchmarkContractError, match="unknown"):
        validate_primary_benchmark_config(unknown)

    numeric_boolean = copy.deepcopy(protocol)
    numeric_boolean["training"]["label_conditioning"] = 0
    with pytest.raises(BenchmarkContractError, match="label_conditioning"):
        validate_primary_benchmark_config(numeric_boolean)


def test_frozen_protocol_requires_concrete_manifest_digests(protocol):
    invalid = copy.deepcopy(protocol)
    invalid["protocol"]["status"] = "frozen"
    invalid["data"]["authoritative_metadata"]["release_reference"] = (
        "fixture://metadata"
    )
    invalid["methods"]["primary"] = [
        "global_target_median",
        "target_prior_sampler",
        "ridge",
        "knn50",
        "simple_mlp",
    ]
    with pytest.raises(BenchmarkContractError, match="manifest_digests"):
        validate_primary_benchmark_config(invalid)


def test_frozen_protocol_requires_portable_metadata_reference(protocol):
    invalid = copy.deepcopy(protocol)
    invalid["protocol"]["status"] = "frozen"
    with pytest.raises(BenchmarkContractError, match="portable"):
        validate_primary_benchmark_config(invalid)


def test_frozen_protocol_rejects_pending_method_adapter(protocol):
    invalid = copy.deepcopy(protocol)
    invalid["protocol"]["status"] = "frozen"
    invalid["data"]["authoritative_metadata"]["release_reference"] = (
        "fixture://metadata"
    )
    with pytest.raises(BenchmarkContractError, match="pending_adapter"):
        validate_primary_benchmark_config(invalid)


def test_digest_changes_when_scientific_setting_changes(protocol):
    changed = copy.deepcopy(protocol)
    changed["training"]["seeds"] = [4207]
    assert protocol_digest(changed) != protocol_digest(protocol)
