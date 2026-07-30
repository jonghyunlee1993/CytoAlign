"""Code-owned capabilities for methods eligible for primary benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class MethodCapabilities:
    uses_cell_labels: bool
    uses_endpoint_labels: bool
    uses_test_target: bool
    output_mode: Literal["point", "distribution", "both"]
    implementation_status: Literal["implemented", "pending_adapter", "oracle_only"]


METHOD_REGISTRY: dict[str, MethodCapabilities] = {
    "global_target_median": MethodCapabilities(
        False, False, False, "point", "implemented"
    ),
    "target_prior_sampler": MethodCapabilities(
        False, False, False, "distribution", "implemented"
    ),
    "ridge": MethodCapabilities(False, False, False, "point", "implemented"),
    "knn50": MethodCapabilities(False, False, False, "both", "implemented"),
    "boosted_tree": MethodCapabilities(
        False, False, False, "point", "pending_adapter"
    ),
    "simple_mlp": MethodCapabilities(False, False, False, "point", "implemented"),
    "cytobackbone": MethodCapabilities(
        False, False, False, "distribution", "pending_adapter"
    ),
    "cycombine": MethodCapabilities(
        False, False, False, "distribution", "implemented"
    ),
    "cytovi": MethodCapabilities(False, False, False, "both", "implemented"),
    "uvae": MethodCapabilities(False, False, False, "both", "implemented"),
    "cell_type_median": MethodCapabilities(
        True, False, False, "point", "oracle_only"
    ),
}


def primary_method_violations(
    method_names: set[str],
    *,
    require_implemented: bool,
) -> list[str]:
    """Return unknown or forbidden-capability methods in stable order."""

    violations = []
    for name in sorted(method_names):
        capabilities = METHOD_REGISTRY.get(name)
        if capabilities is None:
            violations.append(f"{name}:unknown")
        elif (
            capabilities.uses_cell_labels
            or capabilities.uses_endpoint_labels
            or capabilities.uses_test_target
        ):
            violations.append(f"{name}:forbidden_information")
        elif require_implemented and capabilities.implementation_status != "implemented":
            violations.append(f"{name}:{capabilities.implementation_status}")
    return violations
