"""Validate a recoverability benchmark protocol before expensive work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.benchmark.contract import (
    BenchmarkContractError,
    protocol_digest,
    validate_primary_benchmark_config,
)
from src.benchmark.manifest_validation import (
    validate_manifest_bundle,
    validate_manifest_index,
)
from src.config import load_config


def _find_repository_root(protocol_path: Path) -> Path:
    """Resolve repository-relative manifest paths without depending on CWD."""

    for candidate in (protocol_path.parent, *protocol_path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    cwd = Path.cwd().resolve()
    if (cwd / "pyproject.toml").is_file():
        return cwd
    return protocol_path.parent


def run_preflight(
    protocol_path: str | Path,
    *,
    mode: str = "draft",
) -> dict:
    """Validate one protocol; full mode is the mandatory execution gate."""

    path = Path(protocol_path).resolve()
    config = load_config(path)
    validate_primary_benchmark_config(config)
    if mode not in {"draft", "full"}:
        raise BenchmarkContractError("Preflight mode must be 'draft' or 'full'")
    status = str(config["protocol"]["status"])
    if mode == "full" and status != "frozen":
        raise BenchmarkContractError(
            f"Protocol must be frozen for a full run, got {status!r}"
        )

    repository_root = _find_repository_root(path)
    manifest_paths = {
        name: (repository_root / relative).resolve()
        if not Path(relative).is_absolute()
        else Path(relative)
        for name, relative in config["manifests"].items()
    }
    missing = sorted(
        name
        for name, manifest_path in manifest_paths.items()
        if not manifest_path.is_file() or manifest_path.stat().st_size == 0
    )
    pending_digests = sorted(
        name
        for name, digest in config["manifest_digests"].items()
        if digest == "pending"
    )
    if mode == "full" and (missing or pending_digests):
        raise BenchmarkContractError(
            "Full preflight requires non-empty frozen manifest files and digests: "
            f"missing={missing}, pending_digests={pending_digests}"
        )

    validated_manifests = {}
    for name, manifest_path in sorted(manifest_paths.items()):
        digest = config["manifest_digests"][name]
        if name in missing or digest == "pending":
            continue
        validated_manifests[name] = validate_manifest_index(
            manifest_path,
            expected_type=name,
            expected_protocol_id=str(config["protocol"]["id"]),
            expected_index_digest=digest,
        )
    bundle_validated = len(validated_manifests) == len(manifest_paths)
    if bundle_validated:
        validate_manifest_bundle(
            validated_manifests,
            n_splits=int(config["split"]["n_splits"]),
        )

    ready = status == "frozen" and bundle_validated and not missing and not pending_digests
    return {
        "status": "ok",
        "mode": mode,
        "protocol_path": str(path),
        "protocol_id": str(config["protocol"]["id"]),
        "protocol_version": int(config["protocol"]["version"]),
        "protocol_status": status,
        "protocol_digest": protocol_digest(config),
        "repository_root": str(repository_root),
        "manifest_paths": {
            name: str(manifest_path)
            for name, manifest_path in sorted(manifest_paths.items())
        },
        "missing_manifest_paths": missing,
        "pending_manifest_digests": pending_digests,
        "validated_manifest_types": sorted(validated_manifests),
        "manifest_bundle_validated": bundle_validated,
        "ready_for_full_run": ready,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="configs/benchmark/protocol_v1.yaml",
        help="Machine-readable benchmark protocol",
    )
    parser.add_argument(
        "--mode",
        choices=("draft", "full"),
        default="draft",
        help="Full mode requires frozen, content-validated manifests",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_preflight(
                args.protocol,
                mode=args.mode,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
