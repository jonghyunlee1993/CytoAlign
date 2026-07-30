"""Deterministic, label-free reference-row reservoir materialization."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np


class ReferenceRowError(ValueError):
    """Raised when a reference-row artifact violates its declared contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_reference_rows(
    path: Path,
    *,
    eligible_events: int,
    selected_events: int,
    row_seed: int | None = None,
) -> str:
    """Validate structure and, when supplied, exact deterministic content."""

    try:
        rows = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ReferenceRowError(
            f"Cannot load reference-row artifact: {path}"
        ) from error
    if rows.dtype != np.dtype("uint32") or rows.ndim != 1:
        raise ReferenceRowError(
            f"Reference rows must be one-dimensional uint32: {path}"
        )
    if rows.size != int(selected_events):
        raise ReferenceRowError(
            f"Reference-row count differs from contract: {path}"
        )
    if rows.size:
        if int(rows[-1]) >= int(eligible_events):
            raise ReferenceRowError(
                f"Reference row is outside expression bounds: {path}"
            )
        if rows.size > 1 and not bool(np.all(rows[1:] > rows[:-1])):
            raise ReferenceRowError(
                f"Reference rows must be unique and strictly sorted: {path}"
            )
    if row_seed is not None:
        generator = np.random.Generator(np.random.PCG64(int(row_seed)))
        expected = generator.choice(
            int(eligible_events),
            size=int(selected_events),
            replace=False,
        )
        expected = np.sort(expected.astype(np.uint32, copy=False))
        if not np.array_equal(rows, expected):
            raise ReferenceRowError(
                f"Reference rows do not match their deterministic seed: {path}"
            )
    return sha256_file(path)


def materialize_reference_row_indices(
    audit: dict,
    output_root: str | Path,
) -> dict:
    """Materialize each modality/specimen/seed reservoir exactly once.

    The function consumes only the expression-derived eligible event count and
    precomputed deterministic seed. Evaluation labels are never opened.
    Duplicate bank memberships are rewired to the same physical row-index file.
    """

    inventory = audit["summary"]["inventory"]
    if not inventory.get("source_hashes_complete"):
        raise ReferenceRowError(
            "Source expression hashes must be complete before row materialization"
        )
    output = Path(output_root)
    row_root = output / "reference_rows"
    row_root.mkdir(parents=True, exist_ok=True)
    data_by_key: dict[tuple[str, str], Mapping[str, object]] = {
        (str(row["modality"]), str(row["specimen_id"])): row
        for row in audit["data_records"]
        if str(row["excluded"]) == "false"
    }
    unique_contracts: dict[
        tuple[str, str, int], tuple[int, int, int]
    ] = {}
    for record in audit["bank_records"]:
        key = (
            str(record["modality"]),
            str(record["specimen_id"]),
            int(record["seed"]),
        )
        data_row = data_by_key.get(key[:2])
        if data_row is None:
            raise ReferenceRowError(
                f"Bank row has no eligible data record: {key[:2]}"
            )
        contract = (
            int(data_row["eligible_events"]),
            int(record["event_count"]),
            int(record["row_index_seed"]),
        )
        previous = unique_contracts.setdefault(key, contract)
        if previous != contract:
            raise ReferenceRowError(
                f"Inconsistent duplicate reservoir contract: {key}"
            )

    artifact_by_key = {}
    reused = 0
    created = 0
    total_selected_rows = 0
    for key, (eligible_events, selected_events, row_seed) in sorted(
        unique_contracts.items()
    ):
        modality, specimen, seed = key
        path = row_root / modality / specimen / f"seed_{seed}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        valid_existing = False
        if path.is_file():
            try:
                digest = validate_reference_rows(
                    path,
                    eligible_events=eligible_events,
                    selected_events=selected_events,
                    row_seed=row_seed,
                )
                valid_existing = True
            except ReferenceRowError:
                valid_existing = False
        if valid_existing:
            reused += 1
        else:
            generator = np.random.Generator(np.random.PCG64(row_seed))
            rows = generator.choice(
                eligible_events,
                size=selected_events,
                replace=False,
            )
            rows = np.sort(rows.astype(np.uint32, copy=False))
            np.save(path, rows, allow_pickle=False)
            digest = validate_reference_rows(
                path,
                eligible_events=eligible_events,
                selected_events=selected_events,
                row_seed=row_seed,
            )
            created += 1
        artifact_by_key[key] = {
            "path": path,
            "sha256": digest,
        }
        total_selected_rows += selected_events

    for record in audit["bank_records"]:
        key = (
            str(record["modality"]),
            str(record["specimen_id"]),
            int(record["seed"]),
        )
        artifact = artifact_by_key[key]
        record["row_index_file"] = artifact["path"].relative_to(output).as_posix()
        record["row_index_sha256"] = artifact["sha256"]
        record["status"] = "materialized_validated"

    blockers = audit["summary"].get("unresolved_blockers", [])
    audit["summary"]["unresolved_blockers"] = [
        blocker
        for blocker in blockers
        if blocker != "reference-bank row indices and digests pending"
    ]
    audit["summary"]["reference_row_materialization"] = {
        "unique_artifacts": len(unique_contracts),
        "created_artifacts": created,
        "reused_artifacts": reused,
        "total_selected_rows_across_unique_artifacts": total_selected_rows,
        "root": row_root.as_posix(),
        "status": "materialized_validated",
    }
    return audit["summary"]["reference_row_materialization"]
