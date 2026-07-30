"""Materialize frozen processed-AML rows for repeated self-recoverability runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.data.markers import DEFAULT_TECHNICAL_MARKERS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _technical_token(marker: str) -> str:
    return str(marker).strip().upper().replace(" ", "")


def biological_markers(header: Sequence[str]) -> tuple[str, ...]:
    technical = {_technical_token(marker) for marker in DEFAULT_TECHNICAL_MARKERS}
    return tuple(
        str(marker)
        for marker in header
        if _technical_token(marker) not in technical
    )


def _label_column(frame: pd.DataFrame) -> str:
    for candidate in ("cell_type", "label", "event_type"):
        if candidate in frame.columns:
            return candidate
    if frame.shape[1] != 1:
        raise ValueError("Could not identify the cell-label column")
    return str(frame.columns[0])


def _write_npz_atomic(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def materialize_processed_cache(
    *,
    data_root: str | Path,
    modality: str,
    specimens: Sequence[str],
    row_index_root: str | Path,
    row_index_seed: int,
    cache_root: str | Path,
    overwrite: bool = False,
) -> dict:
    """Create one compact, row-aligned NPZ per specimen.

    Row selection is defined exclusively by an existing frozen ``.npy`` file.
    Labels are retained for downstream evaluation but never affect row
    inclusion.
    """

    data_path = Path(data_root)
    row_root = Path(row_index_root)
    output_root = Path(cache_root) / str(modality)
    output_root.mkdir(parents=True, exist_ok=True)
    specimens = tuple(map(str, specimens))
    if not specimens:
        raise ValueError("No specimens were supplied")

    first_cells = data_path / modality / "cells" / f"{specimens[0]}.csv"
    header = tuple(pd.read_csv(first_cells, nrows=0).columns.astype(str))
    markers = biological_markers(header)
    records = []
    for specimen in specimens:
        output = output_root / f"{specimen}.npz"
        cells_path = data_path / modality / "cells" / f"{specimen}.csv"
        labels_path = data_path / modality / "labels" / f"{specimen}.csv"
        rows_path = (
            row_root / modality / specimen / f"seed_{int(row_index_seed)}.npy"
        )
        for required in (cells_path, labels_path, rows_path):
            if not required.exists():
                raise FileNotFoundError(required)

        if output.exists() and not overwrite:
            with np.load(output, allow_pickle=False) as cached:
                if tuple(cached["markers"].astype(str)) != markers:
                    raise ValueError(f"Cached marker mismatch for {modality}/{specimen}")
                records.append(
                    {
                        "specimen": specimen,
                        "rows": int(cached["values"].shape[0]),
                        "cache_path": str(output),
                        "cache_sha256": _sha256(output),
                        "status": "reused",
                    }
                )
            continue

        current_header = tuple(
            pd.read_csv(cells_path, nrows=0).columns.astype(str)
        )
        if current_header != header:
            raise ValueError(f"Header mismatch for {modality}/{specimen}")
        row_indices = np.load(rows_path, allow_pickle=False).astype(np.int64)
        if (
            row_indices.ndim != 1
            or row_indices.size == 0
            or np.any(row_indices < 0)
            or len(np.unique(row_indices)) != len(row_indices)
        ):
            raise ValueError(f"Invalid row-index artifact for {modality}/{specimen}")

        cells = pd.read_csv(cells_path, usecols=list(markers))[list(markers)]
        label_frame = pd.read_csv(labels_path)
        if len(cells) != len(label_frame) or row_indices.max() >= len(cells):
            raise ValueError(f"Row alignment failure for {modality}/{specimen}")
        labels = (
            label_frame[_label_column(label_frame)]
            .astype(str)
            .to_numpy()[row_indices]
        )
        values = cells.iloc[row_indices].to_numpy(dtype=np.float32, copy=True)
        _write_npz_atomic(
            output,
            values=values,
            labels=np.asarray(labels, dtype=str),
            row_indices=row_indices.astype(np.uint32),
            markers=np.asarray(markers, dtype=str),
        )
        records.append(
            {
                "specimen": specimen,
                "rows": int(values.shape[0]),
                "cache_path": str(output),
                "cache_sha256": _sha256(output),
                "status": "created",
            }
        )

    manifest = {
        "modality": str(modality),
        "source": "processed_transformed_upstream_pregated",
        "claim_scope": "conditional_sensitivity_only",
        "row_selection": "frozen_row_indices_without_runtime_label_filter",
        "row_index_seed": int(row_index_seed),
        "markers": list(markers),
        "specimens": list(specimens),
        "records": records,
    }
    manifest_path = output_root / "cache_manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def load_cached_specimen(
    cache_root: str | Path,
    modality: str,
    specimen: str,
) -> dict:
    path = Path(cache_root) / str(modality) / f"{specimen}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as cached:
        return {
            "values": np.asarray(cached["values"], dtype=np.float32),
            "labels": cached["labels"].astype(str),
            "row_indices": cached["row_indices"].astype(np.int64),
            "markers": tuple(cached["markers"].astype(str)),
            "cache_path": str(path),
        }
