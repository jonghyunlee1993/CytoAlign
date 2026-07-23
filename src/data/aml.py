"""Small, explicit readers for the processed AML specimen CSVs."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


COARSE_CELL_TYPES: tuple[str, ...] = (
    "Blast",
    "Monocyte",
    "T cell",
    "B cell",
    "NK cell",
)


def _coarsen_one(label: str) -> str | None:
    value = str(label).strip().lower()
    if value.startswith("blast"):
        return "Blast"
    if value.startswith("monocyte"):
        return "Monocyte"
    if value.startswith("t cell"):
        return "T cell"
    if value.startswith("b cell"):
        return "B cell"
    if value.startswith("nk cell"):
        return "NK cell"
    return None


def coarsen_cell_types(labels: Sequence[str]) -> np.ndarray:
    """Map fine AML gates to the five predeclared coarse categories."""
    return np.asarray([_coarsen_one(label) for label in labels], dtype=object)


def _label_column(frame: pd.DataFrame) -> str:
    for candidate in ("cell_type", "label", "event_type"):
        if candidate in frame.columns:
            return candidate
    if frame.shape[1] != 1:
        raise ValueError(
            "Could not identify a label column; expected cell_type, label, event_type, "
            "or a single-column label CSV"
        )
    return str(frame.columns[0])


@dataclass(frozen=True)
class SpecimenData:
    modality: str
    specimen_id: str
    markers: tuple[str, ...]
    values: np.ndarray
    cell_types: np.ndarray
    original_row_indices: np.ndarray


def load_specimen(
    data_root: str | Path,
    modality: str,
    specimen_id: str,
    marker_columns: Sequence[str] | None = None,
    *,
    drop_unmapped: bool = True,
    maximum_rows: int | None = None,
) -> SpecimenData:
    """Load one already-transformed AML specimen without further transforms.

    ``maximum_rows`` is an explicit head cap for fast pilot screens only. It
    keeps cell/label rows aligned but is not a replacement for the stratified
    full-file sampler required by final experiments.
    """

    if maximum_rows is not None and int(maximum_rows) < 1:
        raise ValueError("maximum_rows must be positive when provided")
    root = Path(data_root) / modality
    stem = Path(str(specimen_id)).stem
    cell_path = root / "cells" / f"{stem}.csv"
    label_path = root / "labels" / f"{stem}.csv"
    if not cell_path.exists() or not label_path.exists():
        raise FileNotFoundError(f"Missing cell or label CSV for {modality}/{stem}")
    header = tuple(pd.read_csv(cell_path, nrows=0).columns.astype(str))
    markers = header if marker_columns is None else tuple(map(str, marker_columns))
    missing = sorted(set(markers) - set(header))
    if missing:
        raise ValueError(f"{modality}/{stem} is missing markers: {missing}")
    # pandas preserves file order rather than ``usecols`` order, so select a
    # second time to enforce the immutable marker manifest.
    cells = pd.read_csv(
        cell_path,
        usecols=list(markers),
        nrows=None if maximum_rows is None else int(maximum_rows),
    )[list(markers)]
    label_frame = pd.read_csv(
        label_path, nrows=None if maximum_rows is None else int(maximum_rows)
    )
    if len(cells) != len(label_frame):
        raise ValueError(
            f"{modality}/{stem} cell/label row mismatch: {len(cells)} != {len(label_frame)}"
        )
    coarse = coarsen_cell_types(label_frame[_label_column(label_frame)].to_numpy())
    rows = np.arange(len(cells), dtype=np.int64)
    if drop_unmapped:
        keep = coarse != None  # noqa: E711 - intentional object-array comparison
        cells = cells.iloc[np.flatnonzero(keep)]
        coarse = coarse[keep]
        rows = rows[keep]
    return SpecimenData(
        modality=str(modality),
        specimen_id=stem,
        markers=markers,
        values=cells.to_numpy(dtype=np.float32, copy=True),
        cell_types=coarse,
        original_row_indices=rows,
    )


def load_specimen_reservoir(
    data_root: str | Path,
    modality: str,
    specimen_id: str,
    marker_columns: Sequence[str] | None = None,
    *,
    maximum_cells: int = 50_000,
    chunk_size: int = 100_000,
    random_state: int = 42,
    drop_unmapped: bool = True,
) -> SpecimenData:
    """Uniformly sample mapped cells while streaming the complete CSV pair.

    Every eligible row receives an independent random priority and the
    ``maximum_cells`` smallest priorities are retained. This is equivalent to
    reservoir sampling without replacement, avoids acquisition-order bias,
    and keeps the separately stored cell and label CSVs exactly aligned.
    """

    if int(maximum_cells) < 1:
        raise ValueError("maximum_cells must be positive")
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive")
    root = Path(data_root) / modality
    stem = Path(str(specimen_id)).stem
    cell_path = root / "cells" / f"{stem}.csv"
    label_path = root / "labels" / f"{stem}.csv"
    if not cell_path.exists() or not label_path.exists():
        raise FileNotFoundError(f"Missing cell or label CSV for {modality}/{stem}")
    header = tuple(pd.read_csv(cell_path, nrows=0).columns.astype(str))
    markers = header if marker_columns is None else tuple(map(str, marker_columns))
    missing = sorted(set(markers) - set(header))
    if missing:
        raise ValueError(f"{modality}/{stem} is missing markers: {missing}")

    cell_reader = pd.read_csv(
        cell_path, usecols=list(markers), chunksize=int(chunk_size)
    )
    label_reader = pd.read_csv(label_path, chunksize=int(chunk_size))
    rng = np.random.RandomState(int(random_state))
    held_values = np.empty((0, len(markers)), dtype=np.float32)
    held_labels = np.empty(0, dtype=object)
    held_rows = np.empty(0, dtype=np.int64)
    held_priorities = np.empty(0, dtype=np.float64)
    row_offset = 0

    for cells, labels in zip_longest(cell_reader, label_reader):
        if cells is None or labels is None:
            raise ValueError(f"{modality}/{stem} cell/label chunk count mismatch")
        if len(cells) != len(labels):
            raise ValueError(
                f"{modality}/{stem} cell/label row mismatch within streamed chunk: "
                f"{len(cells)} != {len(labels)}"
            )
        # pandas returns usecols in file order; restore the requested contract.
        current_values = cells[list(markers)].to_numpy(dtype=np.float32, copy=True)
        current_labels = coarsen_cell_types(labels[_label_column(labels)].to_numpy())
        current_rows = row_offset + np.arange(len(cells), dtype=np.int64)
        row_offset += len(cells)
        if drop_unmapped:
            keep = current_labels != None  # noqa: E711
            current_values = current_values[keep]
            current_labels = current_labels[keep]
            current_rows = current_rows[keep]
        if current_rows.size == 0:
            continue
        current_priorities = rng.random_sample(current_rows.size)
        held_values = np.concatenate([held_values, current_values], axis=0)
        held_labels = np.concatenate([held_labels, current_labels])
        held_rows = np.concatenate([held_rows, current_rows])
        held_priorities = np.concatenate([held_priorities, current_priorities])
        if held_rows.size > int(maximum_cells):
            selected = np.argpartition(
                held_priorities, int(maximum_cells) - 1
            )[: int(maximum_cells)]
            held_values = held_values[selected]
            held_labels = held_labels[selected]
            held_rows = held_rows[selected]
            held_priorities = held_priorities[selected]

    if held_rows.size == 0:
        raise ValueError(f"{modality}/{stem} has no eligible mapped cells")
    order = np.argsort(held_rows)
    return SpecimenData(
        modality=str(modality),
        specimen_id=stem,
        markers=markers,
        values=np.asarray(held_values[order], dtype=np.float32),
        cell_types=np.asarray(held_labels[order], dtype=object),
        original_row_indices=np.asarray(held_rows[order], dtype=np.int64),
    )
