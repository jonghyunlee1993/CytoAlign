"""Paired, cell-unpaired AML cross-panel dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.aml import SpecimenData, load_specimen, load_specimen_reservoir
from src.data.markers import (
    DEFAULT_TECHNICAL_MARKERS,
    PairMarkerManifest,
    build_pair_marker_manifest,
)
from src.data.splits import (
    build_patient_grouped_manifest,
    discover_exact_specimen_pairs,
)


@dataclass(frozen=True)
class CrossPanelDataset:
    source_modality: str
    target_modality: str
    common_markers: tuple[str, ...]
    source_common_columns: tuple[str, ...]
    target_common_columns: tuple[str, ...]
    source_exclusive_columns: tuple[str, ...]
    target_exclusive_columns: tuple[str, ...]
    source: dict[str, SpecimenData]
    target: dict[str, SpecimenData]
    splits: dict


def _header(path: Path) -> tuple[str, ...]:
    return tuple(pd.read_csv(path, nrows=0).columns.astype(str))


def _biological_source_columns(manifest: PairMarkerManifest) -> tuple[str, ...]:
    technical = {
        marker.upper().replace(" ", "") for marker in DEFAULT_TECHNICAL_MARKERS
    }
    return tuple(
        marker
        for marker in manifest.source_exclusive_columns
        if marker.upper().replace(" ", "") not in technical
    )


def load_cross_panel_dataset(config: dict) -> CrossPanelDataset:
    """Load uniformly sampled paired specimens from two AML modalities."""

    root = Path(config["root"])
    source_modality = str(config["source_modality"])
    target_modality = str(config["target_modality"])
    specimens = discover_exact_specimen_pairs(root, source_modality, target_modality)
    if not specimens:
        raise ValueError("No exact specimen pairs were found")

    source_header = _header(root / source_modality / "cells" / f"{specimens[0]}.csv")
    target_header = _header(root / target_modality / "cells" / f"{specimens[0]}.csv")
    marker_manifest = build_pair_marker_manifest(source_header, target_header)
    source_exclusive = _biological_source_columns(marker_manifest)
    target_exclusive = marker_manifest.target_primary_exclusive_columns
    if not source_exclusive or not target_exclusive:
        raise ValueError("Both panels need biological exclusive markers")

    maximum = int(config["max_cells_per_specimen"])
    chunk_size = int(config["chunk_size"])
    seed = int(config["sample_seed"])
    sampling = str(config.get("sampling", "reservoir"))
    source_columns = marker_manifest.source_common_columns + source_exclusive
    target_columns = marker_manifest.target_common_columns + target_exclusive
    source = {}
    target = {}
    for index, specimen in enumerate(specimens):
        if sampling == "head":
            source[specimen] = load_specimen(
                root,
                source_modality,
                specimen,
                source_columns,
                maximum_rows=maximum,
            )
            target[specimen] = load_specimen(
                root,
                target_modality,
                specimen,
                target_columns,
                maximum_rows=maximum,
            )
        else:
            source[specimen] = load_specimen_reservoir(
                root,
                source_modality,
                specimen,
                source_columns,
                maximum_cells=maximum,
                chunk_size=chunk_size,
                random_state=seed + 2 * index,
            )
            target[specimen] = load_specimen_reservoir(
                root,
                target_modality,
                specimen,
                target_columns,
                maximum_cells=maximum,
                chunk_size=chunk_size,
                random_state=seed + 2 * index + 1,
            )

    split_config = config["split"]
    splits = build_patient_grouped_manifest(
        specimens,
        n_splits=int(split_config["n_splits"]),
        seed=int(split_config["seed"]),
        validation_fraction=float(split_config["validation_fraction"]),
        pair_name=f"{source_modality}_to_{target_modality}",
    )
    return CrossPanelDataset(
        source_modality=source_modality,
        target_modality=target_modality,
        common_markers=marker_manifest.common_markers,
        source_common_columns=marker_manifest.source_common_columns,
        target_common_columns=marker_manifest.target_common_columns,
        source_exclusive_columns=source_exclusive,
        target_exclusive_columns=target_exclusive,
        source=source,
        target=target,
        splits=splits,
    )
