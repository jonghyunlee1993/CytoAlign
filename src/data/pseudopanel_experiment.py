"""Shared cache and block helpers for pseudo-panel OT experiments."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src.preprocessing.common_space import EmpiricalPercentileTransformer


def marker_scale(values: np.ndarray) -> np.ndarray:
    """Robust per-marker scale used by normalized evaluation losses."""

    values = np.asarray(values)
    iqr = np.quantile(values, 0.75, axis=0) - np.quantile(
        values, 0.25, axis=0
    )
    standard = np.std(values, axis=0)
    return np.maximum.reduce(
        [iqr, 0.1 * standard, np.full(iqr.shape, 1.0e-3)]
    )


def indices(markers: Sequence[str], selected: Sequence[str]) -> np.ndarray:
    """Return immutable marker-order indices."""

    lookup = {marker: index for index, marker in enumerate(markers)}
    return np.asarray(
        [lookup[str(marker)] for marker in selected],
        dtype=np.int64,
    )


def load_cache(cache: Path) -> tuple[dict, dict[str, dict]]:
    """Load a cache produced by ``prepare_pseudopanel_cache.py``."""

    manifest = json.loads((cache / "manifest.json").read_text())
    if manifest.get("status") != "ok":
        raise ValueError("Cache manifest is not complete")
    panels = {}
    for specimen, record in manifest["specimens"].items():
        path = Path(record["cache_file"])
        if not path.is_absolute():
            path = cache / path
        with np.load(path, allow_pickle=False) as archive:
            values = np.asarray(archive["values"], dtype=np.float32)
            labels = np.asarray(archive["cell_types"]).astype(str)
            rows = np.asarray(
                archive["original_row_indices"],
                dtype=np.int64,
            )
        if values.shape[0] != labels.size or labels.size != rows.size:
            raise ValueError(f"Misaligned cache arrays for {specimen}")
        panels[specimen] = {
            "values": values,
            "labels": labels,
            "rows": rows,
        }
    return manifest, panels


@dataclass(frozen=True)
class SampledBlock:
    specimen: str
    cell_type: str
    source_indices: np.ndarray
    target_indices: np.ndarray


@dataclass(frozen=True)
class PreparedBlock:
    specimen: str
    cell_type: str
    source_h: np.ndarray
    source_x: np.ndarray
    source_y: np.ndarray
    target_h: np.ndarray
    target_y: np.ndarray


def sample_blocks(
    panels: Mapping[str, dict],
    specimens: Sequence[str],
    *,
    k_max: int,
    k_min: int,
    seed: int,
) -> tuple[SampledBlock, ...]:
    """Split each specimen×cell-type stratum into disjoint source/target rows."""

    rng = np.random.RandomState(int(seed))
    blocks = []
    for specimen in sorted(specimens):
        labels = np.asarray(panels[specimen]["labels"]).astype(str)
        for cell_type in sorted(set(labels)):
            rows = np.flatnonzero(labels == cell_type)
            k = min(int(k_max), rows.size // 2)
            if k < int(k_min):
                continue
            selected = rng.permutation(rows)[: 2 * k]
            blocks.append(
                SampledBlock(
                    specimen=str(specimen),
                    cell_type=str(cell_type),
                    source_indices=np.sort(selected[:k]),
                    target_indices=np.sort(selected[k:]),
                )
            )
    if len(blocks) < 2:
        raise ValueError("Fewer than two eligible specimen×cell-type blocks")
    return tuple(blocks)


def prepare_blocks(
    blocks: Sequence[SampledBlock],
    target_order: np.ndarray,
    panels: Mapping[str, dict],
    transformer: EmpiricalPercentileTransformer,
    h_index: np.ndarray,
    x_index: np.ndarray,
    y_index: np.ndarray,
) -> tuple[PreparedBlock, ...]:
    """Materialize source blocks and an explicit same-type target order."""

    prepared = []
    for source_index, source_block in enumerate(blocks):
        target_block = blocks[int(target_order[source_index])]
        if source_block.cell_type != target_block.cell_type:
            raise RuntimeError("OT target order crosses cell types")
        source_values = panels[source_block.specimen]["values"]
        target_values = panels[target_block.specimen]["values"]
        prepared.append(
            PreparedBlock(
                specimen=source_block.specimen,
                cell_type=source_block.cell_type,
                source_h=transformer.transform(
                    source_values[source_block.source_indices][:, h_index]
                ),
                source_x=np.asarray(
                    source_values[source_block.source_indices][:, x_index],
                    dtype=np.float32,
                ),
                source_y=np.asarray(
                    source_values[source_block.source_indices][:, y_index],
                    dtype=np.float32,
                ),
                target_h=transformer.transform(
                    target_values[target_block.target_indices][:, h_index]
                ),
                target_y=np.asarray(
                    target_values[target_block.target_indices][:, y_index],
                    dtype=np.float32,
                ),
            )
        )
    return tuple(prepared)


def prepare_pooled_blocks(
    blocks: Sequence[SampledBlock],
    panels: Mapping[str, dict],
    transformer: EmpiricalPercentileTransformer,
    h_index: np.ndarray,
    x_index: np.ndarray,
    y_index: np.ndarray,
    *,
    seed: int,
) -> tuple[PreparedBlock, ...]:
    """Use a same-type target pool drawn only from other specimens."""

    rng = np.random.RandomState(int(seed))
    by_type: dict[str, list[SampledBlock]] = defaultdict(list)
    for block in blocks:
        by_type[block.cell_type].append(block)
    prepared = []
    for source_block in blocks:
        target_blocks = [
            block
            for block in by_type[source_block.cell_type]
            if block.specimen != source_block.specimen
        ]
        if not target_blocks:
            raise ValueError(
                f"No cross-specimen target pool for {source_block.cell_type}"
            )
        target_h_raw = np.concatenate(
            [
                panels[block.specimen]["values"][block.target_indices][
                    :, h_index
                ]
                for block in target_blocks
            ]
        )
        target_y_all = np.concatenate(
            [
                panels[block.specimen]["values"][block.target_indices][
                    :, y_index
                ]
                for block in target_blocks
            ]
        )
        k = source_block.source_indices.size
        selected = rng.choice(
            target_h_raw.shape[0],
            size=k,
            replace=target_h_raw.shape[0] < k,
        )
        source_values = panels[source_block.specimen]["values"]
        prepared.append(
            PreparedBlock(
                specimen=source_block.specimen,
                cell_type=source_block.cell_type,
                source_h=transformer.transform(
                    source_values[source_block.source_indices][:, h_index]
                ),
                source_x=np.asarray(
                    source_values[source_block.source_indices][:, x_index],
                    dtype=np.float32,
                ),
                source_y=np.asarray(
                    source_values[source_block.source_indices][:, y_index],
                    dtype=np.float32,
                ),
                target_h=transformer.transform(target_h_raw[selected]),
                target_y=np.asarray(
                    target_y_all[selected],
                    dtype=np.float32,
                ),
            )
        )
    return tuple(prepared)
