#!/usr/bin/env python3
"""Prepare a shared uniform-cell cache for CyTOF exact-truth experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.aml import load_specimen_reservoir  # noqa: E402
from src.data.markers import build_pair_marker_manifest  # noqa: E402
from src.data.pseudo_panels import (  # noqa: E402
    build_two_sided_pseudo_panel_manifest,
)
from src.data.splits import (  # noqa: E402
    build_patient_grouped_manifest,
    discover_exact_specimen_pairs,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "AML")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cells-per-specimen", type=int, default=50_000)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=4207)
    return parser.parse_args()


def header(path: Path) -> tuple[str, ...]:
    return tuple(pd.read_csv(path, nrows=0).columns.astype(str))


def main() -> None:
    args = arguments()
    started = time.time()
    pairs = discover_exact_specimen_pairs(args.data_root, "spectral_flow", "cytof")
    split = build_patient_grouped_manifest(
        pairs,
        n_splits=5,
        seed=42,
        validation_fraction=0.15,
        pair_name="sf_cytof",
    )
    sf_markers = header(args.data_root / "spectral_flow" / "cells" / f"{pairs[0]}.csv")
    cytof_markers = header(args.data_root / "cytof" / "cells" / f"{pairs[0]}.csv")
    cross_panel = build_pair_marker_manifest(sf_markers, cytof_markers)
    common = cross_panel.target_common_columns
    exclusive = cross_panel.target_exclusive_columns
    # Biology-blind deterministic partition: alternate the immutable CyTOF
    # panel order after removing H. X receives the first marker when odd.
    source_exclusive = exclusive[::2]
    target_exclusive = exclusive[1::2]
    pseudo = build_two_sided_pseudo_panel_manifest(
        cytof_markers,
        common_markers=common,
        source_exclusive_markers=source_exclusive,
        target_exclusive_markers=target_exclusive,
        require_complete_partition=True,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    specimen_dir = args.output / "specimens"
    specimen_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    for index, specimen in enumerate(pairs):
        sampled = load_specimen_reservoir(
            args.data_root,
            "cytof",
            specimen,
            marker_columns=cytof_markers,
            maximum_cells=args.cells_per_specimen,
            chunk_size=args.chunk_size,
            random_state=args.seed + 1009 * index,
        )
        output = specimen_dir / f"{specimen}.npz"
        np.savez(
            output,
            values=np.asarray(sampled.values, dtype=np.float32),
            cell_types=np.asarray(sampled.cell_types, dtype=str),
            original_row_indices=np.asarray(sampled.original_row_indices, dtype=np.int64),
        )
        row_hash = hashlib.sha256(sampled.original_row_indices.tobytes()).hexdigest()
        records[specimen] = {
            "n_cells": int(sampled.values.shape[0]),
            "row_index_sha256": row_hash,
            "cache_file": str(output),
        }
        print(
            json.dumps(
                {
                    "event": "sampled",
                    "specimen": specimen,
                    "n_cells": sampled.values.shape[0],
                    "index": index + 1,
                    "total": len(pairs),
                }
            ),
            flush=True,
        )

    artifact = {
        "status": "ok",
        "host": socket.gethostname(),
        "data_root": str(args.data_root),
        "sampling": {
            "kind": "full_file_uniform_random_priority_reservoir",
            "cells_per_specimen": int(args.cells_per_specimen),
            "chunk_size": int(args.chunk_size),
            "seed": int(args.seed),
            "acquisition_order_biased": False,
        },
        "partition_rule": "alternate CyTOF panel order outside H; X gets first marker",
        "pseudo_panel": pseudo.to_dict(),
        "cross_panel_common_canonical": list(cross_panel.common_markers),
        "split_manifest": split,
        "specimens": records,
        "elapsed_seconds": time.time() - started,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "event": "complete",
                "manifest": str(manifest_path),
                "elapsed_seconds": artifact["elapsed_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
