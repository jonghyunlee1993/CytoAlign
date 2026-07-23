"""Predict target-exclusive markers for one source specimen."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.data.aml import load_specimen
from src.models.cytoalign import CytoAlign


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--specimen", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model = CytoAlign.load(args.model)
    columns = model.source_common_columns + model.source_exclusive_columns
    specimen = load_specimen(
        args.data_root,
        model.source_modality,
        args.specimen,
        columns,
    )
    n_common = len(model.source_common_columns)
    predictions = model.predict(
        specimen.values[:, :n_common],
        specimen.values[:, n_common:],
        specimen.cell_types,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        predictions=predictions,
        target_markers=np.asarray(model.target_markers),
        cell_types=specimen.cell_types.astype(str),
        source_row_indices=specimen.original_row_indices,
    )


if __name__ == "__main__":
    main()
