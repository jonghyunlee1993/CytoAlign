"""CLI for the literature-plus-inductive marker-imputation comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.literature_baseline_summary import (
    summarize_literature_baselines,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--literature-root",
        type=Path,
        default=Path("outputs/aml_literature_baselines_v0"),
    )
    parser.add_argument(
        "--same-cell-root",
        type=Path,
        default=Path("outputs/aml_same_cell_recoverability_v0"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=4207)
    args = parser.parse_args()
    literature_root = (
        args.literature_root
        if args.literature_root.is_absolute()
        else PROJECT_ROOT / args.literature_root
    )
    same_cell_root = (
        args.same_cell_root
        if args.same_cell_root.is_absolute()
        else PROJECT_ROOT / args.same_cell_root
    )
    result = summarize_literature_baselines(
        literature_root,
        same_cell_root,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
