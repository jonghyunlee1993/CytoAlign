"""CLI for patient-first same-cell benchmark aggregation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.self_recoverability_summary import (
    summarize_self_recoverability,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/aml_same_cell_recoverability_v0"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=4207)
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else PROJECT_ROOT / args.root
    result = summarize_self_recoverability(
        root,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
