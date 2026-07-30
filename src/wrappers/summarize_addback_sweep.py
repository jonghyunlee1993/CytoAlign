"""CLI for patient-paired add-back sweep aggregation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.addback_summary import summarize_addback_sweep


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/aml_h19_addback_screen_v0"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=4207)
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else PROJECT_ROOT / args.root
    result = summarize_addback_sweep(
        root,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
