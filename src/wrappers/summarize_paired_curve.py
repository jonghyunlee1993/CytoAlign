"""CLI for paired-specimen dose-response aggregation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.paired_curve import summarize_paired_curve


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs")
    args = parser.parse_args()
    result = summarize_paired_curve(args.output_root / args.experiment)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
