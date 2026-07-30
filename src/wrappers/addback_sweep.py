"""CLI for one modality/fold clinical10-to-H19 add-back sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_config
from src.training.addback_sweep import run_addback_sweep
from src.wrappers.self_recoverability import _resolve_config_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--modality",
        choices=("spectral_flow", "cytof"),
        required=True,
    )
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=4207)
    args = parser.parse_args()
    config = _resolve_config_paths(load_config(args.config))
    result = run_addback_sweep(
        config,
        modality=args.modality,
        fold_index=args.fold,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "modality": result["modality"],
                "fold": result["fold"],
                "seed": result["seed"],
                "elapsed_seconds": result["elapsed_seconds"],
                "hardware": result["hardware"],
                "artifacts": result["artifacts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
