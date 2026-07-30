"""CLI for one same-cell marker-recoverability run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_config
from src.training.self_recoverability import run_self_recoverability


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_config_paths(config: dict) -> dict:
    for section, key in (
        ("data", "root"),
        ("data", "cache_root"),
        ("data", "split_manifest"),
        ("data", "row_index_root"),
        ("output", "root"),
    ):
        value = Path(config[section][key])
        if not value.is_absolute():
            config[section][key] = str(PROJECT_ROOT / value)
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--modality",
        choices=("spectral_flow", "cytof"),
        required=True,
    )
    parser.add_argument("--panel", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=4207)
    args = parser.parse_args()
    config = _resolve_config_paths(load_config(args.config))
    result = run_self_recoverability(
        config,
        modality=args.modality,
        panel_name=args.panel,
        fold_index=args.fold,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "modality": result["modality"],
                "panel": result["panel"],
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

