"""CLI for one literature marker-imputation baseline run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_config
from src.training.literature_baseline import run_literature_baseline


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_config_paths(config: dict) -> dict:
    for section, key in (
        ("data", "root"),
        ("data", "cache_root"),
        ("data", "split_manifest"),
        ("data", "row_index_root"),
        ("output", "root"),
    ):
        if key not in config.get(section, {}):
            continue
        value = Path(config[section][key])
        if not value.is_absolute():
            config[section][key] = str(PROJECT_ROOT / value)
    for method, keys in (
        ("cycombine", ("script", "rscript")),
        ("uvae", ("runner", "python", "external_root")),
    ):
        for key in keys:
            value = Path(config["methods"][method][key])
            if not value.is_absolute():
                config["methods"][method][key] = str(PROJECT_ROOT / value)
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=("cycombine", "cytovi", "uvae"),
        required=True,
    )
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
    result = run_literature_baseline(
        config,
        method=args.method,
        modality=args.modality,
        panel_name=args.panel,
        fold_index=args.fold,
        seed=args.seed,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
