"""CLI for fine/coarse cell-type probes of translated cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_config
from src.training.cell_type_probe import (
    run_cell_type_probe_fold,
    summarize_cell_type_probe,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.fold is not None:
        config["experiment"]["fold"] = int(args.fold)
    if args.seed is not None:
        config["experiment"]["seed"] = int(args.seed)
    for section, key in (("data", "root"), ("output", "root")):
        path = Path(config[section][key])
        if not path.is_absolute():
            config[section][key] = str(PROJECT_ROOT / path)

    if args.summarize:
        result = summarize_cell_type_probe(
            Path(config["output"]["root"]) / config["experiment"]["name"],
            expected_folds=int(config["data"]["split"]["n_splits"]),
            bootstrap_replicates=int(config["probe"]["bootstrap_replicates"]),
            seed=int(config["experiment"]["seed"]),
        )
        print(json.dumps(result, sort_keys=True))
        return

    result = run_cell_type_probe_fold(config)
    print(
        json.dumps(
            {
                "status": result["status"],
                "fold": result["fold"],
                "seed": result["seed"],
                "result_path": result["result_path"],
                "elapsed_seconds": result["elapsed_seconds"],
                "hardware": result["hardware"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
