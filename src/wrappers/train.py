"""CLI for one end-to-end CytoAlign fold and seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_config
from src.training import run_experiment


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.fold is not None:
        config["experiment"]["fold"] = args.fold
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed
    for section, key in (("data", "root"), ("output", "root")):
        path = Path(config[section][key])
        if not path.is_absolute():
            config[section][key] = str(PROJECT_ROOT / path)
    result = run_experiment(config)
    summary = {
        "status": result["status"],
        "experiment": result["experiment"],
        "fold": result["fold"],
        "seed": result["seed"],
        "residual_baseline": result["residual_baseline"],
    }
    if "methods" in result:
        summary["model"] = result["model"]
        summary["test_wasserstein"] = {
            name: method["test"]["patient_first_normalized_wasserstein"]
            for name, method in result["methods"].items()
        }
    else:
        summary["paired_curve"] = {
            count: {
                "alpha": point["methods"]["cytoalign"]["selected_alpha"],
                "cytoalign_wasserstein": point["methods"]["cytoalign"]["test"][
                    "patient_first_normalized_wasserstein"
                ],
                "ot_hl_wasserstein": point["methods"]["ot_hl"]["test"][
                    "patient_first_normalized_wasserstein"
                ],
            }
            for count, point in result["paired_curve"].items()
        }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
