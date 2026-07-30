"""CLI for materializing the processed same-cell benchmark cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.benchmark.self_recoverability_cache import materialize_processed_cache
from src.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _absolute(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--modality",
        choices=("spectral_flow", "cytof"),
        required=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    split_path = _absolute(config["data"]["split_manifest"])
    split = json.loads(split_path.read_text(encoding="utf-8"))
    specimens = tuple(map(str, split["specimens"]))
    result = materialize_processed_cache(
        data_root=_absolute(config["data"]["root"]),
        modality=args.modality,
        specimens=specimens,
        row_index_root=_absolute(config["data"]["row_index_root"]),
        row_index_seed=int(config["data"]["row_index_seed"]),
        cache_root=_absolute(config["data"]["cache_root"]),
        overwrite=bool(args.overwrite),
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "modality": result["modality"],
                "specimens": len(result["specimens"]),
                "manifest_path": result["manifest_path"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

