"""Small YAML configuration loader with single-file inheritance."""

from __future__ import annotations

from pathlib import Path

import yaml


def _merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict:
    """Load YAML and recursively merge its optional ``extends`` parent."""

    config_path = Path(path).resolve()
    current = yaml.safe_load(config_path.read_text()) or {}
    parent = current.pop("extends", None)
    if parent is None:
        return current
    parent_path = (config_path.parent / parent).resolve()
    return _merge(load_config(parent_path), current)
