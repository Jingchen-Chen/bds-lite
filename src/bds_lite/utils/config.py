from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    merged: dict[str, Any] = {}
    for parent in config.pop("extends", []) or []:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = (
                path.parent.parent.parent / parent
                if str(parent).startswith("configs/")
                else path.parent / parent
            )
        merged = deep_update(merged, load_config(parent_path))

    return deep_update(merged, config)
