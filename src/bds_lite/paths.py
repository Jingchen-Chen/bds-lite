"""Resolve paths of the original revision layout in the current repository layout.

Frozen protocols, configurations and run queues cite files by the paths they had when they were
frozen (for example ``revision/ABLATION_PROTOCOL.md``). The repository has since been reorganised;
``evidence/path_map.csv`` maps every original path to its current location.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _path_map() -> dict[str, str]:
    table = ROOT / "evidence/path_map.csv"
    if not table.is_file():
        return {}
    with table.open(newline="") as stream:
        return {row["original_path"]: row["current_path"] for row in csv.DictReader(stream)}


def current(path: str | Path) -> str:
    """Return the current repository-relative path of an original path (unchanged if not moved)."""
    return _path_map().get(str(path), str(path))
