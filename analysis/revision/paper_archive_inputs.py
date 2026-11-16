"""Read saved paper inputs from the public evidence archives; never execute experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tarfile
import tempfile
from pathlib import Path

from bds_lite.paths import current

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(
    os.environ.get("BDS_PAPER_CACHE", Path(tempfile.gettempdir()) / "bds-lite-paper-inputs")
)


def prepare() -> None:
    """Materialize only case CSVs, completion records and split manifests, checking every byte."""
    archive = ROOT / "evidence/primary-evidence-20261110.tar.gz"
    manifest = {}
    for line in archive.with_suffix("").with_suffix(".sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        manifest[name] = digest
    with tarfile.open(archive) as stream:
        for member in stream:
            name = member.name
            if not member.isfile() or not (
                name.startswith("results/revision_runs/")
                and name.endswith(("/per_case_test.csv", "/final_metrics.json"))
                or name.startswith("recovered/followup/splits/")
                and name.endswith(".json")
            ):
                continue
            data = stream.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != manifest[name]:
                raise ValueError(f"Archive checksum mismatch: {name}")
            target = CACHE / name
            if target.exists() and target.read_bytes() != data:
                raise ValueError(f"Archive cache differs: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(data)


def logical(path: str | Path) -> str:
    path = Path(path)
    if path.is_absolute():
        for base in (CACHE, ROOT):
            if path.is_relative_to(base):
                return current(path.relative_to(base))
        raise ValueError("Paper input is outside the public repository and archive cache")
    return current(path)


def resolve(path: str | Path) -> Path:
    name = logical(path)
    saved = CACHE / name
    # Prefer archive bytes for the extracted inputs, even if local training outputs exist.
    return saved if saved.is_file() else ROOT / name


def config_weights() -> dict:
    with (ROOT / "results/paper_inputs/inventory_config_weights.csv").open() as stream:
        return {row["run_path"]: row for row in csv.DictReader(stream)}


def verify_config_weights() -> None:
    records = config_weights()
    by_archive = {}
    for record in records.values():
        by_archive.setdefault(record["archive"], []).append(record)
    for name, rows in by_archive.items():
        with tarfile.open(ROOT / name) as stream:
            for row in rows:
                data = stream.extractfile(row["member"]).read()
                if hashlib.sha256(data).hexdigest() != row["sha256"]:
                    raise ValueError(f"Configuration source mismatch: {row['member']}")
                weighted = bool(json.loads(data).get("loss", {}).get("foreground_class_weights"))
                if str(weighted) != row["weighted"]:
                    raise ValueError(f"Configuration classification mismatch: {row['member']}")
