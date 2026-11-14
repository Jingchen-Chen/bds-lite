"""Amendment 1, safeguard 2: hash every preserved artifact before the amendment is applied.

    .venv/bin/python scripts/acdc_confirm/amendment1/snapshot.py write    # once, before amending
    .venv/bin/python scripts/acdc_confirm/amendment1/snapshot.py verify   # any time afterwards

`write` copies the original (failed) integrity output to
revision/acdc_confirm/amendment1/integrity_original_failed.json and records the sha256 of:
every file in the 34 replication run folders and the 6 exploratory run folders; the queue
registration; the 34 configs; the pre-registration; every registered source file; the original
integrity output; the equivalence-check record; the official-test manifest and preprocessing
record. `write` refuses to overwrite an existing snapshot. `verify` recomputes every hash and
exits non-zero on any difference, missing file or new file inside a snapshotted run folder.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.acdc_confirm.common import (  # noqa: E402
    CONFIGS,
    EXPLORATORY,
    OFFICIAL_MANIFEST,
    OFFICIAL_PREPROCESSING,
    PREREG,
    QUEUE,
    RUNS,
    WORKSPACE,
    registered_runs,
    run_name,
)
from scripts.kd.common import sha  # noqa: E402

FOLDER = WORKSPACE / "amendment1"
SNAPSHOT = FOLDER / "pre_amendment_snapshot.json"
ORIGINAL_INTEGRITY = FOLDER / "integrity_original_failed.json"


def run_folders() -> list[Path]:
    names = [run_name(a, s) for a, s in registered_runs()]
    names += [n for arm in ("decoder", "kd_band") for n in EXPLORATORY[arm]]
    return [RUNS / n for n in names]


def files() -> list[Path]:
    queue = yaml.safe_load(QUEUE.read_text())
    paths = [QUEUE, PREREG, OFFICIAL_MANIFEST, OFFICIAL_PREPROCESSING]
    paths += [WORKSPACE / "equivalence_check.json", ORIGINAL_INTEGRITY]
    paths += sorted(CONFIGS.glob("*.yaml"))
    paths += [ROOT / s["path"] for s in queue["runs"][0]["sources"]]
    for folder in run_folders():
        paths += sorted(p for p in folder.rglob("*") if p.is_file())
    return paths


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def write() -> None:
    if SNAPSHOT.exists() or ORIGINAL_INTEGRITY.exists():
        raise FileExistsError("snapshot already written")
    FOLDER.mkdir(parents=True, exist_ok=True)
    shutil.copy2(WORKSPACE / "integrity.json", ORIGINAL_INTEGRITY)
    hashes = {rel(p): sha(p) for p in files()}
    record = dict(
        written_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        run_folders=[rel(f) for f in run_folders()],
        files=len(hashes),
        sha256=hashes,
    )
    SNAPSHOT.write_text(json.dumps(record, indent=2) + "\n")
    print(len(hashes), "files hashed;", rel(SNAPSHOT), "sha256", sha(SNAPSHOT))


def verify() -> list[str]:
    record = json.loads(SNAPSHOT.read_text())
    problems = []
    for path, digest in record["sha256"].items():
        if not (ROOT / path).is_file():
            problems.append(f"missing: {path}")
        elif sha(ROOT / path) != digest:
            problems.append(f"changed: {path}")
    for folder in record["run_folders"]:
        for p in (ROOT / folder).rglob("*"):
            if p.is_file() and rel(p) not in record["sha256"]:
                problems.append(f"new file in a run folder: {rel(p)}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["write", "verify"])
    if parser.parse_args().mode == "write":
        write()
        return
    problems = verify()
    print(json.dumps(dict(unchanged=not problems, problems=problems), indent=2))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
