"""Verify source preservation and recovered bytes against pre-migration SHA256."""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    originals = list(csv.DictReader((ROOT / "revision/audit/inventory.csv").open()))
    migrated = list(csv.DictReader((ROOT / "revision/audit/migration.csv").open()))
    failures = []
    count = 0
    for row in originals:
        if row["kind"] != "file":
            continue
        path = Path(row["root"]) / row["relative_path"]
        if digest(path) != row["sha256"]:
            failures.append(dict(path=str(path), reason="original bytes changed"))
        count += 1
    for row in migrated:
        path = ROOT / row["destination"]
        if not path.resolve().is_relative_to(ROOT):
            failures.append(dict(path=str(path), reason="link outside primary"))
        elif digest(path) != row["sha256"]:
            failures.append(dict(path=str(path), reason="destination hash mismatch"))
    report = dict(
        original_files_checked=count, migrated_files_checked=len(migrated), failures=failures
    )
    (ROOT / "revision/audit/migration_verification.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report))
    if failures:
        raise ValueError("migration/source preservation failed")


if __name__ == "__main__":
    main()
