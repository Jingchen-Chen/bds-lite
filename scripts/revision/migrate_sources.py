"""Non-overwriting migration from a precomputed SHA256 inventory."""

import csv
import hashlib
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALIASES = {"bds-followup": "followup", "bds-lite-new": "lite-new", "bds-latex": "peerj-submission"}


def main():
    rows = list(csv.DictReader((ROOT / "revision/audit/inventory.csv").open()))
    known = {}
    for row in rows:
        if row["root"] == str(ROOT) and row["kind"] == "file":
            known.setdefault(row["sha256"], ROOT / row["relative_path"])
    manifest = ROOT / "revision/audit/migration.csv"
    with manifest.open("x") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["source", "destination", "sha256", "action", "target"],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            alias = ALIASES.get(Path(row["root"]).name)
            if alias is None or row["kind"] != "file":
                continue
            rel = Path(row["relative_path"])
            if str(rel).startswith("data/tools/") or rel.suffix in {
                ".aux",
                ".blg",
                ".fdb_latexmk",
                ".fls",
                ".synctex.gz",
            }:
                continue
            source = Path(row["root"]) / rel
            dest = ROOT / "recovered" / alias / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() or dest.is_symlink():
                raise FileExistsError(dest)
            target = known.get(row["sha256"])
            if target:
                dest.symlink_to(os.path.relpath(target, dest.parent))
                action = "deduplicated_internal_symlink"
            else:
                shutil.copy2(source, dest)
                if hashlib.file_digest(dest.open("rb"), "sha256").hexdigest() != row["sha256"]:
                    raise ValueError(f"transfer hash mismatch: {dest}")
                known[row["sha256"]] = dest
                action = "copied_hash_verified"
            writer.writerow(
                dict(
                    source=str(source),
                    destination=str(dest.relative_to(ROOT)),
                    sha256=row["sha256"],
                    action=action,
                    target=str(target.relative_to(ROOT)) if target else "",
                )
            )
        stream.flush()
    print("Migration complete:", manifest)


if __name__ == "__main__":
    main()
