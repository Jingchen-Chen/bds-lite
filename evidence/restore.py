"""Restore checksum-verified supporting records without overwriting local changes.

Uses only the Python standard library. Frozen scripts keep their original paths;
the public checkout keeps the corresponding records in evidence archives.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import tarfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {"results": "results/", "protocols": "protocols/", "provenance": "docs/"}


def relative_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or str(path) != name:
        raise ValueError(f"Invalid archived path: {name}")
    return path


def contained(base: Path, name: str) -> Path:
    relative_name(name)
    target = base / name
    if target.is_symlink() or not target.resolve().is_relative_to(base.resolve()):
        raise ValueError(f"Archived path escapes destination or is a symlink: {name}")
    return target


def restore(
    destination: Path = ROOT,
    groups: tuple[str, ...] = (),
    *,
    check: bool = False,
) -> tuple[int, int]:
    with (ROOT / "evidence/archived_paths.csv").open(newline="") as stream:
        records = list(csv.DictReader(stream))
    if groups:
        prefixes = tuple(GROUPS[group] for group in groups)
        records = [row for row in records if row["path"].startswith(prefixes)]
    archives = defaultdict(lambda: defaultdict(list))
    paths = set()
    for row in records:
        name, archive = row["path"], row["archive"]
        if name in paths:
            raise ValueError(f"Duplicate destination in manifest: {name}")
        paths.add(name)
        source = contained(ROOT, archive)
        if source.parent != ROOT / "evidence" or not archive.endswith(".tar.gz"):
            raise ValueError(f"Invalid evidence archive: {archive}")
        relative_name(row["member"])
        target = contained(destination, name)
        if target.exists():
            if (
                not target.is_file()
                or hashlib.sha256(target.read_bytes()).hexdigest() != row["sha256"]
            ):
                raise ValueError(f"Refusing to overwrite a different local file: {target}")
        archives[archive][row["member"]].append(row)

    verified = restored = 0
    for archive, requested in archives.items():
        seen = set()
        with tarfile.open(ROOT / archive, "r|gz") as stream:
            for member in stream:
                if member.name not in requested:
                    continue
                if member.name in seen or not member.isfile():
                    raise ValueError(f"Invalid or duplicate archive member: {member.name}")
                seen.add(member.name)
                data = stream.extractfile(member).read()
                digest = hashlib.sha256(data).hexdigest()
                for row in requested[member.name]:
                    if digest != row["sha256"]:
                        raise ValueError(f"Checksum mismatch: {archive}:{member.name}")
                    verified += 1
                    target = contained(destination, row["path"])
                    if not check and not target.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        # Exclusive creation also protects files created since the preflight.
                        with target.open("xb") as output:
                            output.write(data)
                        restored += 1
        missing = requested.keys() - seen
        if missing:
            raise ValueError(f"Missing archive members in {archive}: {sorted(missing)}")
    return verified, restored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", action="append", choices=GROUPS, default=[])
    parser.add_argument("--destination", type=Path, default=ROOT)
    parser.add_argument(
        "--check", action="store_true", help="Verify archives without writing files"
    )
    args = parser.parse_args()
    verified, restored = restore(args.destination, tuple(args.group), check=args.check)
    print(f"{verified} records verified; {restored} files restored")


if __name__ == "__main__":
    main()
