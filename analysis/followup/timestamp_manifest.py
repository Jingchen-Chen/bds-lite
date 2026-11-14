"""Write revision/followup/TIMESTAMP_MANIFEST.txt for the registration commit.

Same content as revision/acdc_confirm/TIMESTAMP_MANIFEST.txt: the commit and tree hashes and the
sha256 of every registered file, taken from the committed bytes (git show <commit>:<path>).

    .venv/bin/python analysis/followup/timestamp_manifest.py <registration commit>
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "revision/followup/TIMESTAMP_MANIFEST.txt"


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def digest(commit: str, path: str) -> str:
    return hashlib.sha256(git("show", f"{commit}:{path}")).hexdigest()


def main() -> None:
    commit = git("rev-parse", sys.argv[1]).decode().strip()
    tree = git("rev-parse", f"{commit}^{{tree}}").decode().strip()
    files = git("ls-tree", "-r", "--name-only", commit).decode().splitlines()
    configs = sorted(f for f in files if f.startswith("configs/revision/followup/"))
    scripts = sorted(f for f in files if f.startswith("scripts/kd_followup/"))
    lines = [
        "Follow-up controls registration: timestamp manifest",
        "All sha256 values are of the bytes committed in the registration commit.",
        "",
        f"registration_commit {commit}",
        f"registration_commit_tree {tree}",
    ]
    for label, path in (
        ("registration_sha256", "revision/followup_registration.md"),
        ("queue_e1e2_sha256", "revision/queue_followup_e1e2.yaml"),
        ("queue_e3_sha256", "revision/queue_followup_e3.yaml"),
    ):
        lines.append(f"{label} {digest(commit, path)}  {path}")
    lines += ["", f"registered_configs ({len(configs)}):"]
    lines += [f"{digest(commit, p)}  {p}" for p in configs]
    lines += ["", "scripts/kd_followup:"]
    lines += [f"{digest(commit, p)}  {p}" for p in scripts]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(OUT.relative_to(ROOT), len(configs), "configs")


if __name__ == "__main__":
    main()
