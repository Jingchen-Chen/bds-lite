"""Refuse a public release whose history contains any restricted figure (LICENSING_AUDIT.md).

Usage: python analysis/revision/check_release_clean.py <git-ref> [--repo PATH]
Scans every blob reachable from <git-ref> (full history) and fails if a blob has the content of a
file in revision/restricted_local/ or one of the restricted paths below. Exit code 1 on failure.
--repo scans <git-ref> in another repository (for example a clean release repository) while the
restricted files are still read from this repository; without it, this repository is scanned.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESTRICTED = ROOT / "revision/restricted_local"
PATH_MARKERS = (
    "dataset_examples",
    "qualitative_isic2018.",
    "qualitative_acdc.",
    "qualitative_synapse.",
    "official_isic_qualitative_draft",
    "isic_duplicate_threshold",
    "figure_dataset_examples",
    "restricted_local/",
    # Found 2026-11-08 in the public repository (first release 1b0c4f6 to 5319de3, and the
    # manuscript PDF at 3104cb3): panels and a manuscript PDF showing dataset images.
    "failure_panels",
    "failure_cases",
    "manuscript/BDS-Lite.pdf",
)


def git(*args: str, repo: Path = ROOT) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout


def main(ref: str, repo: Path = ROOT) -> int:
    restricted = {
        git("hash-object", str(p)).strip(): p.name for p in RESTRICTED.glob("*") if p.is_file()
    }
    if not restricted:
        print("no restricted files found in revision/restricted_local/; cannot check content")
        return 1
    problems = []
    for line in git("rev-list", "--objects", ref, repo=repo).splitlines():
        blob, _, path = line.partition(" ")
        if blob in restricted:
            problems.append(f"content of {restricted[blob]} at {path or blob}")
        elif (
            path
            and any(marker in path for marker in PATH_MARKERS)
            and path.endswith((".png", ".pdf", ".jpg"))
        ):
            problems.append(f"restricted path {path}")
    for item in sorted(set(problems)):
        print("RESTRICTED", item)
    print(f"{ref}: {'FAIL' if problems else 'clean'} ({len(restricted)} restricted files known)")
    return 1 if problems else 0


if __name__ == "__main__":
    args = sys.argv[1:]
    target = ROOT
    if "--repo" in args:
        i = args.index("--repo")
        target = Path(args[i + 1]).resolve()
        del args[i : i + 2]
    sys.exit(main(args[0] if args else "HEAD", target))
