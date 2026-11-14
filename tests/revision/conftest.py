"""Skip the two Amendment 1 tests that read the freeze commit when that commit is absent.

`tests/revision/test_acdc_confirm_amendment1.py` is listed with its SHA-256 in
`revision/acdc_confirm/amendment1/AMENDMENT_1.md`, so it is left byte-identical. Two of its tests
call `integrity_amended.frozen_study_files()`, which reads commit 345cacd with `git ls-tree` and
`git show`. The public snapshot carries no development history, so there the commit object does
not exist and the two tests are skipped. Where the commit exists, nothing is changed and the tests
run exactly as before.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FREEZE_COMMIT = "345cacd02855bc93105dd2c2e0a1730c564ea953"
NEEDS_HISTORY = {
    "test_acdc_confirm_amendment1.py::test_real_registration_matches_the_freeze_commit_split",
    "test_acdc_confirm_amendment1.py::test_no_top_level_study_file_was_added_after_the_freeze",
}


def _freeze_commit_present() -> bool:
    try:
        found = subprocess.run(
            ["git", "cat-file", "-e", f"{FREEZE_COMMIT}^{{commit}}"],
            cwd=ROOT,
            capture_output=True,
        )
    except OSError:
        return False
    return found.returncode == 0


def pytest_collection_modifyitems(config, items):
    if _freeze_commit_present():
        return
    skip = pytest.mark.skip(
        reason=f"requires the private development history (commit {FREEZE_COMMIT[:7]})"
    )
    for item in items:
        if f"{item.path.name}::{item.name}" in NEEDS_HISTORY:
            item.add_marker(skip)
