"""Regression tests for Amendment 1 (integrity wrapper): valid and corrupted manifests."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
import yaml

from scripts.acdc_confirm.amendment1 import integrity_amended as ia

ROOT = Path(__file__).resolve().parents[2]
RUN = "acdc_confirm_kd_band_seed201"
TOLERATED = f"{RUN}: source hashes differ from registration"


def h(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def manifest() -> dict:
    other = {f"src/bds_lite/m{i:02d}.py": h(f"other{i}") for i in range(ia.N_RECORDED)}
    study = {f"scripts/acdc_confirm/s{i:02d}.py": h(f"study{i}") for i in range(ia.N_STUDY)}
    registered = {**other, **study}
    return dict(registered=registered, recorded=dict(other), on_disk=dict(registered), frozen=study)


def run(problems: list[str], sources: dict | None = None) -> dict:
    sources = {RUN: manifest()} if sources is None else sources
    return ia.amend(dict(passed=False, problems=problems), sources)


def corrupt(edit) -> dict:
    m = copy.deepcopy(manifest())
    edit(m)
    return run([TOLERATED], {RUN: m})


def test_valid_manifest_is_accepted():
    result = run([TOLERATED])
    assert result["passed"] and result["problems"] == []
    assert result["accepted_under_amendment_1"] == [TOLERATED]


def test_no_problems_passes():
    assert run([])["passed"]


@pytest.mark.parametrize(
    "edit",
    [
        lambda m: m["recorded"].update({"src/bds_lite/m00.py": h("tampered")}),
        lambda m: m["recorded"].pop("src/bds_lite/m05.py"),
        lambda m: m["recorded"].update({"src/bds_lite/new.py": h("new")}),
        lambda m: m["recorded"].update({"scripts/acdc_confirm/s00.py": h("study0")}),
        lambda m: m["on_disk"].update({"scripts/acdc_confirm/s03.py": h("edited")}),
        lambda m: m["on_disk"].update({"src/bds_lite/m10.py": None}),
        lambda m: m["on_disk"].update({"src/bds_lite/m11.py": h("edited")}),
        lambda m: m["frozen"].update({"scripts/acdc_confirm/s04.py": h("other bytes")}),
        lambda m: m["frozen"].update({"scripts/acdc_confirm/extra.py": h("x")}),
        lambda m: m["frozen"].pop("scripts/acdc_confirm/s12.py"),
        lambda m: m["registered"].update({"scripts/acdc_confirm/s99.py": h("s99")}),
        lambda m: m["registered"].pop("src/bds_lite/m60.py"),
        lambda m: m["registered"].update({"scripts/acdc_confirm/s01.py": h("re-registered")}),
    ],
    ids=[
        "recorded-hash-differs",
        "recorded-missing",
        "recorded-extra",
        "recorded-includes-study-file",
        "study-file-changed-on-disk",
        "registered-file-missing-on-disk",
        "other-file-changed-on-disk",
        "study-file-differs-from-freeze-commit",
        "freeze-commit-has-extra-study-file",
        "freeze-commit-lacks-study-file",
        "registration-has-extra-study-file",
        "registration-lacks-other-file",
        "registration-study-hash-differs",
    ],
)
def test_corrupted_manifest_is_rejected(edit):
    result = corrupt(edit)
    assert not result["passed"]
    assert TOLERATED in result["problems"]
    assert result["accepted_under_amendment_1"] == []
    assert any("amendment condition failed" in p for p in result["problems"])


@pytest.mark.parametrize(
    "problem",
    [
        f"{RUN}: artifact hash best.pt",
        f"{RUN}: config hash",
        f"{RUN}: epochs not contiguous 1-150",
        f"{RUN}: test output present",
        f"{RUN}: source hashes differ from registration (extra)",
        "acdc_confirm_decoder_seed999: source hashes differ from registration",
        "kd_band: fewer than 2 completed runs; primary analysis impossible",
        "pre-freeze equivalence check did not pass",
    ],
)
def test_any_other_problem_is_rejected(problem):
    result = run([TOLERATED, problem])
    assert not result["passed"]
    assert result["problems"] == [problem]
    assert result["accepted_under_amendment_1"] == [TOLERATED]


def test_tolerated_problem_needs_the_run_manifest():
    result = run([TOLERATED], sources={})
    assert not result["passed"] and result["problems"] == [TOLERATED]


QUEUE = ROOT / "revision/queue_acdc_confirm.yaml"


@pytest.mark.skipif(not QUEUE.exists(), reason="registration not present")
def test_real_registration_matches_the_freeze_commit_split():
    frozen = ia.frozen_study_files()
    assert len(frozen) == ia.N_STUDY
    for entry in yaml.safe_load(QUEUE.read_text())["runs"]:
        registered = {s["path"]: s["sha256"] for s in entry["sources"]}
        study = {p: d for p, d in registered.items() if p.startswith(ia.STUDY_PREFIX)}
        assert study == frozen
        assert len(registered) == ia.N_REGISTERED


def test_no_top_level_study_file_was_added_after_the_freeze():
    current = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "scripts/acdc_confirm").glob("*.py"))
    assert current == sorted(ia.frozen_study_files())


def test_evaluation_wrapper_leaves_the_original_gate_on_import():
    from scripts.acdc_confirm import evaluate_test, integrity
    from scripts.acdc_confirm.amendment1 import evaluate_amended  # noqa: F401

    assert evaluate_test.check is integrity.check
