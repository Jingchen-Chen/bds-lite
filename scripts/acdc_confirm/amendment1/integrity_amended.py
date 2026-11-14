"""Amendment 1 integrity wrapper (pre-test). Runs every original check; tolerates one exact defect.

The unchanged scripts/acdc_confirm/integrity.py::check is run first and its result is kept
verbatim. Its only tolerated problem is "<run>: source hashes differ from registration", produced by
analysis/kd3/integrity.py::check_run because the queue registers 74 sources (the parent study's
kd_sources(), 61 files, plus the 13 scripts/acdc_confirm/*.py study files) while the frozen
scripts/kd_mc/train_mc.py records only the 61 kd_sources() in provenance.json. That problem is
accepted for a run only if ALL of the following hold, otherwise it stays a failure:

1. the registration has exactly 74 sources: 13 under scripts/acdc_confirm/ and 61 elsewhere;
2. the run's recorded provenance has exactly the 61 non-study paths, each with the registered hash
   (no missing, extra or differing entry);
3. the 13 registered study paths are exactly the scripts/acdc_confirm/*.py files of freeze commit
   345cacd, and each file's sha256 on disk, in the registration and in that commit are equal;
4. every one of the 74 registered files on disk still has its registered sha256.

Any other problem reported by the original check, in any run or globally, is a failure. Writes
revision/acdc_confirm/amendment1/integrity_amended[_post_test].json (with the original result).

    .venv/bin/python scripts/acdc_confirm/amendment1/integrity_amended.py [--post-test]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.acdc_confirm import integrity  # noqa: E402
from scripts.acdc_confirm.common import QUEUE, RUNS, WORKSPACE  # noqa: E402
from scripts.kd.common import sha  # noqa: E402

FREEZE_COMMIT = "345cacd02855bc93105dd2c2e0a1730c564ea953"
STUDY_PREFIX = "scripts/acdc_confirm/"
TOLERATED = "source hashes differ from registration"
N_REGISTERED, N_STUDY, N_RECORDED = 74, 13, 61
OUT = WORKSPACE / "amendment1"


def frozen_study_files(commit: str = FREEZE_COMMIT) -> dict[str, str]:
    """sha256 of each scripts/acdc_confirm/*.py (top level) as committed in `commit`."""
    listing = subprocess.run(
        ["git", "ls-tree", "--name-only", commit, STUDY_PREFIX],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    out = {}
    for path in listing:
        if path.endswith(".py") and "/" not in path[len(STUDY_PREFIX) :]:
            blob = subprocess.run(
                ["git", "show", f"{commit}:{path}"], cwd=ROOT, check=True, capture_output=True
            ).stdout
            out[path] = hashlib.sha256(blob).hexdigest()
    return out


def source_problems(
    registered: dict[str, str],
    recorded: dict[str, str],
    on_disk: dict[str, str | None],
    frozen: dict[str, str],
) -> list[str]:
    """Conditions 1-4 for one run; empty means the mismatch is exactly the documented defect."""
    problems = []
    study = {p: h for p, h in registered.items() if p.startswith(STUDY_PREFIX)}
    rest = {p: h for p, h in registered.items() if not p.startswith(STUDY_PREFIX)}
    if (len(registered), len(study), len(rest)) != (N_REGISTERED, N_STUDY, N_RECORDED):
        problems.append(
            f"registration has {len(registered)} sources ({len(study)} study, {len(rest)} other);"
            f" expected {N_REGISTERED} ({N_STUDY}, {N_RECORDED})"
        )
    if len(recorded) != N_RECORDED:
        problems.append(f"provenance records {len(recorded)} sources; expected {N_RECORDED}")
    for path in sorted(set(rest) - set(recorded)):
        problems.append(f"registered source not recorded: {path}")
    for path in sorted(set(recorded) - set(rest)):
        problems.append(f"recorded source not in the non-study registration: {path}")
    for path in sorted(set(rest) & set(recorded)):
        if rest[path] != recorded[path]:
            problems.append(f"recorded hash differs from registration: {path}")
    if set(study) != set(frozen):
        problems.append(
            f"study files differ from {FREEZE_COMMIT[:7]}: registered-only "
            f"{sorted(set(study) - set(frozen))}, commit-only {sorted(set(frozen) - set(study))}"
        )
    for path in sorted(set(study) & set(frozen)):
        if study[path] != frozen[path]:
            problems.append(
                f"study file registration hash differs from {FREEZE_COMMIT[:7]}: {path}"
            )
    for path in sorted(registered):
        if on_disk.get(path) != registered[path]:
            problems.append(f"registered source changed on disk: {path}")
    return problems


def amend(original: dict, run_sources: dict[str, dict]) -> dict:
    """Classify problems; run_sources[run] = dict(registered, recorded, on_disk, frozen)."""
    accepted, rejected = [], []
    for problem in original["problems"]:
        run, sep, message = problem.partition(": ")
        if not sep or message != TOLERATED or run not in run_sources:
            rejected.append(problem)
            continue
        issues = source_problems(**run_sources[run])
        if issues:
            rejected.append(problem)
            rejected += [f"{run}: amendment condition failed: {i}" for i in issues]
        else:
            accepted.append(problem)
    return dict(
        passed=not rejected,
        problems=rejected,
        accepted_under_amendment_1=accepted,
        original_passed=original["passed"],
    )


def collect(entries: dict[str, dict]) -> dict[str, dict]:
    frozen = frozen_study_files()
    out = {}
    for name, entry in entries.items():
        provenance = RUNS / name / "provenance.json"
        if not provenance.exists():
            continue  # the original check already reports incomplete runs; never accepted here
        registered = {s["path"]: s["sha256"] for s in entry["sources"]}
        out[name] = dict(
            registered=registered,
            recorded=json.loads(provenance.read_text())["source_hashes"],
            on_disk={p: sha(ROOT / p) if (ROOT / p).is_file() else None for p in registered},
            frozen=frozen,
        )
    return out


def check(allow_test: bool = False) -> dict:
    original = integrity.check(allow_test)
    entries = {e["run_id"]: e for e in yaml.safe_load(QUEUE.read_text())["runs"]}
    result = amend(original, collect(entries))
    result.update(
        amendment="revision/acdc_confirm/amendment1/AMENDMENT_1.md",
        freeze_commit=FREEZE_COMMIT,
        original=original,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "_post_test" if allow_test else ""
    (OUT / f"integrity_amended{suffix}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--post-test", action="store_true")
    result = check(parser.parse_args().post_test)
    summary = dict(
        passed=result["passed"],
        problems=result["problems"],
        accepted_under_amendment_1=len(result["accepted_under_amendment_1"]),
    )
    print(json.dumps(summary, indent=2))
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
