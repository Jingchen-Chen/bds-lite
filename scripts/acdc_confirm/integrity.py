"""Integrity checks for the 34 replication runs, before the one-time test evaluation.

Per run (analysis/kd3/integrity.py::check_run, unchanged): 150 contiguous epochs; validation-only
final marker; artifact hashes; config bytes and source hashes equal to the queue registration;
selected epoch = first maximum of validation DSC; finite logged losses; no test output.
In addition: each config passes common.validate; teacher-cache hash in every provenance; seeds
201-217 complete in both arms; L_kd logged in every D+KD-band run and in no decoder-only run;
the pre-freeze equivalence check passed. Feasibility (reported, never used to exclude): L_kd
decreased in every D+KD-band run. Runs declared failed under section 5 (failed_runs.json) are
reported, not checked; the rest must be complete. Writes
revision/acdc_confirm/integrity[_post_test].json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from analysis.kd3.integrity import check_run  # noqa: E402
from scripts.acdc_confirm.common import (  # noqa: E402
    QUEUE,
    RUNS,
    SEEDS,
    TEACHER_CACHE_SHA256,
    WORKSPACE,
    failed_runs,
    registered_runs,
    run_name,
    validate,
)


def equivalence_passed() -> bool:
    report = json.loads((WORKSPACE / "equivalence_check.json").read_text())
    c = report["train_run_vs_launcher"]
    return (
        c["initial"]["bitwise_equal"] == c["initial"]["tensors"]
        and c["final"]["bitwise_equal"] == c["final"]["tensors"]
        and c["optimizer_state_max_abs_diff"] == 0.0
        and c["batch_order_identical"]
        and not report["evaluation_outputs_produced"]
    )


def check(allow_test: bool = False) -> dict:
    queue = yaml.safe_load(QUEUE.read_text())
    entries = {e["run_id"]: e for e in queue["runs"]}
    problems: list[str] = []
    records = []
    failed = failed_runs()
    for arm, seed in registered_runs():
        name = run_name(arm, seed)
        folder = RUNS / name
        if name in failed:  # reported as failed (section 5), never silently excluded
            if any((folder / n).exists() for n in ["preds_test.npz", "per_case_test.csv"]):
                problems.append(f"{name}: test output in a failed run")
            continue
        if not (folder / "final_metrics.json").exists():
            problems.append(f"{name}: not complete")
            continue
        record = check_run(folder, entries[name], problems, allow_test)
        config = json.loads((folder / "config.json").read_text())
        try:
            validate(config)
        except ValueError as error:
            problems.append(f"{name}: {error}")
        provenance = json.loads((folder / "provenance.json").read_text())
        cache = provenance["data_hashes"].get("teacher_cache")
        if cache != {"train_teacher_prob_fp16.npy": TEACHER_CACHE_SHA256}:
            problems.append(f"{name}: teacher cache hash in provenance")
        has_kd = "kd_epoch1" in record
        if has_kd != (arm == "kd_band"):
            problems.append(f"{name}: L_kd logged={has_kd} for arm {arm}")
        record.update(arm=arm, seed=seed)
        records.append(record)
    for arm in ("decoder", "kd_band"):
        done = sorted(r["seed"] for r in records if r["arm"] == arm)
        expected = [s for s in SEEDS if run_name(arm, s) not in failed]
        if done != expected:
            problems.append(f"{arm}: completed seeds {done} != registered minus failed {expected}")
        if len(done) < 2:
            problems.append(f"{arm}: fewer than 2 completed runs; primary analysis impossible")
    if not equivalence_passed():
        problems.append("pre-freeze equivalence check did not pass")
    feasibility = dict(
        kd_decreased_all=all(r.get("kd_decreased") for r in records if r["arm"] == "kd_band")
    )
    result = dict(
        passed=not problems,
        problems=problems,
        feasibility=feasibility,
        failed_runs=list(failed.values()),
        runs=records,
    )
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    suffix = "_post_test" if allow_test else ""
    (WORKSPACE / f"integrity{suffix}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--post-test", action="store_true")
    result = check(parser.parse_args().post_test)
    print(json.dumps(dict(passed=result["passed"], problems=result["problems"]), indent=2))
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
