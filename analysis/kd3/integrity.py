"""Integrity checks before the one-time test evaluation (Amendment 1 item 4; multiclass protocol 6).

For one dataset: the teacher, the cache step and the three D+KD-band runs. Checks, all of which must
pass: 150 contiguous epochs; final marker validation-only; artifact hashes; config bytes and source
hashes equal to the queue registration; selected epoch equal to the first maximum of validation
DSC; all logged losses finite; teacher cache hash in each student's provenance equal to the cache
record; gate PASSED; distinct seeds with differing first-epoch training losses; no test output.
Feasibility (Amendment 1 item 3; multiclass protocol 6): L_kd decreased in every student.
Writes revision/kd_unified/integrity_<dataset>.json and exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.kd.common import sha  # noqa: E402

RUNS = ROOT / "results/revision_runs"
OUT = ROOT / "revision/kd_unified"
QUEUES = {
    "isic2018": "revision/queue_kd_phase1.yaml",
    "acdc": "revision/queue_kd_mc.yaml",
    "synapse": "revision/queue_kd_mc.yaml",
}
TEST_OUTPUTS = ["preds_test.npz", "per_case_test.csv", "test_metrics.json"]


def names(dataset: str) -> dict:
    return dict(
        teacher=f"{dataset}_kd_teacher_seed100",
        cache=f"{dataset}_kd_teacher_cache",
        students=[f"{dataset}_kd_band_seed{s}" for s in (1, 2, 3)],
    )


def finite(history) -> bool:
    return all(
        math.isfinite(v)
        for row in history
        for part in ("train", "val")
        for v in row[part].values()
        if isinstance(v, float)
    )


def check_run(folder: Path, entry: dict, problems: list, allow_test: bool) -> dict:
    final = json.loads((folder / "final_metrics.json").read_text())
    history = json.loads((folder / "history.json").read_text())
    provenance = json.loads((folder / "provenance.json").read_text())
    config_file = ROOT / entry["config"]
    record = dict(run=folder.name)

    def need(condition, message):
        if not condition:
            problems.append(f"{folder.name}: {message}")

    need(final["epochs"] == 150 and final["test_evaluated"] is False, "final marker")
    need([h["epoch"] for h in history] == list(range(1, 151)), "epochs not contiguous 1-150")
    for name, digest in final["artifacts"].items():
        need(sha(folder / name) == digest, f"artifact hash {name}")
    need(sha(config_file) == entry["config_sha256"] == final["config_sha256"], "config hash")
    need(
        json.loads((folder / "config.json").read_text()) == yaml.safe_load(config_file.read_text()),
        "config.json differs from registered config",
    )
    registered = {s["path"]: s["sha256"] for s in entry["sources"]}
    need(provenance["source_hashes"] == registered, "source hashes differ from registration")
    dsc = [h["val"]["val_dsc"] for h in history]
    first = dsc.index(max(dsc)) + 1
    need(final["selected_epoch"] == first and final["best_val_dsc"] == max(dsc), "selection")
    need(finite(history), "nonfinite logged loss")
    if not allow_test:
        need(not any((folder / n).exists() for n in TEST_OUTPUTS), "test output present")
    record.update(
        selected_epoch=final["selected_epoch"],
        epoch1_train_total=history[0]["train"]["total"],
        cache_hash=provenance["data_hashes"].get("teacher_cache"),
    )
    if "kd" in history[0]["train"]:
        kd = [h["train"]["kd"] for h in history]
        record["kd_epoch1"], record["kd_last10_mean"] = kd[0], sum(kd[-10:]) / 10
        record["kd_decreased"] = record["kd_last10_mean"] < kd[0]
    return record


def check(dataset: str, allow_test: bool = False) -> dict:
    queue = yaml.safe_load((ROOT / QUEUES[dataset]).read_text())
    entries = {e["run_id"]: e for e in queue["runs"]}
    n = names(dataset)
    problems: list[str] = []
    cache = json.loads((RUNS / n["cache"] / "cache_complete.json").read_text())
    if sha(RUNS / n["cache"] / "train_teacher_prob_fp16.npy") != cache["cache_sha256"]:
        problems.append("teacher cache hash")
    if cache["gate"]["status"] != "PASSED":
        problems.append(f"teacher gate {cache['gate']['status']}")
    teacher = check_run(RUNS / n["teacher"], entries[n["teacher"]], problems, allow_test)
    students = [check_run(RUNS / s, entries[s], problems, allow_test) for s in n["students"]]
    for s in students:
        expected = {"train_teacher_prob_fp16.npy": cache["cache_sha256"]}
        if s["cache_hash"] != expected:
            problems.append(f"{s['run']}: cache hash in provenance")
    seeds = [
        json.loads((RUNS / s / "config.json").read_text())["project"]["seed"] for s in n["students"]
    ]
    if sorted(seeds) != [1, 2, 3]:
        problems.append("student seeds are not 1, 2, 3")
    if len({s["epoch1_train_total"] for s in students}) != 3:
        problems.append("first-epoch losses identical across seeds")
    feasibility = dict(
        F1_gate=cache["gate"]["status"] == "PASSED",
        F2_kd_decreased=all(s.get("kd_decreased") for s in students),
        F3_batch_size=len(
            {
                json.loads((RUNS / s / "config.json").read_text())["trainer"]["batch_size"]
                for s in n["students"]
            }
        )
        == 1,
    )
    if not all(feasibility.values()):
        problems.append(f"feasibility {feasibility}")
    result = dict(
        dataset=dataset,
        passed=not problems,
        problems=problems,
        gate=cache["gate"],
        cache_check=cache["check"],
        feasibility=feasibility,
        teacher=teacher,
        students=students,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "_post_test" if allow_test else ""
    (OUT / f"integrity_{dataset}{suffix}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(QUEUES))
    args = parser.parse_args()
    result = check(args.dataset)
    print(json.dumps(dict(passed=result["passed"], problems=result["problems"]), indent=2))
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
