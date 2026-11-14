"""Mandatory sanity checks before any teacher inference (results/teacher_eval/PROTOCOL.md).

For every student run of the six comparison groups (88 runs):
1. saved per-case CSV and prediction NPZ hashes equal the one-time evaluation record;
2. per-slice, per-class metrics are recomputed from the saved argmax predictions against the
   official masks (hash-checked official loader) with the unchanged evaluator function
   `analysis.revision.recover_predictions.case_metrics` (the one train_run.evaluate calls), and
   must equal the saved per-case rows exactly;
3. run means are aggregated with the unchanged `expansion_statistics.unit_values`; group means of
   run-mean Boundary F1 must match the paper (|diff| <= 1e-4) and the frozen full-precision
   values (|diff| <= 1e-10).
Writes results/teacher_eval/sanity_checks.json and sanity_student_runs.csv. Exit 1 on failure.

    PYTHONPATH=src:. .venv/bin/python analysis/teacher_eval/sanity.py
"""

from __future__ import annotations

import csv
import io
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _path in (REPO, REPO / "src", REPO / "analysis/revision", HERE):
    sys.path.insert(0, str(_path))
import expansion_statistics as ex  # noqa: E402
from common import (  # noqa: E402
    EXPECTED_RUNS,
    FROZEN_BF1,
    GROUPS,
    OUT,
    PAPER_BF1,
    SANITY_TOLERANCE,
    case_file,
    pred_file,
    run_names,
)

from analysis.revision.recover_predictions import case_metrics  # noqa: E402
from scripts.acdc_confirm.evaluate_test import official_loader  # noqa: E402
from scripts.kd.common import sha, write_json  # noqa: E402

MASKS: dict[str, np.ndarray] = {}


def official_masks() -> dict[str, np.ndarray]:
    dataset = official_loader().dataset  # verifies manifest and every NPZ hash
    out = {}
    for i in range(len(dataset)):
        item = dataset[i]
        out[item["id"]] = item["mask"].numpy()
    return out


def as_csv_text(rows: list[dict]) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def recompute(job: tuple[str, str]) -> dict:
    group, run = job
    data = np.load(pred_file(group, run))
    rows = []
    for sample, pred in zip(data["ids"].tolist(), data["predictions"], strict=True):
        for label in range(1, 4):
            rows.append(dict(id=sample, **case_metrics(pred, MASKS[sample], label)))
    saved = case_file(group, run).read_text()
    return dict(
        group=group,
        run=run,
        slices=int(len(data["ids"])),
        rows=len(rows),
        recomputed_equals_saved=as_csv_text(rows) == saved,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    problems = []
    hash_checks = 0
    for group, (folder, _, _, record_name) in GROUPS.items():
        record = {
            r["run"]: r
            for r in json.loads((folder / record_name).read_text())["runs"]
            if r["split"] == "official_test"
        }
        names = run_names(group)
        if len(names) != EXPECTED_RUNS[group]:
            problems.append(f"{group}: {len(names)} runs")
        for run in names:
            entry = record.get(run)
            if entry is None:
                problems.append(f"{run}: no official_test entry in {record_name}")
                continue
            for name, digest in entry["artifacts"].items():
                hash_checks += 1
                if sha(folder / run / name) != digest:
                    problems.append(f"{run}/{name}: hash differs from {record_name}")
    MASKS.update(official_masks())
    jobs = [(g, r) for g in GROUPS for r in run_names(g)]
    with ProcessPoolExecutor(
        max_workers=12, mp_context=multiprocessing.get_context("fork")
    ) as pool:
        recomputed = list(pool.map(recompute, jobs))
    rows, group_values = [], {}
    for item in recomputed:
        if not item["recomputed_equals_saved"]:
            problems.append(f"{item['run']}: recomputed per-case rows differ from saved file")
        values = ex.unit_values(case_file(item["group"], item["run"]), "acdc")
        units = len(values["boundary_f1"])
        if units != 50:
            problems.append(f"{item['run']}: {units} patients")
        means = {m: float(np.mean(list(values[m].values()))) for m in ex.METRICS}
        group_values.setdefault(item["group"], []).append(means["boundary_f1"])
        rows.append(dict(item, patients=units, **{f"mean_{m}": means[m] for m in ex.METRICS}))
    checks = []
    for group, values in group_values.items():
        mean = float(np.mean(values))
        if group in PAPER_BF1:
            diff = mean - PAPER_BF1[group]
            ok = abs(diff) <= SANITY_TOLERANCE
            checks.append(
                dict(
                    group=group,
                    runs=len(values),
                    reference="paper (4 decimals)",
                    expected=PAPER_BF1[group],
                    recomputed=mean,
                    difference=diff,
                    tolerance=SANITY_TOLERANCE,
                    passed=ok,
                )
            )
            if not ok:
                problems.append(f"{group}: {mean} vs paper {PAPER_BF1[group]}")
        diff = mean - FROZEN_BF1[group]
        ok = abs(diff) <= 1e-10
        checks.append(
            dict(
                group=group,
                runs=len(values),
                reference="frozen full-precision arm mean",
                expected=FROZEN_BF1[group],
                recomputed=mean,
                difference=diff,
                tolerance=1e-10,
                passed=ok,
            )
        )
        if not ok:
            problems.append(f"{group}: {mean} vs frozen {FROZEN_BF1[group]}")
    with (OUT / "sanity_student_runs.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    result = dict(
        started_utc=started,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        passed=not problems,
        problems=problems,
        artifact_hash_checks=hash_checks,
        runs_recomputed_from_predictions=len(recomputed),
        runs_recomputed_equal_to_saved=sum(r["recomputed_equals_saved"] for r in recomputed),
        group_checks=checks,
    )
    write_json(OUT / "sanity_checks.json", result)
    print(json.dumps(dict(passed=result["passed"], problems=problems, checks=checks), indent=1))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
