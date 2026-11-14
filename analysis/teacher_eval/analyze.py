"""Descriptive teacher-vs-student comparison and the pre-specified reading (PROTOCOL.md).

Reads only per-case files: the new teacher evaluations under results/teacher_eval/ and the
existing one-time student evaluations (read only). Aggregation is the unchanged
expansion_statistics.unit_values (patient = unit); per-structure values use
analysis/kd3/unified_analysis.label_values. No significance test is computed.

Outputs (results/teacher_eval/): per_patient_teachers.csv, per_patient_paired_bf1.csv,
student_run_means.csv, summary.json.

    PYTHONPATH=src:. .venv/bin/python analysis/teacher_eval/analyze.py
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _path in (REPO, REPO / "src", REPO / "analysis/revision", HERE):
    sys.path.insert(0, str(_path))
import expansion_statistics as ex  # noqa: E402
from common import (  # noqa: E402
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    GROUPS,
    OUT,
    ROOT,
    SPLIT,
    STRUCTURES,
    TEACHERS,
    case_file,
    run_names,
    teacher_dir,
)

from analysis.kd3.unified_analysis import label_values  # noqa: E402
from scripts.kd.common import sha, write_json  # noqa: E402

METRICS = ex.METRICS
OUTCOME_TEXT = {
    "A": (
        "Augmented students outperform the teacher, consistent with the disappearance of the "
        "distillation benefit."
    ),
    "B": "This explanation is not supported.",
    "C": "Inconclusive.",
}


def mean(values: dict) -> float:
    return float(np.mean(list(values.values())))


def profile(path: Path) -> dict:
    """metric -> patient -> value, plus bf1_<structure> -> patient -> value."""
    values = ex.unit_values(path, "acdc")
    for label, name in STRUCTURES.items():
        values[f"bf1_{name}"] = label_values(path, "acdc", label)["boundary_f1"]
    return values


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_ci(diff: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)  # same resamples for every comparison
    draws = diff[rng.integers(0, diff.size, size=(BOOTSTRAP_DRAWS, diff.size))].mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def main() -> None:
    sanity = json.loads((OUT / "sanity_checks.json").read_text())
    if not sanity["passed"]:
        raise RuntimeError("sanity checks failed; no teacher result is reported")
    record = json.loads((OUT / "evaluation_record.json").read_text())
    for teacher, entry in record["teachers"].items():
        for name, digest in entry["artifacts"].items():
            if sha(teacher_dir(teacher) / name) != digest:
                raise ValueError(f"{teacher}/{name} changed after evaluation")
    keys = list(METRICS) + [f"bf1_{s}" for s in STRUCTURES.values()]

    teachers, teacher_rows = {}, []
    for teacher in TEACHERS:
        values = profile(teacher_dir(teacher) / f"per_case_{SPLIT}.csv")
        if len(values["boundary_f1"]) != 50:
            raise ValueError(f"{teacher}: {len(values['boundary_f1'])} patients")
        teachers[teacher] = values
        for patient in sorted(values["boundary_f1"]):
            teacher_rows.append(
                dict(teacher=teacher, patient=patient, **{k: values[k].get(patient) for k in keys})
            )
    write_csv(OUT / "per_patient_teachers.csv", teacher_rows)

    groups, run_rows = {}, []
    for group in GROUPS:
        runs = {run: profile(case_file(group, run)) for run in run_names(group)}
        groups[group] = runs
        for run, values in runs.items():
            run_rows.append(dict(group=group, run=run, **{k: mean(values[k]) for k in keys}))
    write_csv(OUT / "student_run_means.csv", run_rows)

    summary_teachers, comparisons, paired_rows = {}, {}, []
    for teacher, values in teachers.items():
        entry = record["teachers"][teacher]
        summary_teachers[teacher] = dict(
            run=entry["run"],
            best_sha256=entry["identity"]["best_sha256"],
            parameters=entry["parameters"],
            input_processing=entry["input_processing"],
            patient_level={k: mean(values[k]) for k in keys},
            patients_with_defined_value={k: len(values[k]) for k in keys},
            row_level_metrics_from_evaluate=entry["row_level_metrics_from_evaluate"],
            reproduction_of_prior_evaluation=entry["reproduction_of_prior_evaluation"],
        )
        teacher_bf1 = mean(values["boundary_f1"])
        patients = sorted(values["boundary_f1"])
        comparisons[teacher] = {}
        for group, runs in groups.items():
            run_means = np.array([mean(v["boundary_f1"]) for v in runs.values()])
            matrix = np.array([[v["boundary_f1"][p] for p in patients] for v in runs.values()])
            group_patient = matrix.mean(axis=0)
            diff = np.array([values["boundary_f1"][p] for p in patients]) - group_patient
            low, high = bootstrap_ci(diff)
            comparisons[teacher][group] = dict(
                runs=int(run_means.size),
                run_mean_bf1_mean=float(run_means.mean()),
                run_mean_bf1_min=float(run_means.min()),
                run_mean_bf1_max=float(run_means.max()),
                teacher_bf1=teacher_bf1,
                runs_exceeding_teacher=int((run_means > teacher_bf1).sum()),
                paired_difference_teacher_minus_group=float(diff.mean()),
                bootstrap_95ci=[low, high],
                patients=len(patients),
                group_mean_all_metrics={
                    k: float(np.mean([mean(v[k]) for v in runs.values()])) for k in keys
                },
            )
            for i, patient in enumerate(patients):
                paired_rows.append(
                    dict(
                        teacher=teacher,
                        group=group,
                        patient=patient,
                        teacher_bf1=values["boundary_f1"][patient],
                        group_mean_bf1=float(group_patient[i]),
                        difference=float(diff[i]),
                    )
                )
    write_csv(OUT / "per_patient_paired_bf1.csv", paired_rows)

    ext = comparisons["external"]["D_aug"]
    if ext["teacher_bf1"] >= ext["run_mean_bf1_mean"]:
        outcome = "B"
    elif ext["bootstrap_95ci"][1] < 0:
        outcome = "A"
    else:
        outcome = "C"
    summary = dict(
        label="post hoc, descriptive, inference only; patients 101-150 previously examined",
        created_utc=datetime.now(timezone.utc).isoformat(),
        protocol="results/teacher_eval/PROTOCOL.md",
        protocol_sha256=sha(OUT / "PROTOCOL.md"),
        aggregation=(
            "per slice: mean over foreground classes excluding both-empty pairs and undefined "
            "HD95/ASSD; per patient: mean over slices; teacher/run: mean over 50 patients; "
            "group: mean of run means (expansion_statistics.unit_values)"
        ),
        bootstrap=dict(
            draws=BOOTSTRAP_DRAWS,
            seed=BOOTSTRAP_SEED,
            method="percentile; patients resampled with replacement; same draws per comparison",
        ),
        sanity_checks=dict(
            passed=sanity["passed"],
            runs_recomputed_from_predictions=sanity["runs_recomputed_from_predictions"],
            runs_recomputed_equal_to_saved=sanity["runs_recomputed_equal_to_saved"],
            artifact_hash_checks=sanity["artifact_hash_checks"],
            group_checks=sanity["group_checks"],
        ),
        teachers=summary_teachers,
        comparisons=comparisons,
        interpretation=dict(
            rule_applied_to="external teacher vs D_aug (PROTOCOL.md, verbatim from the brief)",
            external_teacher_bf1=ext["teacher_bf1"],
            d_aug_mean_bf1=ext["run_mean_bf1_mean"],
            paired_difference=ext["paired_difference_teacher_minus_group"],
            bootstrap_95ci=ext["bootstrap_95ci"],
            outcome=outcome,
            conclusion=OUTCOME_TEXT[outcome],
            note="Descriptive only; no significance test; single frozen teacher per type.",
        ),
        source_hashes={
            str(p.relative_to(ROOT)): sha(p)
            for p in sorted((ROOT / "analysis/teacher_eval").glob("*.py"))
        },
    )
    write_json(OUT / "summary.json", summary)
    print(json.dumps(dict(teachers=summary_teachers, interpretation=summary["interpretation"])))


if __name__ == "__main__":
    main()
