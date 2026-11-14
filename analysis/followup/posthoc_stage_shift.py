"""POST HOC: why D's ACDC Boundary F1 mean is lower in Stage 3 (0.7797) than in Stage 1 (0.7993).

Registration section 10.3. Uses only existing evaluations; no model is run.
2 x 2 layout for D (and, for context, D+KD-band):
- seed set / trainer: Stage 1-2 seeds 1-3 (scripts/revision/train_run.py for D; train_mc.py for
  D+KD-band) versus Stage 3 seeds 201-217 (train_mc.py through the launcher);
- patients: the 20 internal test patients (third-party preprocessing) versus the 50 official
  test patients 101-150 (reconstructed preprocessing, Supplementary S18).
Reports cell means of run means, run SDs, the seed-set contrast on fixed patients (frozen
run-level estimator) and the patient-set shift within the same runs (mean of per-run differences
with a t interval over runs; patients differ, so this is descriptive).

    PYTHONPATH=src:. .venv/bin/python analysis/followup/posthoc_stage_shift.py
Outputs: revision/followup/results/posthoc_stage_shift.{json,csv}
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis/revision"))
import expansion_statistics as ex  # noqa: E402
import family_statistics as fs  # noqa: E402

RUNS = ROOT / "results/revision_runs"
EVAL = ROOT / "results/acdc_confirm_eval"
OUT = ROOT / "revision/followup/results"
STAGE3 = range(201, 218)
EARLY = (1, 2, 3)
EARLY_NAME = {
    "decoder": "acdc_m2_aux_boundary_only_unweighted_seed{}",
    "kd_band": "acdc_kd_band_seed{}",
}


def path(arm: str, seeds: str, patients: str, seed: int) -> Path:
    if seeds == "stage3":
        return EVAL / f"acdc_confirm_{arm}_seed{seed}" / f"per_case_{patients}.csv"
    name = EARLY_NAME[arm].format(seed)
    if patients == "official_test":
        return EVAL / name / "per_case_official_test.csv"
    return RUNS / name / "per_case_test.csv"


def run_means(values: list[dict], metric: str) -> np.ndarray:
    return np.array([np.mean([v for v in r[metric].values() if np.isfinite(v)]) for r in values])


def slice_mean(csv_path: Path, metric: str) -> float:
    """Table 4 aggregation: class macro per slice, then mean over slices (not over patients)."""
    per_slice: dict[str, list[float]] = {}
    with csv_path.open() as stream:
        for r in csv.DictReader(stream):
            v = fs._value(r, metric)
            if np.isfinite(v):
                per_slice.setdefault(r["id"], []).append(v)
    return float(np.mean([np.mean(v) for v in per_slice.values()]))


def main() -> None:
    cells, rows = {}, []
    for arm in ("decoder", "kd_band"):
        for seeds, seed_list in (("early", EARLY), ("stage3", STAGE3)):
            for patients in ("internal_test", "official_test"):
                cells[(arm, seeds, patients)] = [
                    ex.unit_values(path(arm, seeds, patients, s), "acdc") for s in seed_list
                ]
    for metric in ("boundary_f1", "dsc"):
        for (arm, seeds, patients), values in cells.items():
            m = run_means(values, metric)
            seed_list = EARLY if seeds == "early" else STAGE3
            sl = np.array([slice_mean(path(arm, seeds, patients, s), metric) for s in seed_list])
            rows.append(
                dict(
                    kind="cell, slice-level aggregation (as Table 4)",
                    metric=metric,
                    arm=arm,
                    seeds=seeds,
                    patients=patients,
                    runs=sl.size,
                    mean=float(sl.mean()),
                    sd=float(sl.std(ddof=1)),
                )
            )
            rows.append(
                dict(
                    kind="cell, patient-level aggregation (Stage 3 estimator)",
                    metric=metric,
                    arm=arm,
                    seeds=seeds,
                    patients=patients,
                    runs=m.size,
                    mean=float(m.mean()),
                    sd=float(m.std(ddof=1)),
                )
            )
        for arm in ("decoder", "kd_band"):
            for patients in ("internal_test", "official_test"):
                c = ex.contrast(
                    cells[(arm, "stage3", patients)], cells[(arm, "early", patients)], metric
                )
                rows.append(
                    dict(
                        kind="seed set (stage3 - early), same patients",
                        metric=metric,
                        arm=arm,
                        patients=patients,
                        delta=c["delta"],
                        low=float(c["low"]),
                        high=float(c["high"]),
                        p=c["p"],
                    )
                )
            for seeds in ("early", "stage3"):
                d = run_means(cells[(arm, seeds, "official_test")], metric) - run_means(
                    cells[(arm, seeds, "internal_test")], metric
                )
                t = stats.t.ppf(0.975, d.size - 1)
                half = t * d.std(ddof=1) / np.sqrt(d.size)
                rows.append(
                    dict(
                        kind="patient set (official - internal), same runs",
                        metric=metric,
                        arm=arm,
                        seeds=seeds,
                        delta=float(d.mean()),
                        low=float(d.mean() - half),
                        high=float(d.mean() + half),
                        runs=d.size,
                    )
                )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "posthoc_stage_shift.json").write_text(
        json.dumps(dict(label="post hoc", rows=rows), indent=2) + "\n"
    )
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / "posthoc_stage_shift.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        if r["metric"] == "boundary_f1":
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
