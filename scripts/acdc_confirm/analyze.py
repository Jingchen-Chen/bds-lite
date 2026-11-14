"""Pre-registered analysis (section 7). Deterministic; reads only the one-time evaluation outputs.

Primary: D+KD-band - decoder-only, test Boundary F1 on PRIMARY_SET, replication seeds 201-217,
frozen analysis/revision/expansion_statistics.py::contrast with unit_values (patient = unit),
two-sided alpha 0.05; decision per section 11. Everything else is descriptive.

    .venv/bin/python scripts/acdc_confirm/analyze.py
Outputs: revision/acdc_confirm/results/{results.json, contrasts.csv, per_seed.csv, per_class.csv}.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis/revision"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import expansion_statistics as ex  # noqa: E402
import family_statistics as fs  # noqa: E402

from analysis.kd3.unified_analysis import label_values  # noqa: E402
from scripts.acdc_confirm.common import (  # noqa: E402
    EVAL,
    EXPLORATORY,
    PRIMARY_SET,
    RUNS,
    SEEDS,
    WORKSPACE,
    failed_runs,
    run_name,
)
from scripts.kd.common import sha  # noqa: E402

OUT = WORKSPACE / "results"
METRICS = ex.METRICS  # dsc, iou, boundary_f1, hd95, assd
SESOI = ex.SESOI
STRUCTURES = {1: "RV", 2: "myocardium", 3: "LV"}
ALPHA = 0.05
ARMS = ("kd_band", "decoder")
# Section 6 inputs (exploratory seeds 1-3, internal split): pooled variance of run means (4 df)
# and the per-patient difference variance. Achieved power uses assumption A, fixed SE_unit.
POOLED_RUN_VARIANCE = 4.9400e-05
PER_PATIENT_DIFFERENCE_VARIANCE = 3.5053e-04


def achieved_power(n_a: int, n_b: int, units: int, reps: int = 200000) -> dict:
    """Section 6 simulation at the realised numbers of completed runs per arm."""
    rng = np.random.default_rng(20261107)
    se_unit = math.sqrt(PER_PATIENT_DIFFERENCE_VARIANCE / units)
    out = dict(n_kd_band=n_a, n_decoder=n_b, units=units, assumption="A, fixed SE_unit")
    for bound in (0.8, 0.9):
        var = POOLED_RUN_VARIANCE * 4 / stats.chi2.ppf(1 - bound, 4)
        for delta in (0.015, 0.024):
            ra = rng.normal(delta, math.sqrt(var), (reps, n_a))
            rb = rng.normal(0.0, math.sqrt(var), (reps, n_b))
            va, vb = ra.var(1, ddof=1) / n_a, rb.var(1, ddof=1) / n_b
            d = ra.mean(1) - rb.mean(1)
            se = np.sqrt(va + vb + se_unit**2)
            df = (va + vb) ** 2 / (va**2 / (n_a - 1) + vb**2 / (n_b - 1))
            p = 2 * stats.t.sf(np.abs(d / se), df)
            out[f"delta_{delta}_sd_bound_{int(bound * 100)}"] = float(
                np.mean((p < ALPHA) & (d > 0))
            )
    return out


def new_path(arm: str, seed: int, split: str) -> Path:
    if split == "val":
        return RUNS / run_name(arm, seed) / "per_case_val.csv"
    return EVAL / run_name(arm, seed) / f"per_case_{split}.csv"


def exploratory_path(arm: str, index: int, split: str) -> Path:
    name = list(EXPLORATORY[arm])[index]
    if split == "official_test":
        return EVAL / name / "per_case_official_test.csv"
    return RUNS / name / ("per_case_val.csv" if split == "val" else "per_case_test.csv")


def decision(c: dict) -> dict:
    if c["low"] > 0:
        outcome = "success"
        size = "above the smallest effect of interest" if c["low"] > SESOI["boundary_f1"] else (
            "of unclear size"
        )  # fmt: skip
    elif c["high"] < SESOI["boundary_f1"]:
        outcome, size = "failure", None
    else:
        outcome, size = "inconclusive", None
    return dict(outcome=outcome, size=size, alpha=ALPHA, two_sided=True)


def paired(a: list[dict], b: list[dict], metric: str) -> dict:
    """Seed-paired sensitivity: SD of per-seed differences, plus the same unit-level SE."""
    sign = -1.0 if metric in fs.DISTANCE else 1.0
    units = sorted(set.intersection(*[set(r[metric]) for r in a + b]))
    ya = sign * np.array([[r[metric][u] for u in units] for r in a])
    yb = sign * np.array([[r[metric][u] for u in units] for r in b])
    d = ya.mean(1) - yb.mean(1)
    per_unit = ya.mean(0) - yb.mean(0)
    se_unit = float(per_unit.std(ddof=1) / math.sqrt(len(units)))
    se = math.hypot(float(d.std(ddof=1) / math.sqrt(d.size)), se_unit)
    df = d.size - 1
    t = stats.t.ppf(0.975, df)
    delta = float(d.mean())
    return dict(
        delta=delta,
        low=delta - t * se,
        high=delta + t * se,
        se=se,
        se_unit=se_unit,
        df=df,
        p=float(2 * stats.t.sf(abs(delta / se), df)),
        n_pairs=int(d.size),
        seed_correlation=float(np.corrcoef(ya.mean(1), yb.mean(1))[0, 1]),
    )


def row(analysis: str, split: str, metric: str, c: dict, **extra) -> dict:
    return dict(
        analysis=analysis,
        split=split,
        metric=metric,
        delta=c["delta"],
        low=c["low"],
        high=c["high"],
        p=c["p"],
        se_run=c.get("se_run"),
        se_unit=c["se_unit"],
        df=c["df"],
        n_a=c.get("n_a", c.get("n_pairs")),
        n_b=c.get("n_b", c.get("n_pairs")),
        arm_a_mean=c.get("arm_a_mean"),
        arm_b_mean=c.get("arm_b_mean"),
        arm_a_sd=c.get("arm_a_sd"),
        arm_b_sd=c.get("arm_b_sd"),
        reading=ex.classify(c["low"], c["high"], SESOI[metric]),
        **extra,
    )


def main() -> None:
    record = json.loads((EVAL / "evaluation_record.json").read_text())
    for r in record["runs"]:
        for name, digest in r["artifacts"].items():
            if sha(EVAL / r["run"] / name) != digest:
                raise ValueError(f"evaluation output changed: {r['run']}/{name}")
    primary_split = PRIMARY_SET
    failed = failed_runs()
    done = {arm: [s for s in SEEDS if run_name(arm, s) not in failed] for arm in ARMS}
    pairs = [s for s in SEEDS if s in done["kd_band"] and s in done["decoder"]]
    values = {
        split: {
            arm: {s: ex.unit_values(new_path(arm, s, split), "acdc") for s in done[arm]}
            for arm in ARMS
        }
        for split in {primary_split, "internal_test", "val"}
    }
    new = {split: {arm: list(v[arm].values()) for arm in ARMS} for split, v in values.items()}
    paired_new = {
        split: {arm: [v[arm][s] for s in pairs] for arm in ARMS} for split, v in values.items()
    }
    contrasts, per_seed, per_class = [], [], []
    primary = ex.contrast(
        new[primary_split]["kd_band"], new[primary_split]["decoder"], "boundary_f1"
    )
    units = len(
        set.intersection(*[set(r["boundary_f1"]) for a in ARMS for r in new[primary_split][a]])
    )
    result = dict(
        primary_set=primary_split,
        primary=dict(contrast=primary, decision=decision(primary), seeds=done),
        failed_runs=list(failed.values()),
        achieved_power=achieved_power(len(done["kd_band"]), len(done["decoder"]), units),
    )
    # (d) all metrics on the primary set; (b) replication seeds on the internal split; validation
    splits = [(primary_split, "replication seeds, primary set")]
    if primary_split != "internal_test":
        splits.append(("internal_test", "replication seeds, internal 20-patient split (seen)"))
    splits.append(("val", "replication seeds, validation (selection set)"))
    for split, label in splits:
        for metric in METRICS:
            c = ex.contrast(new[split]["kd_band"], new[split]["decoder"], metric)
            contrasts.append(
                row(
                    label,
                    split,
                    metric,
                    c,
                    primary=split == primary_split and metric == "boundary_f1",
                )
            )
            p = paired(paired_new[split]["kd_band"], paired_new[split]["decoder"], metric)
            contrasts.append(row(f"{label}; seed-paired sensitivity", split, metric, p))
    # (a) exploratory models on the held-out set; combined exploratory + replication (descriptive)
    if primary_split == "official_test":
        expl = {
            arm: [
                ex.unit_values(exploratory_path(arm, i, "official_test"), "acdc") for i in range(3)
            ]
            for arm in ("kd_band", "decoder")
        }
        for metric in METRICS:
            c = ex.contrast(expl["kd_band"], expl["decoder"], metric)
            contrasts.append(
                row("exploratory seeds 1-3, official test", "official_test", metric, c)
            )
            both = {a: expl[a] + new["official_test"][a] for a in expl}
            c = ex.contrast(both["kd_band"], both["decoder"], metric)
            contrasts.append(
                row("exploratory + replication seeds, official test", "official_test", metric, c)
            )
    else:
        expl = {
            arm: [
                ex.unit_values(exploratory_path(arm, i, "internal_test"), "acdc") for i in range(3)
            ]
            for arm in ("kd_band", "decoder")
        }
        for metric in METRICS:
            both = {a: expl[a] + new["internal_test"][a] for a in expl}
            c = ex.contrast(both["kd_band"], both["decoder"], metric)
            contrasts.append(
                row("exploratory + replication seeds, internal split", "internal_test", metric, c)
            )
    # per-structure, primary set
    for label, structure in STRUCTURES.items():
        kd = [
            label_values(new_path("kd_band", s, primary_split), "acdc", label)
            for s in done["kd_band"]
        ]
        de = [
            label_values(new_path("decoder", s, primary_split), "acdc", label)
            for s in done["decoder"]
        ]
        for metric in ["dsc", "boundary_f1", "hd95"]:
            c = ex.contrast(kd, de, metric)
            per_class.append(
                row(f"per-structure {structure}", primary_split, metric, c, structure=structure)
            )
    # per-seed run means and consistency, primary set
    for arm in ("kd_band", "decoder"):
        for seed, run in values[primary_split][arm].items():
            for metric in METRICS:
                v = [x for x in run[metric].values() if math.isfinite(x)]
                per_seed.append(
                    dict(
                        arm=arm,
                        seed=seed,
                        split=primary_split,
                        metric=metric,
                        value=float(np.mean(v)),
                    )
                )
    bf1 = {
        a: [r["value"] for r in per_seed if r["arm"] == a and r["metric"] == "boundary_f1"]
        for a in ("kd_band", "decoder")
    }
    result["consistency_boundary_f1"] = dict(
        kd_runs_above_decoder_mean=sum(v > np.mean(bf1["decoder"]) for v in bf1["kd_band"]),
        min_kd_minus_max_decoder=min(bf1["kd_band"]) - max(bf1["decoder"]),
        max_kd_minus_min_decoder=max(bf1["kd_band"]) - min(bf1["decoder"]),
    )

    def epoch_hours(arm: str) -> float:
        total = 0.0
        for s in done[arm]:
            history = json.loads((RUNS / run_name(arm, s) / "history.json").read_text())
            total += sum(h["seconds"] for h in history)
        return total / 3600

    result["training_epoch_hours"] = {arm: epoch_hours(arm) for arm in ("kd_band", "decoder")}
    result["contrasts"] = contrasts
    result["per_class"] = per_class
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    for name, rows in [("contrasts", contrasts), ("per_class", per_class), ("per_seed", per_seed)]:
        fields = sorted(
            {k for r in rows for k in r},
            key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99,
        )
        with (OUT / f"{name}.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(result["primary"], indent=2, default=float))


if __name__ == "__main__":
    main()
