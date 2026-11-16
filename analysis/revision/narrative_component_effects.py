# ruff: noqa: E501
"""Component main effects with run and test-unit uncertainty. Exploratory and post hoc.

The seven retrained ablation arms contain four components (decoder, surface loss, distillation,
gate). Fitting one additive model to the 21 runs per dataset uses all arms at once, instead of one
pairwise contrast at a time. Each coefficient is a fixed linear combination of the run values, so
it can also be computed per test unit. Two uncertainty sources are then combined:
  run:  residual run-to-run variance (16 residual df), test set held fixed;
  unit: variance over test units of the per-unit coefficient (paired across arms), runs held fixed.
Total SE = sqrt(se_run^2 + se_unit^2). Intervals use t(0.975, 16).

Classification against the SESOI of prereg_revision.md:
  relevant gain/harm  interval wholly beyond +/-SESOI
  gain/harm, size unclear  interval excludes zero but overlaps the threshold
  no relevant effect  interval wholly inside +/-SESOI (evidence against a relevant effect)
  undetermined        interval includes zero and extends beyond the threshold (no conclusion)

Assumes additivity; the design has no decoder + surface cell, so interactions are not estimable.
Pooled rows are inverse-variance fixed-effect estimates over the three datasets.

Output: results/tables/families/component_effects.csv
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis/revision"))
import family_statistics as fs  # noqa: E402

DATASETS = ["isic2018", "acdc", "synapse"]
METRICS = fs.METRICS
SESOI = fs.SESOI
COMPONENTS = ["decoder", "surface", "distill", "gate"]
DESIGN = {
    "M0": (0, 0, 0, 0),
    "M1": (0, 1, 0, 0),
    "M2": (1, 0, 0, 0),
    "M3": (1, 0, 1, 1),
    "M4": (1, 1, 0, 1),
    "M5": (1, 1, 1, 0),
    "M6": (1, 1, 1, 1),
}
OUT = ROOT / "results/tables/families/component_effects.csv"


def unit_values(arm: str, dataset: str, seed: int) -> dict[str, dict[str, float]]:
    """metric -> unit -> value for one run (label macro per sample, then unit mean)."""
    samples: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with fs.per_case_path(arm, dataset, seed).open() as stream:
        for row in csv.DictReader(stream):
            for metric in METRICS:
                samples[row["id"]][metric].append(fs._value(row, metric))
    out: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        units: dict[str, list[float]] = defaultdict(list)
        for sample, values in samples.items():
            arr = np.array(values[metric], float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                units[fs.cluster_of(dataset, sample)].append(float(arr.mean()))
        out[metric] = {u: float(np.mean(v)) for u, v in units.items()}
    return out


def classify(low: float, high: float, sesoi: float) -> str:
    if low > sesoi:
        return "relevant gain"
    if high < -sesoi:
        return "relevant harm"
    if -sesoi < low < high < sesoi:
        return "no relevant effect"
    if low > 0:
        return "gain, size unclear"
    if high < 0:
        return "harm, size unclear"
    return "undetermined"


def main() -> None:
    runs = [(arm, seed) for arm in DESIGN for seed in fs.SEEDS]
    x = np.array([(1, *DESIGN[arm]) for arm, _ in runs], float)
    pinv = np.linalg.inv(x.T @ x) @ x.T
    df = len(runs) - x.shape[1]
    t = stats.t.ppf(0.975, df)
    rows = []
    for dataset in DATASETS:
        values = {key: unit_values(key[0], dataset, key[1]) for key in runs}
        for metric in METRICS:
            sign = -1.0 if metric in fs.DISTANCE else 1.0
            units = sorted(set.intersection(*[set(values[k][metric]) for k in runs]))
            y = sign * np.array([[values[k][metric][u] for u in units] for k in runs])
            run_means = y.mean(1)
            beta = pinv @ run_means
            resid = run_means - x @ beta
            cov = (resid @ resid / df) * np.linalg.inv(x.T @ x)
            per_unit = pinv @ y
            for i, name in enumerate(COMPONENTS, start=1):
                se_run = math.sqrt(cov[i, i])
                se_unit = per_unit[i].std(ddof=1) / math.sqrt(len(units))
                se = math.hypot(se_run, se_unit)
                low, high = beta[i] - t * se, beta[i] + t * se
                rows.append(
                    dict(
                        dataset=dataset,
                        metric=metric,
                        component=name,
                        estimate=beta[i],
                        se_run=se_run,
                        se_unit=se_unit,
                        low=low,
                        high=high,
                        sesoi=SESOI[metric],
                        classification=classify(low, high, SESOI[metric]),
                    )
                )
    pooled = []
    for metric in METRICS:
        for name in COMPONENTS:
            part = [r for r in rows if r["metric"] == metric and r["component"] == name]
            est = np.array([r["estimate"] for r in part])
            var = np.array([r["se_run"] ** 2 + r["se_unit"] ** 2 for r in part])
            w = 1 / var
            mean = float((w * est).sum() / w.sum())
            se = math.sqrt(1 / w.sum())
            q = float((w * (est - mean) ** 2).sum())
            low, high = mean - 1.96 * se, mean + 1.96 * se
            pooled.append(
                dict(
                    dataset=f"pooled (Q p={1 - stats.chi2.cdf(q, 2):.2f})",
                    metric=metric,
                    component=name,
                    estimate=mean,
                    se_run=float("nan"),
                    se_unit=float("nan"),
                    low=low,
                    high=high,
                    sesoi=SESOI[metric],
                    classification=classify(low, high, SESOI[metric]),
                )
            )
    with OUT.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows + pooled)
    for name in COMPONENTS:
        counts: dict[str, int] = defaultdict(int)
        for r in rows:
            if r["component"] == name:
                counts[r["classification"]] += 1
        print(name, dict(counts))
    for r in rows + pooled:
        if r["metric"] in ("dsc", "boundary_f1", "hd95"):
            print(
                f"{r['dataset'][:22]:22s} {r['metric']:12s} {r['component']:8s} {r['estimate']:+.4f} [{r['low']:+.4f}, {r['high']:+.4f}] {r['classification']}"
            )


if __name__ == "__main__":
    main()
