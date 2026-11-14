# ruff: noqa: E501
"""How often would a study with few seeds reach the wrong conclusion? Exploratory, written after the
ten-seed results were seen; not part of any registered family.

Uses completed runs only and resamples them; no new training.

same-recipe: two disjoint groups of k runs of ONE recipe. There is no true difference, so any
  "significant" image-level result is a false positive caused by retraining alone. The test is the
  registered one: Wilcoxon signed-rank on per-unit differences of k-run means.
contrast: k runs of each of two arms, compared with the estimate from all runs.

Output: revision/tables/families/seed_subset.csv
"""

from __future__ import annotations

import csv
import itertools
import random
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis/revision"))
sys.path.insert(0, str(ROOT))
import expansion_statistics as ex  # noqa: E402

METRICS = ["boundary_f1", "dsc", "hd95"]
SEEDS = list(range(1, 11))
ARMS = ["unet", "decoder", "full", "gsl"]
SAMPLES = 2000
OUT = ROOT / "revision/tables/families/seed_subset.csv"


def matrix(arm: str, metric: str, cache: dict) -> np.ndarray:
    """Runs x units, oriented so that larger is better."""
    if arm not in cache:
        cache[arm] = [ex.unit_values(ex.isic_path(arm, seed), "isic2018") for seed in SEEDS]
    runs = cache[arm]
    units = sorted(set.intersection(*[set(r[metric]) for r in runs]))
    sign = -1.0 if metric in ex.fs.DISTANCE else 1.0
    return sign * np.array([[r[metric][u] for u in units] for r in runs])


def common(metric: str, cache: dict, first: str, second: str):
    units = sorted(set.intersection(*[set(r[metric]) for r in cache[first] + cache[second]]))
    sign = -1.0 if metric in ex.fs.DISTANCE else 1.0
    pick = lambda arm: sign * np.array([[r[metric][u] for u in units] for r in cache[arm]])  # noqa: E731
    return pick(first), pick(second)


def p_value(diff: np.ndarray) -> float:
    diff = diff[diff != 0]
    return float(stats.wilcoxon(diff).pvalue)


def summarise(deltas, ps, sesoi, truth=None) -> dict:
    d = np.array(deltas)
    row = dict(
        subsets=len(d),
        mean_abs_delta=float(np.abs(d).mean()),
        low=float(np.quantile(d, 0.025)),
        high=float(np.quantile(d, 0.975)),
        share_beyond_sesoi=float((np.abs(d) > sesoi).mean()),
        share_p05=float((np.array(ps) < 0.05).mean()),
    )
    if truth is not None:
        same = np.sign(d) == np.sign(truth)
        row.update(
            all_run_delta=truth,
            share_same_sign=float(same.mean()),
            share_opposite_beyond_sesoi=float((~same & (np.abs(d) > sesoi)).mean()),
            share_opposite_p05=float((~same & (np.array(ps) < 0.05)).mean()),
        )
    return row


def main() -> None:
    random.seed(20261107)
    cache: dict = {}
    rows = []
    for metric in METRICS:
        sesoi = ex.SESOI[metric]
        for arm in ARMS:
            y = matrix(arm, metric, cache)
            for k in (3, 5):
                splits = []
                for group in itertools.combinations(range(10), k):
                    rest = [i for i in range(10) if i not in group]
                    for other in itertools.combinations(rest, k):
                        if group < other:
                            splits.append((group, other))
                if len(splits) > SAMPLES:
                    splits = random.sample(splits, SAMPLES)
                deltas, ps = [], []
                for g, o in splits:
                    diff = y[list(g)].mean(0) - y[list(o)].mean(0)
                    deltas.append(float(diff.mean()))
                    ps.append(p_value(diff))
                rows.append(
                    dict(kind="same-recipe", first=arm, second=arm, k=k, metric=metric)
                    | summarise(deltas, ps, sesoi)
                )
                print(rows[-1], flush=True)
        for first, second in ex.PAIRS:
            ya, yb = common(metric, cache, first, second)
            truth = float(ya.mean() - yb.mean())
            for k in (3, 5, 7):
                combos = list(itertools.combinations(range(10), k))
                pairs = [(random.choice(combos), random.choice(combos)) for _ in range(SAMPLES)]
                deltas, ps = [], []
                for g, o in pairs:
                    diff = ya[list(g)].mean(0) - yb[list(o)].mean(0)
                    deltas.append(float(diff.mean()))
                    ps.append(p_value(diff))
                rows.append(
                    dict(kind="contrast", first=first, second=second, k=k, metric=metric)
                    | summarise(deltas, ps, sesoi, truth)
                )
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with OUT.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", OUT, len(rows))


if __name__ == "__main__":
    main()
