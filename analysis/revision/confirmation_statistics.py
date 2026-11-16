# ruff: noqa: E501
"""Frozen analysis for protocols/confirmation/CONFIRMATION_PROTOCOL.md. Committed before any confirmation run.

Reuses the expansion estimator (expansion_statistics.contrast). Positive favours the first arm.
Outputs in results/tables/families/: confirmation_contrasts.csv, confirmation_pooled.csv.
--dry-run <dir> treats seeds 1-3 as the fresh seeds and writes to <dir>. It exists only to test this
code on data already known.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis/revision"))
sys.path.insert(0, str(ROOT))
import expansion_statistics as ex  # noqa: E402
import family_statistics as fs  # noqa: E402

from analysis.revision.confirmation_arms import ARM_NAMES  # noqa: E402

METRICS = ex.METRICS
DATASETS = ["isic2018", "acdc", "synapse"]
PRIMARY_METRIC = "boundary_f1"
FRESH = {"isic2018": list(range(11, 18)), "acdc": list(range(4, 11)), "synapse": list(range(4, 11))}
ALL_SEEDS = {
    ("isic2018", "unet"): list(range(1, 18)),
    ("isic2018", "decoder"): list(range(1, 18)),
    ("isic2018", "full"): list(range(1, 11)),
    ("isic2018", "gsl"): list(range(1, 11)),
}
CELL = {"unet": "M0", "decoder": "M2", "full": "M6", "gsl": "GSL"}
PAIRS = ex.PAIRS
ISIC_PAIRS_FRESH = [("decoder", "unet")]
POOLED_SECONDARY = [("full", "decoder"), ("full", "unet"), ("gsl", "decoder")]


def run_path(dataset: str, arm: str, seed: int) -> Path:
    if dataset == "isic2018" and seed <= 10:
        return ex.isic_path(arm, seed)
    if seed <= 3:
        return (
            ex.cell_path(CELL[arm], dataset, seed, False)
            if arm != "gsl"
            else fs.per_case_path("GSL", dataset, seed)
        )
    return ex.RUNS / f"{dataset}_{ARM_NAMES[dataset][arm]}_seed{seed}/per_case_test.csv"


def load(dataset: str, arm: str, seeds: list[int]) -> list[dict]:
    return [ex.unit_values(run_path(dataset, arm, seed), dataset) for seed in seeds]


def pool(items: list[dict]) -> dict:
    """Inverse-variance fixed effect, Cochran Q, and DerSimonian-Laird random effects."""
    d = np.array([i["delta"] for i in items])
    se = np.array([math.hypot(i["se_run"], i["se_unit"]) for i in items])
    w = 1 / se**2
    est = float((w * d).sum() / w.sum())
    s = float(w.sum() ** -0.5)
    q = float((w * (d - est) ** 2).sum())
    k = len(items)
    tau2 = max(0.0, (q - (k - 1)) / (w.sum() - (w**2).sum() / w.sum()))
    wr = 1 / (se**2 + tau2)
    re_est = float((wr * d).sum() / wr.sum())
    re_se = float(wr.sum() ** -0.5)
    return dict(
        estimate=est,
        se=s,
        low=est - 1.96 * s,
        high=est + 1.96 * s,
        p=float(2 * stats.norm.sf(abs(est / s))),
        q=q,
        q_p=float(stats.chi2.sf(q, k - 1)) if k > 1 else float("nan"),
        re_estimate=re_est,
        re_low=re_est - 1.96 * re_se,
        re_high=re_est + 1.96 * re_se,
    )


def write(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", type=Path)
    args = parser.parse_args()
    dry = args.dry_run is not None
    out = args.dry_run if dry else ROOT / "results/tables/families"
    out.mkdir(parents=True, exist_ok=True)

    def fresh(dataset: str) -> list[int]:
        return [1, 2, 3] if dry else FRESH[dataset]

    def every(dataset: str, arm: str) -> list[int]:
        if dry:
            return [1, 2, 3]
        return ALL_SEEDS.get((dataset, arm), list(range(1, 11)))

    cache: dict = {}

    def runs(dataset: str, arm: str, seeds: list[int]) -> list[dict]:
        key = (dataset, arm, tuple(seeds))
        if key not in cache:
            cache[key] = load(dataset, arm, seeds)
        return cache[key]

    def compare(dataset: str, first: str, second: str, metric: str, kind: str) -> dict:
        if kind == "fresh":
            a, b = fresh(dataset), fresh(dataset)
        else:
            a, b = every(dataset, first), every(dataset, second)
        return ex.contrast(runs(dataset, first, a), runs(dataset, second, b), metric)

    rows = []
    for metric in METRICS:
        block = []
        for dataset in DATASETS:
            pairs = ISIC_PAIRS_FRESH if dataset == "isic2018" else PAIRS
            for first, second in pairs:
                result = compare(dataset, first, second, metric, "fresh")
                block.append(
                    dict(dataset=dataset, metric=metric, first=first, second=second, **result)
                )
        ex.holm(block)
        for row in block:
            best = compare(row["dataset"], row["first"], row["second"], metric, "all")
            confirmed = row["p_holm"] < 0.05 and (row["low"] > 0 or row["high"] < 0)
            same = np.sign(row["delta"]) == np.sign(best["delta"])
            excl = best["low"] > 0 or best["high"] < 0
            status = (
                "confirmed"
                if confirmed and same
                else "supported by all seeds, not confirmed by fresh seeds"
                if excl
                else "not established"
            )
            head = {k: row[k] for k in ("dataset", "metric", "first", "second")}
            rows.append(
                head
                | dict(status=status)
                | {f"fresh_{k}": v for k, v in row.items() if k not in head}
                | {f"all_{k}": v for k, v in best.items()}
            )
    write(out / "confirmation_contrasts.csv", rows)

    pooled = []
    for metric in METRICS:
        sesoi = ex.SESOI[metric]
        for kind in ("fresh", "all"):
            items = {d: compare(d, "decoder", "unet", metric, kind) for d in DATASETS}
            result = pool(list(items.values()))
            role = "primary" if metric == PRIMARY_METRIC and kind == "fresh" else "reported"
            pooled.append(
                dict(
                    role=role,
                    seeds=kind,
                    metric=metric,
                    first="decoder",
                    second="unet",
                    datasets="all three",
                    **result,
                    classification=ex.classify(result["low"], result["high"], sesoi),
                )
            )
            for left in DATASETS:
                sub = pool([v for d, v in items.items() if d != left])
                pooled.append(
                    dict(
                        role="leave-one-out",
                        seeds=kind,
                        metric=metric,
                        first="decoder",
                        second="unet",
                        datasets=f"without {left}",
                        **sub,
                        classification=ex.classify(sub["low"], sub["high"], sesoi),
                    )
                )
            for first, second in POOLED_SECONDARY:
                names = ["acdc", "synapse"] if kind == "fresh" else DATASETS
                sub = pool([compare(d, first, second, metric, kind) for d in names])
                pooled.append(
                    dict(
                        role="secondary",
                        seeds=kind,
                        metric=metric,
                        first=first,
                        second=second,
                        datasets="+".join(names),
                        **sub,
                        classification=ex.classify(sub["low"], sub["high"], sesoi),
                    )
                )
    write(out / "confirmation_pooled.csv", pooled)
    print("wrote", out)


if __name__ == "__main__":
    main()
