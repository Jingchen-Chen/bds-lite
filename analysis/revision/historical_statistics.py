"""Frozen analysis for revision/HISTORICAL_REEVALUATION_PROTOCOL.md.

Uses the expansion estimator unchanged. Writes to revision/tables/families/:
historical_april.csv, historical_october.csv, historical_tolerance.csv.
Refuses to run unless revision/historical_reevaluation/validation.json reports a pass.
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
import expansion_statistics as ex  # noqa: E402

OUT = ROOT / "revision/historical_reevaluation"
TABLES = ROOT / "revision/tables/families"
EXTRA = ROOT / "revision/extra_cohorts"
SEEDS = (1, 2, 3)
DATASETS = ("isic2018", "acdc", "synapse")
H_CONTRASTS = {
    "H1": ("a_decoder", "a_unet"),
    "H2": ("a_distill", "a_decoder"),
    "H3": ("a_gate", "a_decoder"),
    "H4": ("a_full", "a_gate_surface"),
    "H5": ("a_full", "a_decoder"),
}
D_CONTRASTS = {
    "D1": ("B", "C"),
    "D2": ("C", "G"),
    "D3": ("G", "U"),
    "D4": ("D", "G"),
    "D5": ("B", "D"),
    "D6": ("C", "M2"),
    "D7": ("B", "M2"),
}


def april_path(arm: str, dataset: str, seed: int) -> Path:
    if dataset != "isic390" and arm in ("a_unet", "a_full"):
        cohort = {"a_unet": "famO_unet", "a_full": "famO_full"}[arm]
        return EXTRA / f"{cohort}/{dataset}_seed{seed}/test/per_case.csv"
    return OUT / f"{arm}/{dataset}_seed{seed}/per_case.csv"


def april_values(arm: str, dataset: str) -> list[dict]:
    unit_dataset = "isic2018" if dataset == "isic390" else dataset
    return [ex.unit_values(april_path(arm, dataset, s), unit_dataset) for s in SEEDS]


def october_path(arm: str, seed: int) -> Path:
    if arm == "U":
        return ROOT / f"revision/official_isic/test/U_seed{seed}/per_case.csv"
    if arm == "B":
        return EXTRA / f"isic_M3_B/isic2018_seed{seed}/test/per_case.csv"
    if arm == "M2":
        return (
            ROOT
            / f"results/revision_runs/isic2018_m2_aux_boundary_only_seed{seed}/per_case_test.csv"
        )
    return OUT / f"o_{arm}/isic2018_seed{seed}/per_case.csv"


def write(name: str, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (TABLES / name).open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def april() -> list[dict]:
    arms = {a for pair in H_CONTRASTS.values() for a in pair}
    values = {(a, d): april_values(a, d) for a in arms for d in (*DATASETS, "isic390")}
    rows = []
    for metric in ex.METRICS:
        pooled = []
        for name, (first, second) in H_CONTRASTS.items():
            part = []
            for dataset in (*DATASETS, "isic390"):
                result = ex.contrast(values[(first, dataset)], values[(second, dataset)], metric)
                se = math.hypot(result["se_run"], result["se_unit"])
                row = dict(
                    family="H",
                    id=name,
                    first=first,
                    second=second,
                    metric=metric,
                    dataset=dataset,
                    se=se,
                    **result,
                )
                rows.append(row)
                if dataset != "isic390":
                    part.append(row)
            est = np.array([r["delta"] for r in part])
            weight = 1 / np.array([r["se"] ** 2 for r in part])
            mean, se = float((weight * est).sum() / weight.sum()), math.sqrt(1 / weight.sum())
            q = float((weight * (est - mean) ** 2).sum())
            low, high = mean - 1.96 * se, mean + 1.96 * se
            pooled.append(
                dict(
                    family="H",
                    id=name,
                    first=first,
                    second=second,
                    metric=metric,
                    dataset="pooled",
                    delta=mean,
                    se=se,
                    low=low,
                    high=high,
                    p=float(2 * stats.norm.sf(abs(mean / se))),
                    q_p=float(stats.chi2.sf(q, 2)),
                    classification=ex.classify(low, high, ex.SESOI[metric]),
                )
            )
        ex.holm(pooled)
        rows += pooled
    return rows


def october() -> list[dict]:
    arms = {a for pair in D_CONTRASTS.values() for a in pair}
    values = {a: [ex.unit_values(october_path(a, s), "isic2018") for s in SEEDS] for a in arms}
    ten = [ex.unit_values(ex.isic_path("decoder", s), "isic2018") for s in range(1, 11)]
    rows = []
    for metric in ex.METRICS:
        block = []
        for name, (first, second) in D_CONTRASTS.items():
            result = ex.contrast(values[first], values[second], metric)
            block.append(
                dict(family="D", id=name, first=first, second=second, metric=metric, **result)
            )
        weights = {"B": 1, "C": -1, "D": -1, "G": 1}
        cells = {c: values[c] for c in weights}
        estimate, se_run, se_unit, df = ex.linear_contrast(cells, weights, metric)
        se = math.hypot(se_run, se_unit)
        t = stats.t.ppf(0.975, df)
        low, high = estimate - t * se, estimate + t * se
        block.append(
            dict(
                family="D",
                id="D8",
                first="(B-C)",
                second="(D-G)",
                metric=metric,
                delta=estimate,
                se_run=se_run,
                se_unit=se_unit,
                df=df,
                low=low,
                high=high,
                p=float(2 * stats.t.sf(abs(estimate / se), df)),
                classification=ex.classify(low, high, ex.SESOI[metric]),
            )
        )
        ex.holm(block)
        rows += block
        for name, first in (("D6", "C"), ("D7", "B")):
            result = ex.contrast(values[first], ten, metric)
            rows.append(
                dict(
                    family="D-sensitivity",
                    id=name,
                    first=first,
                    second="M2 ten seeds",
                    metric=metric,
                    **result,
                )
            )
    return rows


def tolerance_values(arm: str, seed: int, tolerance: int) -> dict:
    with (OUT / f"tolerance/{arm}_seed{seed}.csv").open() as stream:
        return {
            "boundary_f1": {
                row["id"]: float(row[f"t{tolerance}"]) for row in csv.DictReader(stream)
            }
        }


def tolerance() -> list[dict]:
    frozen = {}
    with (TABLES / "expansion_contrasts.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["metric"] == "boundary_f1":
                frozen[(row["first"], row["second"])] = float(row["all_delta"])
    rows = []
    for value in (1, 2, 4):
        data = {(a, s): tolerance_values(a, s, value) for a in ex.NAMES for s in ex.ALL}
        for label, seeds in (("all", ex.ALL), ("fresh", ex.FRESH)):
            block = []
            for first, second in ex.PAIRS:
                a, b = [data[(first, s)] for s in seeds], [data[(second, s)] for s in seeds]
                result = ex.contrast(a, b, "boundary_f1")
                block.append(
                    dict(tolerance=value, seeds=label, first=first, second=second, **result)
                )
            ex.holm(block)
            rows += block
    for row in rows:
        if row["tolerance"] == 2 and row["seeds"] == "all":
            gap = abs(row["delta"] - frozen[(row["first"], row["second"])])
            if gap > 1e-9:
                raise SystemExit(
                    f"tolerance 2 does not reproduce the frozen value: {row} gap {gap}"
                )
    return rows


def main() -> None:
    validation = json.loads((OUT / "validation.json").read_text())
    if validation["status"] != "PASSED":
        raise SystemExit(f"validation did not pass: {validation['failures'][:5]}")
    write("historical_april.csv", april())
    write("historical_october.csv", october())
    write("historical_tolerance.csv", tolerance())
    print("written")


if __name__ == "__main__":
    main()
