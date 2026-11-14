# ruff: noqa: E501
"""WP3 sensitivity of BDS-Lite to the gate scale alpha and the auxiliary weights lambda
(revision/SENSITIVITY_PROTOCOL.md; trimmed by ABLATION_PROTOCOL_AMENDMENT_2 / REVISION_STRATEGY.md).

Arms (pipeline-matched, same adapter, partitions, schedule and selection rule as M6):
  alpha  in {0 (= M5), 0.1, 0.25 (= M6), 0.5, 1.0}; ISIC2018 seeds 1-3, Synapse seed 1 only.
  lambda_b = lambda_distill in {0.01, 0.05 (= M6), 0.2}; ISIC2018 seed 1 only.
Each alternative value is contrasted with the registered value (M6) using the conventions of
family_statistics.py (both-empty excluded, seed-averaged label-macro sample values, cluster
means of paired oriented deltas, Wilcoxon, percentile bootstrap, SESOI judgement). Holm
correction is applied only within the three-seed ISIC2018 alpha family (4 values x 5 metrics);
single-seed settings are descriptive references (CI over test units of one training run, no
Holm decision), as fixed in plan.md. Each value is also contrasted with U-Net (M0, same seeds)
to show whether the main ISIC2018/Synapse conclusions depend on alpha or lambda.
Validation DSC (selection set; ISIC2018 locked 519 images) is reported because the protocol
allows statements about a "better" value only on validation data (labelled post hoc).

Outputs: revision/tables/families/sensitivity.csv, revision/tables/families/sensitivity_means.csv
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from analysis.revision import family_statistics as fs

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "revision/tables/families"
FOLLOWUP = ROOT / "recovered/followup/results/followup_20261018_v2"

# (dataset, parameter, value label, numeric value, arm code or run stem, seeds)
SETTINGS = [
    ("isic2018", "alpha", "0", 0.0, "M5", (1, 2, 3)),
    ("isic2018", "alpha", "0.1", 0.1, "s_alpha0p1", (1, 2, 3)),
    ("isic2018", "alpha", "0.25", 0.25, "M6", (1, 2, 3)),
    ("isic2018", "alpha", "0.5", 0.5, "s_alpha0p5", (1, 2, 3)),
    ("isic2018", "alpha", "1.0", 1.0, "s_alpha1p0", (1, 2, 3)),
    ("synapse", "alpha", "0", 0.0, "M5", (1,)),
    ("synapse", "alpha", "0.1", 0.1, "s_alpha0p1", (1,)),
    ("synapse", "alpha", "0.25", 0.25, "M6", (1,)),
    ("synapse", "alpha", "0.5", 0.5, "s_alpha0p5", (1,)),
    ("synapse", "alpha", "1.0", 1.0, "s_alpha1p0", (1,)),
    ("isic2018", "lambda", "0.01", 0.01, "s_lambda0p01", (1,)),
    ("isic2018", "lambda", "0.05", 0.05, "M6", (1,)),
    ("isic2018", "lambda", "0.2", 0.2, "s_lambda0p2", (1,)),
]
REGISTERED = {"alpha": "0.25", "lambda": "0.05"}


def path_for(arm: str, dataset: str, seed: int) -> Path | None:
    if arm in fs.METHODS:
        return fs.per_case_path(arm, dataset, seed)
    run = fs.RUNS / f"{dataset}_{arm}_seed{seed}"
    return run / "per_case_test.csv" if (run / "final_metrics.json").exists() else None


def sample_values(arm: str, dataset: str, seeds: tuple[int, ...]) -> dict | None:
    """family_statistics.sample_values restricted to the given seeds."""
    tables = []
    for seed in seeds:
        path = path_for(arm, dataset, seed)
        if path is None:
            return None
        tables.append({(r["id"], r.get("label", "1")): r for r in csv.DictReader(path.open())})
    keys = sorted(tables[0])
    if any(sorted(t) != keys for t in tables[1:]):
        raise ValueError(f"sample sets differ across seeds: {arm} {dataset}")
    by_sample: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for key in keys:
        for metric in fs.METRICS:
            values = np.array([fs._value(t[key], metric) for t in tables], float)
            finite = values[np.isfinite(values)]
            by_sample[key[0]][metric].append(float(finite.mean()) if finite.size else math.nan)
    out = {}
    for sample, metrics in by_sample.items():
        out[sample] = {}
        for metric, label_values in metrics.items():
            arr = np.array(label_values, float)
            arr = arr[np.isfinite(arr)]
            out[sample][metric] = float(arr.mean()) if arr.size else math.nan
    return out


def seed_means(arm: str, dataset: str, seeds: tuple[int, ...]) -> dict[str, float]:
    """Mean over seeds of the per-seed test mean (family_statistics.seed_level_means convention)."""
    result = {}
    per_seed = {m: [] for m in fs.METRICS}
    for seed in seeds:
        rows = list(csv.DictReader(path_for(arm, dataset, seed).open()))
        by_sample: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for r in rows:
            for m in fs.METRICS:
                by_sample[r["id"]][m].append(fs._value(r, m))
        for m in fs.METRICS:
            vals = [np.nanmean(v[m]) for v in by_sample.values() if np.isfinite(v[m]).any()]
            per_seed[m].append(float(np.mean(vals)))
    for m in fs.METRICS:
        result[m] = float(np.mean(per_seed[m]))
    return result


def validation_dsc(arm: str, dataset: str, seeds: tuple[int, ...]) -> float:
    values = []
    for seed in seeds:
        if arm == "M6" and dataset == "isic2018":
            path = FOLLOWUP / f"isic2018_F_seed{seed}/summary.json"
            values.append(json.loads(path.read_text())["dsc"] if path.exists() else math.nan)
            continue
        stem = fs.run_arm(arm, dataset) if arm in fs.METHODS else arm
        path = fs.RUNS / f"{dataset}_{stem}_seed{seed}/final_metrics.json"
        values.append(
            json.loads(path.read_text())["metrics"]["val"]["dsc"]["mean"]
            if stem and path.exists()
            else math.nan
        )
    return float(np.mean(values))


def contrast(va: dict, vb: dict, dataset: str, metric: str) -> dict:
    sign = -1.0 if metric in fs.DISTANCE else 1.0
    clusters: dict[str, list[float]] = defaultdict(list)
    for sample in va:
        d = sign * (va[sample][metric] - vb[sample][metric])
        if math.isfinite(d):
            clusters[fs.cluster_of(dataset, sample)].append(d)
    values = np.array([np.mean(v) for _, v in sorted(clusters.items())], float)
    p = float(wilcoxon(values).pvalue) if values.size >= 6 and np.any(values) else math.nan
    low, high = fs.bootstrap(values)
    return dict(
        n_clusters=values.size,
        mean_oriented_delta=float(values.mean()),
        ci95_low=low,
        ci95_high=high,
        p_raw=p,
        practical=fs.judgement(low, high, fs.SESOI[metric]),
    )


def main() -> None:
    rows, means = [], []
    for dataset, parameter, label, value, arm, seeds in SETTINGS:
        va = sample_values(arm, dataset, seeds)
        if va is None:
            means.append(dict(dataset=dataset, parameter=parameter, value=label, complete=False))
            continue
        m = seed_means(arm, dataset, seeds)
        means.append(
            dict(
                dataset=dataset,
                parameter=parameter,
                value=label,
                numeric=value,
                arm=arm,
                seeds=len(seeds),
                complete=True,
                registered=label == REGISTERED[parameter],
                val_dsc=validation_dsc(arm, dataset, seeds),
                **{f"{k}_mean": v for k, v in m.items()},
            )
        )
        for reference in ("M6", "M0"):
            if reference == "M6" and label == REGISTERED[parameter]:
                continue
            vb = sample_values(reference, dataset, seeds)
            for metric in fs.METRICS:
                rows.append(
                    dict(
                        dataset=dataset,
                        parameter=parameter,
                        value=label,
                        seeds=len(seeds),
                        reference=reference,
                        metric=metric,
                        **contrast(va, vb, dataset, metric),
                    )
                )
    for dataset, parameter in {(r["dataset"], r["parameter"]) for r in rows}:
        for reference in ("M6", "M0"):
            family = [
                r
                for r in rows
                if r["dataset"] == dataset
                and r["parameter"] == parameter
                and r["reference"] == reference
            ]
            if family and family[0]["seeds"] == 3:
                fs.holm(family)
            for r in family:
                r.setdefault("p_holm", math.nan)
                inferential = r["seeds"] == 3
                r["inference"] = "holm_within_family" if inferential else "single_seed_descriptive"
                significant = (
                    inferential and r["p_holm"] < 0.05 and (r["ci95_low"] > 0 or r["ci95_high"] < 0)
                )
                r["decision"] = (
                    ("favourable" if r["mean_oriented_delta"] > 0 else "unfavourable")
                    if significant
                    else "no decision"
                )
    rows.sort(
        key=lambda r: (
            r["dataset"],
            r["parameter"],
            r["reference"],
            r["value"],
            fs.METRICS.index(r["metric"]),
        )
    )
    fs.write(OUT / "sensitivity.csv", rows)
    fs.write(OUT / "sensitivity_means.csv", means)
    for r in means:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    for r in rows:
        print(
            r["dataset"],
            r["parameter"],
            r["value"],
            r["reference"],
            r["metric"],
            f"{r['mean_oriented_delta']:+.4f} [{r['ci95_low']:+.4f},{r['ci95_high']:+.4f}]",
            f"holm={r['p_holm']:.3g}",
            r["practical"],
            r["decision"],
        )


if __name__ == "__main__":
    main()
