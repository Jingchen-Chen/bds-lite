"""WP2/WP4/WP6 matched-family statistics for every method with complete seeds 1/2/3.

Unit convention reproduces the submitted manuscript's cluster analysis
(bds-followup analysis/generate_rescue_analysis.py): for each test sample and
foreground label, pairs where prediction and reference are both empty are
undefined (excluded); remaining values are averaged across the three seeds, then
macro-averaged across labels (finite values) to a sample value. Paired oriented
sample deltas are averaged within cluster (ISIC image, ACDC patient, Synapse case).
Inference on cluster means: two-sided Wilcoxon (zero_method wilcox, SciPy auto),
percentile bootstrap 10,000 draws (RNG 20261102, prereg_revision.md), sign effect,
leave-one-cluster-out range and revision-stage SESOI classification.
ISIC test = official 1,000-image Task-1 test; ACDC/Synapse = locked test split.

Families (analysis/revision/prereg_revision.md):
P  M6 vs M0, Holm over 5 metrics within dataset.
A  each ablation vs M6, Holm over all ablation x metric within dataset.
A0 addition path vs M0 (M1, M2, M3, M4, M5 vs U-Net), Holm over arm x metric.
C  each comparator vs M0, Holm over comparator x metric within dataset.
Omnibus: Friedman across complete methods per dataset x metric on cluster values,
Holm-adjusted paired Wilcoxon post-hoc over all method pairs.
Families with missing members are written with complete=false (provisional).
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import re
from collections import defaultdict
from functools import cache
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import friedmanchisquare, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
METRICS = ["dsc", "iou", "boundary_f1", "hd95", "assd"]
DISTANCE = {"hd95", "assd"}
SESOI = {"dsc": 0.01, "iou": 0.01, "boundary_f1": 0.01, "hd95": 2.0, "assd": 0.5}
DATASETS = ["isic2018", "acdc", "synapse"]
SEEDS = [1, 2, 3]
DRAWS, RNG_SEED = 10_000, 20261102
FOLLOWUP = ROOT / "recovered/followup/results/followup_20261018_v2"
RUNS = ROOT / "results/revision_runs"
OFFICIAL = ROOT / "revision/official_isic/test"
RECOVERY = ROOT / "revision/prediction_recovery"

# method -> (label, family role). Design (revision/GITHUB_AUDIT.md):
#   S0/S6  submitted models (family O): primary family P. ISIC evaluated on the official test
#          (the submitted 519-image numbers are contaminated, audit C1).
#   M0/M6  pipeline-matched references for ablations/comparators (adapter = p1b pipeline,
#          locked partitions). ACDC M6 = adapter-trained without class weights.
#   R0/R6  p1b follow-up family: independent replication (ACDC R6 class-weighted).
METHODS = {
    "S0": ("U-Net (submitted models)", "primary"),
    "S6": ("BDS-Lite (submitted models)", "primary"),
    "S6g": ("BDS-Lite, gate set to identity at inference (submitted models)", "gate"),
    "M0": ("U-Net", "baseline"),
    "M6": ("BDS-Lite (full)", "full"),
    "M1": ("U-Net + SDF only", "ablation"),
    "M2": ("U-Net + aux. boundary only", "ablation"),
    "M3": ("BDS-Lite - SDF", "ablation"),
    "M4": ("BDS-Lite - distillation", "ablation"),
    "M5": ("BDS-Lite - gate (alpha=0)", "ablation"),
    "GSL": ("U-Net + GSL", "comparator"),
    "C1": ("U-Net + Boundary Loss", "comparator"),
    "C3": ("Attention U-Net", "comparator"),
    "C4": ("LB-UNet", "comparator"),
    "R0": ("U-Net (replication family)", "replication"),
    "R6": ("BDS-Lite (replication family)", "replication"),
}
EXTRA = ROOT / "revision/extra_cohorts"
EXTRA_COHORT = {"S0": "famO_unet", "S6": "famO_full", "S6g": "famO_full_gateid"}
RUN_ARM = {
    "M1": "m1_sdf_only",
    "M2": "m2_aux_boundary_only",
    "M3": "m3_no_sdf",
    "M4": "m4_no_distill",
    "M5": "m5_no_gate",
    "GSL": "gsl",
    "C1": "c1_boundary_loss",
    "C3": "c3_attention_unet",
    "C4": "c4_lbunet",
}
# ACDC ablations/reference without class weights (ABLATION_PROTOCOL_AMENDMENT_1/2.md)
ACDC_RUN_ARM = {
    "M1": "m1_sdf_only_unweighted",
    "M2": "m2_aux_boundary_only_unweighted",
    "M3": "m3_no_sdf_unweighted",
    "M4": "m4_no_distill_unweighted",
    "M5": "m5_no_gate_unweighted",
    "M6": "m6_full_unweighted",
}
OFFICIAL_ARM = {"M0": "U", "M6": "F", "M1": "S", "R0": "U", "R6": "F"}
RECOVERY_FAMILY = {
    "M0": "main_p1b_unet",
    "M6": "main_p1b_full",
    "R0": "main_p1b_unet",
    "R6": "main_p1b_full",
}


def run_arm(method: str, dataset: str) -> str | None:
    if dataset == "acdc" and method in ACDC_RUN_ARM:
        return ACDC_RUN_ARM[method]
    return RUN_ARM.get(method)


def per_case_path(method: str, dataset: str, seed: int) -> Path | None:
    """Return the test per-case CSV for a method, or None if absent/incomplete."""
    candidates = []
    if dataset == "isic2018" and method in OFFICIAL_ARM:
        candidates.append(OFFICIAL / f"{OFFICIAL_ARM[method]}_seed{seed}/per_case.csv")
    acdc_matched_m6 = dataset == "acdc" and method == "M6"
    if dataset != "isic2018" and method in RECOVERY_FAMILY and not acdc_matched_m6:
        folder = RECOVERY / RECOVERY_FAMILY[method] / f"{dataset}_seed{seed}/test"
        if (folder / "evaluation_complete.json").exists():
            candidates.append(folder / "per_case.csv")
    if dataset == "isic2018" and method == "M3":
        folder = EXTRA / f"isic_M3_B/isic2018_seed{seed}/test"
        if (folder / "provenance.json").exists():
            candidates.append(folder / "per_case.csv")
    if method in EXTRA_COHORT:
        folder = EXTRA / EXTRA_COHORT[method] / f"{dataset}_seed{seed}/test"
        if (folder / "provenance.json").exists():
            candidates.append(folder / "per_case.csv")
    arm = run_arm(method, dataset)
    if arm:
        run = RUNS / f"{dataset}_{arm}_seed{seed}"
        if (run / "final_metrics.json").exists():
            candidates.append(run / "per_case_test.csv")
    for path in candidates:
        if path.exists():
            return path
    return None


def cluster_of(dataset: str, sample: str) -> str:
    if dataset == "isic2018":
        return sample
    match = re.match(r"^(patient\d+)" if dataset == "acdc" else r"^(case\d+)", sample)
    if match is None:
        raise ValueError(f"cannot derive cluster from {sample}")
    return match.group(1)


def _value(row: dict, metric: str) -> float:
    if row.get("empty_prediction") == "True" and row.get("empty_target") == "True":
        return math.nan
    text = row.get(metric, "")
    return float(text) if text not in ("", "None", "nan") else math.nan


@cache
def sample_values(method: str, dataset: str) -> dict[str, dict[str, float]] | None:
    """sample -> metric -> seed-averaged label-macro value; None if any seed missing."""
    per_seed = []
    for seed in SEEDS:
        path = per_case_path(method, dataset, seed)
        if path is None:
            return None
        table: dict[tuple[str, str], dict] = {}
        for row in csv.DictReader(path.open()):
            table[(row["id"], row.get("label", "1"))] = row
        per_seed.append(table)
    keys = sorted(per_seed[0])
    if any(sorted(t) != keys for t in per_seed[1:]):
        raise ValueError(f"seed sample/label sets differ: {method} {dataset}")
    by_sample: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for key in keys:
        for metric in METRICS:
            values = np.array([_value(t[key], metric) for t in per_seed], float)
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


def seed_level_means(method: str, dataset: str) -> dict[str, list[float]] | None:
    """Per seed: mean over test samples of the label-macro value (both-empty excluded)."""
    result = {metric: [] for metric in METRICS}
    for seed in SEEDS:
        path = per_case_path(method, dataset, seed)
        if path is None:
            return None
        by_sample: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in csv.DictReader(path.open()):
            for metric in METRICS:
                by_sample[row["id"]][metric].append(_value(row, metric))
        for metric in METRICS:
            sample_means = []
            for metrics in by_sample.values():
                arr = np.array(metrics[metric], float)
                arr = arr[np.isfinite(arr)]
                if arr.size:
                    sample_means.append(arr.mean())
            result[metric].append(float(np.mean(sample_means)))
    return result


def bootstrap(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(RNG_SEED)
    means = np.empty(DRAWS)
    for start in range(0, DRAWS, 500):
        idx = rng.integers(0, values.size, size=(500, values.size))
        means[start : start + 500] = values[idx].mean(1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def judgement(low: float, high: float, sesoi: float) -> str:
    if low > sesoi:
        return "practically superior"
    if high < -sesoi:
        return "practically inferior"
    if low > -sesoi and high < sesoi:
        return "practically equivalent"
    return "inconclusive"


def cluster_deltas(a: str, b: str, dataset: str, metric: str) -> dict[str, float] | None:
    """Oriented (positive favours a) cluster-mean deltas, a minus b."""
    va, vb = sample_values(a, dataset), sample_values(b, dataset)
    if va is None or vb is None:
        return None
    if set(va) != set(vb):
        raise ValueError(f"sample sets differ: {a} vs {b} on {dataset}")
    sign = -1.0 if metric in DISTANCE else 1.0
    clusters: dict[str, list[float]] = defaultdict(list)
    for sample in va:
        delta = sign * (va[sample][metric] - vb[sample][metric])
        if math.isfinite(delta):
            clusters[cluster_of(dataset, sample)].append(delta)
    return {c: float(np.mean(v)) for c, v in sorted(clusters.items())}


def compare(a: str, b: str, dataset: str, metric: str) -> dict | None:
    deltas = cluster_deltas(a, b, dataset, metric)
    if deltas is None:
        return None
    values = np.array(list(deltas.values()), float)
    nonzero = values[values != 0]
    if values.size >= 6 and nonzero.size:
        test = wilcoxon(values, alternative="two-sided", zero_method="wilcox", method="auto")
        statistic, p = float(test.statistic), float(test.pvalue)
    else:
        statistic, p = math.nan, math.nan
    low, high = bootstrap(values)
    loo = [np.mean(np.delete(values, i)) for i in range(values.size)] if values.size > 1 else [0]
    return dict(
        dataset=dataset,
        method=a,
        reference=b,
        metric=metric,
        n_clusters=int(values.size),
        mean_oriented_delta=float(values.mean()),
        median_oriented_delta=float(np.median(values)),
        ci95_low=low,
        ci95_high=high,
        wilcoxon_W=statistic,
        p_raw=p,
        sign_effect=float(((nonzero > 0).sum() - (nonzero < 0).sum()) / nonzero.size)
        if nonzero.size
        else 0.0,
        loo_min=float(np.min(loo)),
        loo_max=float(np.max(loo)),
        sesoi=SESOI[metric],
        practical=judgement(low, high, SESOI[metric]),
    )


def holm(rows: list[dict]) -> None:
    valid = [r for r in rows if math.isfinite(r["p_raw"])]
    order = sorted(valid, key=lambda r: r["p_raw"])
    running = 0.0
    for rank, row in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * row["p_raw"]))
        row["p_holm"] = running
    for row in rows:
        row.setdefault("p_holm", math.nan)
        row["holm_m"] = len(valid)


def family(name: str, pairs: list[tuple[str, str]], dataset: str) -> list[dict]:
    rows, missing = [], []
    for a, b in pairs:
        for metric in METRICS:
            result = compare(a, b, dataset, metric)
            if result is None:
                missing.append(a)
                break
            rows.append(result)
    holm(rows)
    complete = not missing
    for row in rows:
        row.update(
            family=name, family_complete=complete, missing_members=";".join(sorted(set(missing)))
        )
        row["decision"] = (
            "favourable"
            if row["p_holm"] < 0.05 and row["ci95_low"] > 0
            else "unfavourable"
            if row["p_holm"] < 0.05 and row["ci95_high"] < 0
            else "no decision"
        )
    return rows


def omnibus(dataset: str) -> tuple[list[dict], list[dict]]:
    methods = [
        m
        for m in METHODS
        if METHODS[m][1] not in ("primary", "replication", "gate")
        and sample_values(m, dataset) is not None
    ]
    summary, posthoc = [], []
    if len(methods) < 3:
        return summary, posthoc
    for metric in METRICS:
        per_method = {}
        for m in methods:
            v = sample_values(m, dataset)
            clusters: dict[str, list[float]] = defaultdict(list)
            for sample, metrics in v.items():
                if math.isfinite(metrics[metric]):
                    clusters[cluster_of(dataset, sample)].append(metrics[metric])
            per_method[m] = {c: float(np.mean(x)) for c, x in clusters.items()}
        common = sorted(set.intersection(*(set(x) for x in per_method.values())))
        matrix = np.array([[per_method[m][c] for c in common] for m in methods])
        stat, p = friedmanchisquare(*matrix)
        sign = -1 if metric in DISTANCE else 1
        ranks = np.argsort(np.argsort(-sign * matrix, axis=0), axis=0) + 1
        summary.append(
            dict(
                dataset=dataset,
                metric=metric,
                methods=";".join(methods),
                n_clusters=len(common),
                friedman_chi2=float(stat),
                p=float(p),
                mean_ranks=json.dumps(
                    {m: float(r) for m, r in zip(methods, ranks.mean(1), strict=True)}
                ),
            )
        )
        rows = []
        for i, j in itertools.combinations(range(len(methods)), 2):
            d = sign * (matrix[i] - matrix[j])
            p_pair = float(wilcoxon(d).pvalue) if np.any(d) else 1.0
            rows.append(
                dict(
                    dataset=dataset,
                    metric=metric,
                    method=methods[i],
                    reference=methods[j],
                    mean_oriented_delta=float(d.mean()),
                    p_raw=p_pair,
                )
            )
        holm(rows)
        posthoc.extend(rows)
    return summary, posthoc


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def method_summary() -> list[dict]:
    rows = []
    for dataset in DATASETS:
        for method, (label, role) in METHODS.items():
            means = seed_level_means(method, dataset)
            if means is None:
                rows.append(
                    dict(dataset=dataset, method=method, label=label, role=role, complete=False)
                )
                continue
            row = dict(dataset=dataset, method=method, label=label, role=role, complete=True)
            for metric in METRICS:
                row[f"{metric}_mean"] = float(np.mean(means[metric]))
                row[f"{metric}_sd"] = float(np.std(means[metric], ddof=1))
                row[f"{metric}_seeds"] = json.dumps(means[metric])
            rows.append(row)
    return rows


AREA_BINS = [(0.0, 0.01, "<1%"), (0.01, 0.02, "1-2%"), (0.02, 0.05, "2-5%"), (0.05, 1.01, ">5%")]


def collapse_summary() -> tuple[list[dict], list[dict]]:
    """WP1c: missed-structure / false-alarm rates by area bin and per-class DSC (seed means)."""
    bins_out, class_out = [], []
    for dataset in DATASETS:
        for method in METHODS:
            paths = [per_case_path(method, dataset, s) for s in SEEDS]
            if any(p is None for p in paths):
                continue
            per_seed_bins = defaultdict(lambda: defaultdict(list))
            per_seed_class = defaultdict(lambda: defaultdict(list))
            for path in paths:
                counts = defaultdict(lambda: [0, 0, []])  # eligible, missed, dsc
                alarms = defaultdict(lambda: [0, 0])
                class_dsc = defaultdict(list)
                for row in csv.DictReader(path.open()):
                    label = row.get("label", "1")
                    empty_t = row["empty_target"] == "True"
                    empty_p = row["empty_prediction"] == "True"
                    if empty_t:
                        alarms[label][0] += 1
                        alarms[label][1] += int(not empty_p)
                        continue
                    area = (
                        int(float(row["target_area"])) / (224 * 224)
                        if row.get("target_area")
                        else None
                    )
                    if area is None:
                        continue
                    name = next(b[2] for b in AREA_BINS if b[0] <= area < b[1])
                    counts[name][0] += 1
                    counts[name][1] += int(empty_p)
                    counts[name][2].append(float(row["dsc"]))
                    class_dsc[label].append(float(row["dsc"]))
                for name, (eligible, missed, dscs) in counts.items():
                    per_seed_bins[name]["missed_rate"].append(missed / eligible)
                    per_seed_bins[name]["dsc"].append(float(np.mean(dscs)))
                    per_seed_bins[name]["n"].append(eligible)
                total_alarm = sum(a[0] for a in alarms.values())
                per_seed_bins["empty-reference"]["false_alarm_rate"].append(
                    sum(a[1] for a in alarms.values()) / total_alarm if total_alarm else math.nan
                )
                for label, values in class_dsc.items():
                    per_seed_class[label]["dsc_present"].append(float(np.mean(values)))
                    eligible = alarms[label][0]
                    per_seed_class[label]["false_alarm_rate"].append(
                        alarms[label][1] / eligible if eligible else math.nan
                    )
            for name, metrics in per_seed_bins.items():
                row = dict(dataset=dataset, method=method, area_bin=name)
                for key, values in metrics.items():
                    row[key] = float(np.nanmean(values))
                bins_out.append(row)
            for label, metrics in per_seed_class.items():
                row = dict(dataset=dataset, method=method, label=label)
                for key, values in metrics.items():
                    row[key] = float(np.nanmean(values))
                class_out.append(row)
    return bins_out, class_out


def main() -> None:
    out = ROOT / "revision/tables/families"
    ablations = ["M1", "M2", "M3", "M4", "M5"]
    comparators = ["GSL", "C1", "C3", "C4"]
    all_rows, omni, post = [], [], []
    for dataset in DATASETS:
        all_rows += family("P", [("S6", "S0")], dataset)
        all_rows += family("P-replication", [("R6", "R0")], dataset)
        all_rows += family("P-matched", [("M6", "M0")], dataset)
        all_rows += family("G", [("S6g", "S6")], dataset)
        all_rows += family("A", [(m, "M6") for m in ablations], dataset)
        all_rows += family("A0", [(m, "M0") for m in ablations], dataset)
        all_rows += family("C", [(m, "M0") for m in comparators], dataset)
        all_rows += family("C-vs-M6", [(m, "M6") for m in comparators], dataset)
        s, p = omnibus(dataset)
        omni += s
        post += p
    write(out / "family_statistics.csv", all_rows)
    write(out / "omnibus_friedman.csv", omni)
    write(out / "omnibus_posthoc.csv", post)
    write(out / "method_summary.csv", method_summary())
    bins, classes = collapse_summary()
    write(out / "collapse_by_area.csv", bins)
    write(out / "per_class_dsc.csv", classes)
    (out / "provenance.json").write_text(
        json.dumps(
            dict(
                scipy=scipy.__version__,
                numpy=np.__version__,
                draws=DRAWS,
                rng_seed=RNG_SEED,
                inputs={
                    f"{m}/{d}/seed{s}": str(p.relative_to(ROOT)) if p else None
                    for m in METHODS
                    for d in DATASETS
                    for s in SEEDS
                    for p in [per_case_path(m, d, s)]
                },
            ),
            indent=1,
        )
        + "\n"
    )
    print(f"{len(all_rows)} family rows; {len(omni)} omnibus rows -> {out}")


if __name__ == "__main__":
    main()
