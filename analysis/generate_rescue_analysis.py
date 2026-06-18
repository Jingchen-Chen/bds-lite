#!/usr/bin/env python
"""Generate artifact-only analyses for the BDS-Lite midterm Q2 rescue."""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import spearmanr, wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bds_lite.data.boundary import mask_to_boundary  # noqa: E402
from bds_lite.evaluation.metrics import (  # noqa: E402
    assd,
    boundary_f1,
    dice_score,
    hd95,
    iou_score,
)

# Public-release layout: analysis outputs land in analysis/outputs/, figures in
# figures/. The per-case prediction arrays are distributed via Zenodo (see
# ZENODO_MANIFEST.csv); place them under outputs/evaluations/predictions/ to
# reproduce the cluster statistics. The committed analysis/outputs/*.csv are the
# artifacts of record and were verified to reproduce bit-for-bit from this script.
ANALYSIS = ROOT / "analysis" / "outputs"
FIGURES = ROOT / "figures"
FAILURE_FIGURES = FIGURES / "failure_panels"
PRED_ROOT = ROOT / "outputs/evaluations/predictions"
RNG = np.random.default_rng(20260606)

DATASETS = {
    "isic2018": {
        "split": "val",
        "labels": [1],
        "class_names": {1: "lesion"},
    },
    "acdc": {
        "split": "test",
        "labels": [1, 2, 3],
        "class_names": {
            1: "right_ventricle",
            2: "myocardium",
            3: "left_ventricle",
        },
    },
    "synapse": {
        "split": "test",
        "labels": list(range(1, 9)),
        "class_names": {
            1: "aorta",
            2: "gallbladder",
            3: "spleen",
            4: "left_kidney",
            5: "right_kidney",
            6: "liver",
            7: "stomach",
            8: "pancreas",
        },
    },
}
METHODS = ("unet", "bds_lite_full")
SEEDS = (1, 2, 3)
METRICS = ("dsc", "iou", "hd95", "assd", "boundary_f1")
DISTANCE_METRICS = {"hd95", "assd"}


def finite_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(array)) if np.isfinite(array).any() else float("nan")


def csv_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in fields})


def cluster_id(dataset: str, stem: str) -> str:
    if dataset == "isic2018":
        return stem
    pattern = r"^(patient\d+)" if dataset == "acdc" else r"^(case\d+)"
    match = re.match(pattern, stem)
    return match.group(1) if match else stem


def per_label_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    label: int,
) -> dict[str, float]:
    pred_bin = pred == label
    target_bin = target == label
    if not pred_bin.any() and not target_bin.any():
        return {metric: float("nan") for metric in METRICS}
    pred_boundary = mask_to_boundary(pred_bin.astype(np.uint8), radius=1)
    target_boundary = mask_to_boundary(target_bin.astype(np.uint8), radius=1)
    return {
        "dsc": dice_score(pred, target, label),
        "iou": iou_score(pred, target, label),
        "hd95": hd95(pred, target, label),
        "assd": assd(pred, target, label),
        "boundary_f1": boundary_f1(pred_boundary, target_boundary, tolerance=2),
    }


def geometry(mask: np.ndarray) -> dict[str, float | int]:
    foreground = mask > 0
    area = int(foreground.sum())
    boundary = mask_to_boundary(mask, radius=1)
    boundary_length = int(boundary.sum())
    if area:
        complexity = boundary_length / math.sqrt(area)
        compactness = boundary_length**2 / (4.0 * math.pi * area)
    else:
        complexity = float("nan")
        compactness = float("nan")
    return {
        "target_area_pixels": area,
        "target_area_ratio": area / mask.size,
        "boundary_length_pixels": boundary_length,
        "boundary_complexity": complexity,
        "compactness": compactness,
        "n_classes_present": int(np.unique(mask[foreground]).size),
    }


def prediction_path(dataset: str, method: str, seed: int, stem: str) -> Path:
    split = DATASETS[dataset]["split"]
    return PRED_ROOT / f"{dataset}_{method}_seed{seed}_{split}" / f"{stem}.npy"


def evaluate_cases() -> tuple[list[dict], list[dict]]:
    case_rows: list[dict] = []
    class_rows: list[dict] = []
    for dataset, cfg in DATASETS.items():
        gt_dir = ROOT / f"data/processed/{dataset}/{cfg['split']}"
        gt_paths = sorted(gt_dir.glob("*.npz"))
        print(f"{dataset}: {len(gt_paths)} samples")
        for index, gt_path in enumerate(gt_paths, start=1):
            stem = gt_path.stem
            with np.load(gt_path) as sample:
                target = sample["mask"]
            row = {
                "dataset": dataset,
                "split": cfg["split"],
                "sample_id": stem,
                "cluster_id": cluster_id(dataset, stem),
                **geometry(target),
            }
            method_class_values: dict[str, dict[int, dict[str, list[float]]]] = {
                method: {label: {metric: [] for metric in METRICS} for label in cfg["labels"]}
                for method in METHODS
            }
            for method in METHODS:
                for seed in SEEDS:
                    pred_path = prediction_path(dataset, method, seed, stem)
                    if not pred_path.exists():
                        raise FileNotFoundError(pred_path)
                    pred = np.load(pred_path)
                    if pred.shape != target.shape:
                        raise ValueError(f"shape mismatch: {pred_path} vs {gt_path}")
                    for label in cfg["labels"]:
                        metrics = per_label_metrics(pred, target, label)
                        for metric, value in metrics.items():
                            method_class_values[method][label][metric].append(value)
            for method in METHODS:
                for metric in METRICS:
                    class_means = [
                        finite_mean(method_class_values[method][label][metric])
                        for label in cfg["labels"]
                    ]
                    row[f"{method}_{metric}"] = finite_mean(class_means)
            for metric in METRICS:
                delta = row[f"bds_lite_full_{metric}"] - row[f"unet_{metric}"]
                row[f"delta_{metric}"] = delta
                row[f"oriented_delta_{metric}"] = -delta if metric in DISTANCE_METRICS else delta
            case_rows.append(row)
            for label in cfg["labels"]:
                class_row = {
                    "dataset": dataset,
                    "split": cfg["split"],
                    "sample_id": stem,
                    "cluster_id": row["cluster_id"],
                    "class_id": label,
                    "class_name": cfg["class_names"][label],
                    "target_class_area_pixels": int((target == label).sum()),
                    "target_class_area_ratio": float((target == label).mean()),
                }
                for method in METHODS:
                    for metric in METRICS:
                        class_row[f"{method}_{metric}"] = finite_mean(
                            method_class_values[method][label][metric]
                        )
                for metric in METRICS:
                    delta = class_row[f"bds_lite_full_{metric}"] - class_row[f"unet_{metric}"]
                    class_row[f"delta_{metric}"] = delta
                    class_row[f"oriented_delta_{metric}"] = (
                        -delta if metric in DISTANCE_METRICS else delta
                    )
                class_rows.append(class_row)
            if index % 250 == 0:
                print(f"  processed {index}/{len(gt_paths)}")
    return case_rows, class_rows


def distribution_rows(case_rows: list[dict]) -> list[dict]:
    rows = []
    for dataset in DATASETS:
        subset = [row for row in case_rows if row["dataset"] == dataset]
        for metric in METRICS:
            values = np.asarray(
                [row[f"oriented_delta_{metric}"] for row in subset], dtype=np.float64
            )
            values = values[np.isfinite(values)]
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "n": int(values.size),
                    "mean_oriented_delta": float(np.mean(values)),
                    "std_oriented_delta": float(np.std(values, ddof=1)),
                    "median_oriented_delta": float(np.median(values)),
                    "q05": float(np.quantile(values, 0.05)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                    "q95": float(np.quantile(values, 0.95)),
                    "favorable_count": int((values > 0).sum()),
                    "unfavorable_count": int((values < 0).sum()),
                    "tie_count": int((values == 0).sum()),
                    "favorable_fraction": float(np.mean(values > 0)),
                }
            )
    return rows


def top_case_rows(case_rows: list[dict], top_k: int = 10) -> list[dict]:
    rows = []
    for dataset in DATASETS:
        subset = [row for row in case_rows if row["dataset"] == dataset]
        for metric in METRICS:
            valid = [row for row in subset if math.isfinite(row[f"oriented_delta_{metric}"])]
            valid.sort(key=lambda row: row[f"oriented_delta_{metric}"])
            for direction, selected in (
                ("largest_degradation", valid[:top_k]),
                ("largest_improvement", valid[-top_k:][::-1]),
            ):
                for rank, row in enumerate(selected, start=1):
                    rows.append(
                        {
                            "dataset": dataset,
                            "metric": metric,
                            "direction": direction,
                            "rank": rank,
                            "sample_id": row["sample_id"],
                            "cluster_id": row["cluster_id"],
                            "unet_value": row[f"unet_{metric}"],
                            "bds_lite_full_value": row[f"bds_lite_full_{metric}"],
                            "raw_delta_bds_minus_unet": row[f"delta_{metric}"],
                            "oriented_delta_positive_is_favorable": row[f"oriented_delta_{metric}"],
                        }
                    )
    return rows


def outlier_rows(case_rows: list[dict]) -> list[dict]:
    rows = []
    for dataset in DATASETS:
        subset = [row for row in case_rows if row["dataset"] == dataset]
        for metric in METRICS:
            values = np.asarray(
                [row[f"oriented_delta_{metric}"] for row in subset], dtype=np.float64
            )
            values = values[np.isfinite(values)]
            ordered = np.sort(values)
            n = values.size
            trim_1 = max(0, int(math.floor(n * 0.01)))
            trim_5 = max(0, int(math.floor(n * 0.05)))
            abs_order = np.sort(np.abs(values))[::-1]
            top_n = max(1, int(math.ceil(n * 0.01)))
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "n": n,
                    "full_mean": float(np.mean(values)),
                    "trimmed_1pct_mean": float(
                        np.mean(ordered[trim_1 : n - trim_1]) if n > 2 * trim_1 else np.nan
                    ),
                    "trimmed_5pct_mean": float(
                        np.mean(ordered[trim_5 : n - trim_5]) if n > 2 * trim_5 else np.nan
                    ),
                    "minimum": float(ordered[0]),
                    "maximum": float(ordered[-1]),
                    "top_1pct_share_of_absolute_delta": float(
                        abs_order[:top_n].sum() / abs_order.sum() if abs_order.sum() else 0.0
                    ),
                }
            )
    return rows


def assign_tertile(rows: list[dict], key: str) -> dict[str, str]:
    valid = [(row["sample_id"], row[key]) for row in rows if math.isfinite(float(row[key]))]
    values = np.asarray([value for _, value in valid], dtype=np.float64)
    low, high = np.quantile(values, [1 / 3, 2 / 3])
    result = {}
    for sample_id, value in valid:
        result[sample_id] = "low" if value <= low else ("mid" if value <= high else "high")
    return result


def cluster_means(values: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    means = []
    for cluster in np.unique(clusters):
        cluster_values = values[clusters == cluster]
        cluster_values = cluster_values[np.isfinite(cluster_values)]
        if cluster_values.size:
            means.append(float(np.mean(cluster_values)))
    return np.asarray(means, dtype=np.float64)


def bootstrap_mean_ci(values: np.ndarray, iterations: int = 3000) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan"), float("nan")
    samples = RNG.choice(values, size=(iterations, values.size), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def subgroup_rows(case_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    summaries: list[dict] = []
    correlations: list[dict] = []
    properties = (
        "target_area_ratio",
        "boundary_complexity",
        "compactness",
        "n_classes_present",
    )
    for dataset in DATASETS:
        subset = [row for row in case_rows if row["dataset"] == dataset]
        bins = {
            "target_area_ratio": assign_tertile(subset, "target_area_ratio"),
            "boundary_complexity": assign_tertile(subset, "boundary_complexity"),
            "compactness": assign_tertile(subset, "compactness"),
        }
        for row in subset:
            row["area_group"] = bins["target_area_ratio"].get(row["sample_id"], "missing")
            row["complexity_group"] = bins["boundary_complexity"].get(row["sample_id"], "missing")
            row["compactness_group"] = bins["compactness"].get(row["sample_id"], "missing")
            n_classes = int(row["n_classes_present"])
            row["class_count_group"] = "0" if n_classes == 0 else ("1" if n_classes == 1 else "2+")
        subgroup_keys = (
            ("target_area_ratio", "area_group"),
            ("boundary_complexity", "complexity_group"),
            ("compactness", "compactness_group"),
            ("n_classes_present", "class_count_group"),
        )
        for property_name, group_key in subgroup_keys:
            for group_name in sorted({row[group_key] for row in subset}):
                group = [row for row in subset if row[group_key] == group_name]
                for metric in METRICS:
                    values = np.asarray(
                        [row[f"oriented_delta_{metric}"] for row in group], dtype=np.float64
                    )
                    clusters = np.asarray([row["cluster_id"] for row in group])
                    valid = np.isfinite(values)
                    clustered = cluster_means(values[valid], clusters[valid])
                    low, high = bootstrap_mean_ci(clustered)
                    summaries.append(
                        {
                            "dataset": dataset,
                            "property": property_name,
                            "subgroup": group_name,
                            "metric": metric,
                            "n_slices": int(valid.sum()),
                            "n_clusters": int(clustered.size),
                            "mean_oriented_delta": float(np.mean(values[valid])),
                            "median_oriented_delta": float(np.median(values[valid])),
                            "favorable_fraction": float(np.mean(values[valid] > 0)),
                            "cluster_mean_bootstrap_ci_low": low,
                            "cluster_mean_bootstrap_ci_high": high,
                        }
                    )
        for property_name in properties:
            x = np.asarray([row[property_name] for row in subset], dtype=np.float64)
            for metric in METRICS:
                y = np.asarray(
                    [row[f"oriented_delta_{metric}"] for row in subset], dtype=np.float64
                )
                valid = np.isfinite(x) & np.isfinite(y)
                result = spearmanr(x[valid], y[valid]) if valid.sum() >= 6 else None
                correlations.append(
                    {
                        "dataset": dataset,
                        "property": property_name,
                        "metric": metric,
                        "n": int(valid.sum()),
                        "spearman_rho": float(result.statistic) if result else float("nan"),
                        "p_value_uncorrected": float(result.pvalue) if result else float("nan"),
                    }
                )
    return summaries, correlations


def per_class_summary(class_rows: list[dict]) -> list[dict]:
    summaries = []
    groups: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for row in class_rows:
        groups[(row["dataset"], row["class_id"], row["class_name"])].append(row)
    for (dataset, class_id, class_name), rows in groups.items():
        for metric in METRICS:
            unet = np.asarray([row[f"unet_{metric}"] for row in rows], dtype=np.float64)
            bds = np.asarray([row[f"bds_lite_full_{metric}"] for row in rows], dtype=np.float64)
            valid = np.isfinite(unet) & np.isfinite(bds)
            raw_delta = bds[valid] - unet[valid]
            oriented = -raw_delta if metric in DISTANCE_METRICS else raw_delta
            summaries.append(
                {
                    "dataset": dataset,
                    "class_id": class_id,
                    "class_name": class_name,
                    "metric": metric,
                    "n_pairs": int(valid.sum()),
                    "unet_mean": float(np.mean(unet[valid])),
                    "bds_lite_full_mean": float(np.mean(bds[valid])),
                    "raw_delta_bds_minus_unet": float(np.mean(raw_delta)),
                    "oriented_delta_positive_is_favorable": float(np.mean(oriented)),
                    "favorable_fraction": float(np.mean(oriented > 0)),
                }
            )
    return summaries


def holm(rows: list[dict], p_key: str) -> None:
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)
    for group in by_dataset.values():
        valid = [row for row in group if math.isfinite(row[p_key])]
        ordered = sorted(valid, key=lambda row: row[p_key])
        running = 0.0
        for index, row in enumerate(ordered):
            running = max(running, (len(ordered) - index) * row[p_key])
            row["p_value_holm"] = min(1.0, running)
            row["decision_alpha_0_05"] = (
                "significant" if row["p_value_holm"] < 0.05 else "not significant"
            )


def cluster_statistics(case_rows: list[dict]) -> list[dict]:
    results = []
    for dataset in DATASETS:
        subset = [row for row in case_rows if row["dataset"] == dataset]
        for metric in METRICS:
            by_cluster: dict[str, list[float]] = defaultdict(list)
            for row in subset:
                value = row[f"oriented_delta_{metric}"]
                if math.isfinite(value):
                    by_cluster[row["cluster_id"]].append(value)
            values = np.asarray(
                [np.mean(cluster_values) for cluster_values in by_cluster.values()],
                dtype=np.float64,
            )
            if values.size >= 6 and np.any(values != 0):
                test = wilcoxon(values, alternative="two-sided", zero_method="wilcox")
                p_value = float(test.pvalue)
                statistic = float(test.statistic)
            else:
                p_value = float("nan")
                statistic = float("nan")
            low, high = bootstrap_mean_ci(values, iterations=5000)
            leave_one_out = np.asarray(
                [np.mean(np.delete(values, index)) for index in range(values.size)]
                if values.size > 1
                else [np.nan]
            )
            nonzero = values[values != 0]
            results.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "n_clusters": int(values.size),
                    "mean_oriented_delta": float(np.mean(values)),
                    "median_oriented_delta": float(np.median(values)),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "wilcoxon_statistic": statistic,
                    "p_value_two_sided": p_value,
                    "sign_effect_size": float(
                        ((nonzero > 0).sum() - (nonzero < 0).sum()) / nonzero.size
                    )
                    if nonzero.size
                    else 0.0,
                    "leave_one_cluster_out_mean_min": float(np.nanmin(leave_one_out)),
                    "leave_one_cluster_out_mean_max": float(np.nanmax(leave_one_out)),
                }
            )
    holm(results, "p_value_two_sided")
    return results


def pattern_rows(case_rows: list[dict]) -> list[dict]:
    rows = []
    for dataset in DATASETS:
        subset = [row for row in case_rows if row["dataset"] == dataset]
        conditions = {
            "dice_abs_delta_le_0.005_and_bf1_delta_gt_0.01": lambda row: (
                abs(row["delta_dsc"]) <= 0.005 and row["delta_boundary_f1"] > 0.01
            ),
            "dice_positive_but_hd95_unfavorable": lambda row: (
                row["delta_dsc"] > 0 and row["oriented_delta_hd95"] < 0
            ),
            "dice_positive_but_assd_unfavorable": lambda row: (
                row["delta_dsc"] > 0 and row["oriented_delta_assd"] < 0
            ),
            "bf1_positive_but_hd95_unfavorable": lambda row: (
                row["delta_boundary_f1"] > 0 and row["oriented_delta_hd95"] < 0
            ),
            "bf1_positive_but_assd_unfavorable": lambda row: (
                row["delta_boundary_f1"] > 0 and row["oriented_delta_assd"] < 0
            ),
        }
        for pattern, condition in conditions.items():
            eligible = [
                row
                for row in subset
                if all(
                    math.isfinite(row[key])
                    for key in (
                        "delta_dsc",
                        "delta_boundary_f1",
                        "oriented_delta_hd95",
                        "oriented_delta_assd",
                    )
                )
            ]
            count = sum(condition(row) for row in eligible)
            rows.append(
                {
                    "dataset": dataset,
                    "pattern": pattern,
                    "n_eligible": len(eligible),
                    "count": count,
                    "fraction": count / len(eligible) if eligible else float("nan"),
                }
            )
    return rows


def relationship_rows(case_rows: list[dict]) -> list[dict]:
    rows = []
    for dataset in DATASETS:
        subset = [row for row in case_rows if row["dataset"] == dataset]
        for other in ("dsc", "iou", "hd95", "assd"):
            x = np.asarray([row["delta_boundary_f1"] for row in subset], dtype=np.float64)
            y = np.asarray([row[f"oriented_delta_{other}"] for row in subset], dtype=np.float64)
            valid = np.isfinite(x) & np.isfinite(y)
            result = spearmanr(x[valid], y[valid])
            rows.append(
                {
                    "dataset": dataset,
                    "x": "delta_boundary_f1",
                    "y": f"oriented_delta_{other}",
                    "n": int(valid.sum()),
                    "spearman_rho": float(result.statistic),
                    "p_value_uncorrected": float(result.pvalue),
                }
            )
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def supporting_tables() -> dict[str, int]:
    # NOTE (public release): the original script also emitted three GSL supporting
    # tables (gsl_validation_summary.csv, gsl_test_summary.csv,
    # gsl_holm_comparisons.csv) derived from a SUPERSEDED set of GSL baseline
    # evaluations and a manuscript_v2 Holm family. Those sources are NOT part of
    # this release and did NOT feed the manuscript's reported GSL numbers, which
    # come from the matched-protocol Phase-16 rerun (see results/gsl/ and
    # results/matched_gsl_summary.csv). That branch has therefore been removed
    # here. The deterministic artifacts of record (cluster_level_statistics.csv,
    # failure_case_manifest.csv, and the per-case tables) are unaffected.
    resources = read_csv(ROOT / "results/resource_profile_isic2018_comparison.csv")
    write_csv(ANALYSIS / "deployment_resource_comparison.csv", resources)
    gate_source = read_csv(ROOT / "analysis/inputs/p1b_gate_comparison_summary.csv")
    gate_grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in gate_source:
        for metric, column in (
            ("dsc", "dice"),
            ("boundary_f1", "bf1"),
            ("hd95", "hd95"),
            ("assd", "assd"),
        ):
            gate_grouped[(row["dataset"], row["condition"], metric)].append(float(row[column]))
    gate_rows = [
        {
            "dataset": dataset,
            "condition": condition,
            "metric": metric,
            "n_seeds": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
        }
        for (dataset, condition, metric), values in sorted(gate_grouped.items())
    ]
    write_csv(ANALYSIS / "gate_removal_summary.csv", gate_rows)
    return {
        "resource_rows": len(resources),
        "gate_rows": len(gate_rows),
    }


def plot_distributions(case_rows: list[dict]) -> None:
    fig, axes = plt.subplots(3, 5, figsize=(17, 9), constrained_layout=True)
    for row_index, dataset in enumerate(DATASETS):
        subset = [row for row in case_rows if row["dataset"] == dataset]
        for column, metric in enumerate(METRICS):
            values = np.asarray(
                [row[f"oriented_delta_{metric}"] for row in subset], dtype=np.float64
            )
            values = values[np.isfinite(values)]
            axis = axes[row_index, column]
            axis.hist(values, bins=35, color="#356aa0", alpha=0.85)
            axis.axvline(0, color="black", linewidth=0.8)
            axis.axvline(np.mean(values), color="#b22222", linewidth=1.0)
            axis.set_title(f"{dataset}: {metric}")
            axis.set_xlabel("oriented delta")
            axis.set_ylabel("cases")
    fig.savefig(FIGURES / "per_case_delta_distributions.png", dpi=180)
    fig.savefig(FIGURES / "per_case_delta_distributions.pdf")
    plt.close(fig)


def plot_relationships(case_rows: list[dict]) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(10, 12), constrained_layout=True)
    for row_index, dataset in enumerate(DATASETS):
        subset = [row for row in case_rows if row["dataset"] == dataset]
        for column, metric in enumerate(("hd95", "assd")):
            x = np.asarray([row["delta_boundary_f1"] for row in subset])
            y = np.asarray([row[f"oriented_delta_{metric}"] for row in subset])
            valid = np.isfinite(x) & np.isfinite(y)
            axis = axes[row_index, column]
            axis.scatter(x[valid], y[valid], s=8, alpha=0.35)
            axis.axhline(0, color="black", linewidth=0.8)
            axis.axvline(0, color="black", linewidth=0.8)
            rho = spearmanr(x[valid], y[valid]).statistic
            axis.set_title(f"{dataset}: BF1 vs {metric} (rho={rho:.2f})")
            axis.set_xlabel("Boundary F1 delta")
            axis.set_ylabel(f"{metric} oriented delta")
    fig.savefig(FIGURES / "boundary_distance_relationships.png", dpi=180)
    fig.savefig(FIGURES / "boundary_distance_relationships.pdf")
    plt.close(fig)


def plot_per_class(rows: list[dict]) -> None:
    selected = [row for row in rows if row["metric"] in {"dsc", "boundary_f1", "hd95", "assd"}]
    datasets = ("acdc", "synapse")
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    for axis, dataset in zip(axes, datasets, strict=True):
        subset = [row for row in selected if row["dataset"] == dataset]
        classes = list(dict.fromkeys(row["class_name"] for row in subset))
        metrics = ["dsc", "boundary_f1", "hd95", "assd"]
        matrix = np.asarray(
            [
                [
                    next(
                        row["oriented_delta_positive_is_favorable"]
                        for row in subset
                        if row["class_name"] == class_name and row["metric"] == metric
                    )
                    for metric in metrics
                ]
                for class_name in classes
            ]
        )
        scale = np.nanmax(np.abs(matrix))
        image = axis.imshow(matrix, cmap="RdBu", vmin=-scale, vmax=scale, aspect="auto")
        axis.set_xticks(range(len(metrics)), metrics)
        axis.set_yticks(range(len(classes)), classes)
        axis.set_title(f"{dataset}: per-class oriented deltas")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                axis.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=7)
        fig.colorbar(image, ax=axis, fraction=0.025)
    fig.savefig(FIGURES / "per_class_oriented_deltas.png", dpi=180)
    fig.savefig(FIGURES / "per_class_oriented_deltas.pdf")
    plt.close(fig)


def consensus(dataset: str, method: str, stem: str) -> np.ndarray:
    arrays = [
        np.load(prediction_path(dataset, method, seed, stem)).astype(np.int16) for seed in SEEDS
    ]
    stack = np.stack(arrays)
    max_label = max(DATASETS[dataset]["labels"])
    counts = np.stack([(stack == label).sum(axis=0) for label in range(max_label + 1)])
    return counts.argmax(axis=0).astype(np.uint8)


def normalized_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.moveaxis(image, 0, -1)
    image = np.squeeze(image)
    low, high = np.quantile(image, [0.01, 0.99])
    image = np.clip((image - low) / max(high - low, 1e-8), 0, 1)
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    return image


def overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    colors = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.2, 0.2],
            [0.2, 0.8, 0.2],
            [0.2, 0.4, 0.9],
            [0.9, 0.7, 0.1],
            [0.7, 0.2, 0.8],
            [0.1, 0.8, 0.8],
            [0.9, 0.4, 0.1],
            [0.5, 0.5, 0.9],
        ]
    )
    result = image.copy()
    foreground = mask > 0
    result[foreground] = (1 - alpha) * result[foreground] + alpha * colors[
        mask[foreground] % len(colors)
    ]
    return result


def error_image(target: np.ndarray, pred: np.ndarray) -> np.ndarray:
    result = np.ones((*target.shape, 3), dtype=np.float32) * 0.92
    result[(target == 0) & (pred > 0)] = [0.9, 0.15, 0.15]
    result[(target > 0) & (pred == 0)] = [0.15, 0.3, 0.9]
    result[(target > 0) & (pred > 0) & (target != pred)] = [0.95, 0.75, 0.1]
    result[(target == pred) & (target > 0)] = [0.2, 0.75, 0.3]
    return result


def boundary_overlay(
    image: np.ndarray,
    target: np.ndarray,
    unet: np.ndarray,
    bds: np.ndarray,
) -> np.ndarray:
    result = image.copy()
    result[mask_to_boundary(target, radius=1).astype(bool)] = [0.1, 0.9, 0.1]
    result[mask_to_boundary(unet, radius=1).astype(bool)] = [0.95, 0.1, 0.1]
    result[mask_to_boundary(bds, radius=1).astype(bool)] = [0.1, 0.8, 0.95]
    return result


def selection_rows(case_rows: list[dict]) -> list[dict]:
    selections = []
    for dataset in DATASETS:
        subset = [row for row in case_rows if row["dataset"] == dataset]
        for row in subset:
            values = [
                row[f"oriented_delta_{metric}"]
                for metric in METRICS
                if math.isfinite(row[f"oriented_delta_{metric}"])
            ]
            row["_composite"] = float(np.mean(values))
        ordered = sorted(subset, key=lambda row: row["_composite"])
        categories = {
            "clearly_worse": ordered[:2],
            "clearly_better": ordered[-2:][::-1],
        }
        similar = [
            row
            for row in subset
            if abs(row["delta_dsc"]) <= 0.005 and math.isfinite(row["delta_boundary_f1"])
        ]
        similar.sort(key=lambda row: abs(row["delta_boundary_f1"]), reverse=True)
        categories["similar_dice_boundary_difference"] = similar[:2]
        if dataset == "synapse":
            distance = [
                row
                for row in subset
                if math.isfinite(row["oriented_delta_hd95"])
                and math.isfinite(row["oriented_delta_assd"])
            ]
            distance.sort(
                key=lambda row: min(row["oriented_delta_hd95"], row["oriented_delta_assd"])
            )
            categories["distance_metric_failure"] = distance[:3]
        if dataset == "acdc":
            negative = [row for row in subset if row["_composite"] < 0]
            median = float(np.median([row["_composite"] for row in negative]))
            negative.sort(key=lambda row: abs(row["_composite"] - median))
            categories["typical_acdc_negative"] = negative[:3]
        seen = set()
        for category, selected in categories.items():
            for rank, row in enumerate(selected, start=1):
                key = (category, row["sample_id"])
                if key in seen:
                    continue
                seen.add(key)
                selections.append(
                    {
                        "dataset": dataset,
                        "category": category,
                        "rank": rank,
                        "sample_id": row["sample_id"],
                        "cluster_id": row["cluster_id"],
                        "composite_oriented_delta": row["_composite"],
                        **{f"delta_{metric}": row[f"delta_{metric}"] for metric in METRICS},
                    }
                )
    return selections


def plot_failure_cases(selections: list[dict]) -> None:
    for selection in selections:
        dataset = selection["dataset"]
        stem = selection["sample_id"]
        split = DATASETS[dataset]["split"]
        with np.load(ROOT / f"data/processed/{dataset}/{split}/{stem}.npz") as sample:
            image = normalized_image(sample["image"])
            target = sample["mask"]
        unet = consensus(dataset, "unet", stem)
        bds = consensus(dataset, "bds_lite_full", stem)
        panels = [
            (image, "Image"),
            (overlay(image, target), "Ground truth"),
            (overlay(image, unet), "U-Net consensus"),
            (overlay(image, bds), "BDS-Lite consensus"),
            (error_image(target, unet), "U-Net errors"),
            (error_image(target, bds), "BDS-Lite errors"),
            (boundary_overlay(image, target, unet, bds), "Boundaries G/R/C"),
        ]
        fig, axes = plt.subplots(1, len(panels), figsize=(18, 3), constrained_layout=True)
        for axis, (panel, title) in zip(axes, panels, strict=True):
            axis.imshow(panel)
            axis.set_title(title, fontsize=9)
            axis.axis("off")
        fig.suptitle(
            f"{dataset} | {selection['category']} | {stem} | "
            f"dDice={selection['delta_dsc']:+.3f}, "
            f"dBF1={selection['delta_boundary_f1']:+.3f}, "
            f"dHD95={selection['delta_hd95']:+.2f}, "
            f"dASSD={selection['delta_assd']:+.2f}",
            fontsize=10,
        )
        output = FAILURE_FIGURES / (
            f"{dataset}_{selection['category']}_{selection['rank']}_{stem}.png"
        )
        fig.savefig(output, dpi=160)
        plt.close(fig)


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    FAILURE_FIGURES.mkdir(parents=True, exist_ok=True)

    case_rows, class_rows = evaluate_cases()
    distributions = distribution_rows(case_rows)
    top_cases = top_case_rows(case_rows)
    outliers = outlier_rows(case_rows)
    subgroups, property_correlations = subgroup_rows(case_rows)
    class_summary = per_class_summary(class_rows)
    cluster_stats = cluster_statistics(case_rows)
    patterns = pattern_rows(case_rows)
    relationships = relationship_rows(case_rows)
    selections = selection_rows(case_rows)
    support_counts = supporting_tables()

    write_csv(ANALYSIS / "per_case_metrics.csv", case_rows)
    write_csv(ANALYSIS / "per_class_case_metrics.csv", class_rows)
    write_csv(ANALYSIS / "per_case_distribution_summary.csv", distributions)
    write_csv(ANALYSIS / "top_case_changes.csv", top_cases)
    write_csv(ANALYSIS / "outlier_sensitivity.csv", outliers)
    write_csv(ANALYSIS / "subgroup_summary.csv", subgroups)
    write_csv(ANALYSIS / "property_delta_correlations.csv", property_correlations)
    write_csv(ANALYSIS / "per_class_summary.csv", class_summary)
    write_csv(ANALYSIS / "cluster_level_statistics.csv", cluster_stats)
    write_csv(ANALYSIS / "tradeoff_pattern_counts.csv", patterns)
    write_csv(ANALYSIS / "boundary_distance_correlations.csv", relationships)
    write_csv(ANALYSIS / "failure_case_manifest.csv", selections)

    plot_distributions(case_rows)
    plot_relationships(case_rows)
    plot_per_class(class_summary)
    plot_failure_cases(selections)

    summary = {
        "generated_at": "2026-06-06",
        "datasets": {
            dataset: {
                "split": cfg["split"],
                "n_samples": sum(row["dataset"] == dataset for row in case_rows),
                "n_clusters": len(
                    {row["cluster_id"] for row in case_rows if row["dataset"] == dataset}
                ),
            }
            for dataset, cfg in DATASETS.items()
        },
        "n_case_rows": len(case_rows),
        "n_class_case_rows": len(class_rows),
        "n_failure_figures": len(selections),
        "gsl_prediction_arrays_available": False,
        "supporting_table_counts": support_counts,
    }
    (ANALYSIS / "analysis_run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
