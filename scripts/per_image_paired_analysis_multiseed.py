#!/usr/bin/env python
"""Multi-seed per-image paired Wilcoxon analysis.

For each (dataset, method) pair under ``outputs/evaluations/predictions/``,
this script recomputes per-image Dice, IoU, HD95, ASSD, and Boundary F1
against the local ground-truth masks for all three seeds. It then averages
the per-image metric values across seeds (per image) and runs paired
Wilcoxon tests between BDS-Lite full and vanilla U-Net within each
dataset/metric, with Holm correction over the five metrics within each
dataset.

Compared with ``per_image_paired_analysis.py`` (single-seed seed-1 audit
with ISIC2018 train/val overlap), this version:
- uses ISIC2018 val for all three seeds (image-disjoint from train);
- averages over seed-1/2/3 per-image values, so the test is a paired test
  over the three-seed mean per image (n ≈ |images| in each dataset).

Results are written under
``results/final_manuscript_audit_v2/per_image_multiseed/``.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from scipy.stats import wilcoxon

from bds_lite.data.boundary import mask_to_boundary
from bds_lite.evaluation.metrics import assd, boundary_f1, dice_score, hd95, iou_score

ROOT = Path(__file__).resolve().parents[1]
PRED_ROOT = ROOT / "outputs/evaluations/predictions"
OUT_ROOT = ROOT / "results/final_manuscript_audit_v2/per_image_multiseed"

# Dataset -> (eval split, ground-truth directories searched in order)
DATASET_CONFIG = {
    "isic2018": {
        "split": "val",
        "gt_dirs": [ROOT / "data/processed/isic2018/val"],
    },
    "acdc": {
        "split": "test",
        "gt_dirs": [ROOT / "data/processed/acdc/test"],
    },
    "synapse": {
        "split": "test",
        "gt_dirs": [ROOT / "data/processed/synapse/test"],
    },
}

DATASET_LABELS = {
    "isic2018": [1],
    "acdc": [1, 2, 3],
    "synapse": list(range(1, 9)),
}

METRICS = ("dsc", "iou", "hd95", "assd", "boundary_f1")
SEEDS = (1, 2, 3)


def find_ground_truth(dataset: str, stem: str) -> Path | None:
    for root in DATASET_CONFIG[dataset]["gt_dirs"]:
        candidate = root / f"{stem}.npz"
        if candidate.exists():
            return candidate
    return None


def per_image_label_metrics(pred: np.ndarray, target: np.ndarray, label: int, tol: int) -> dict:
    pred_b = (pred == label).astype(np.uint8)
    target_b = (target == label).astype(np.uint8)
    if not pred_b.any() and not target_b.any():
        return {m: float("nan") for m in METRICS}
    metrics = {
        "dsc": dice_score(pred, target, label),
        "iou": iou_score(pred, target, label),
        "hd95": hd95(pred, target, label),
        "assd": assd(pred, target, label),
    }
    pb = mask_to_boundary(pred_b, radius=1)
    tb = mask_to_boundary(target_b, radius=1)
    metrics["boundary_f1"] = boundary_f1(pb, tb, tolerance=tol)
    return metrics


def evaluate_pair(dataset: str, method: str, seed: int) -> dict[str, dict[str, float]]:
    cfg = DATASET_CONFIG[dataset]
    pred_dir = PRED_ROOT / f"{dataset}_{method}_seed{seed}_{cfg['split']}"
    if not pred_dir.exists():
        return {}
    rows: dict[str, dict[str, float]] = {}
    for pred_path in sorted(pred_dir.iterdir()):
        if pred_path.suffix != ".npy":
            continue
        stem = pred_path.stem
        gt_path = find_ground_truth(dataset, stem)
        if gt_path is None:
            continue
        pred = np.load(pred_path)
        with np.load(gt_path) as sample:
            target = sample["mask"]
        if pred.shape != target.shape:
            continue
        labels = DATASET_LABELS[dataset]
        per_label = {m: [] for m in METRICS}
        for label in labels:
            m = per_image_label_metrics(pred, target, label, tol=2)
            for k, v in m.items():
                per_label[k].append(v)
        agg: dict[str, float] = {}
        for k, vals in per_label.items():
            arr = np.asarray(vals, dtype=np.float64)
            agg[k] = float(np.nanmean(arr)) if not np.isnan(arr).all() else float("nan")
        rows[stem] = agg
    return rows


def paired_test(unet: np.ndarray, bds: np.ndarray) -> dict:
    mask = ~(np.isnan(unet) | np.isnan(bds))
    a = unet[mask]
    b = bds[mask]
    n = int(a.size)
    if n < 6:
        return {
            "n_pairs": n,
            "mean_unet_paired": float("nan"),
            "mean_bds_lite_full_paired": float("nan"),
            "median_unet_paired": float("nan"),
            "median_bds_lite_full_paired": float("nan"),
            "statistic": float("nan"),
            "p_value_two_sided": float("nan"),
            "median_diff_bds_minus_unet": float("nan"),
            "mean_diff_bds_minus_unet": float("nan"),
        }
    diff = b - a
    if np.count_nonzero(diff) == 0:
        return {
            "n_pairs": n,
            "mean_unet_paired": float(np.mean(a)),
            "mean_bds_lite_full_paired": float(np.mean(b)),
            "median_unet_paired": float(np.median(a)),
            "median_bds_lite_full_paired": float(np.median(b)),
            "statistic": float("nan"),
            "p_value_two_sided": 1.0,
            "median_diff_bds_minus_unet": 0.0,
            "mean_diff_bds_minus_unet": 0.0,
        }
    method = "exact" if n <= 25 else "auto"
    res = wilcoxon(b, a, alternative="two-sided", zero_method="wilcox", method=method)
    return {
        "n_pairs": n,
        "mean_unet_paired": float(np.mean(a)),
        "mean_bds_lite_full_paired": float(np.mean(b)),
        "median_unet_paired": float(np.median(a)),
        "median_bds_lite_full_paired": float(np.median(b)),
        "median_diff_bds_minus_unet": float(np.median(diff)),
        "mean_diff_bds_minus_unet": float(np.mean(diff)),
        "statistic": float(res.statistic),
        "p_value_two_sided": float(res.pvalue),
    }


def holm_correct(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["dataset"]].append(row)
    corrected: list[dict] = []
    for _ds, group in grouped.items():
        valid = [r for r in group if not np.isnan(r["p_value_two_sided"])]
        invalid = [r for r in group if np.isnan(r["p_value_two_sided"])]
        m = len(valid)
        valid_sorted = sorted(valid, key=lambda r: r["p_value_two_sided"])
        running = 0.0
        for idx, r in enumerate(valid_sorted):
            adj = (m - idx) * r["p_value_two_sided"]
            running = max(running, adj)
            r["p_value_holm"] = min(1.0, running)
            r["decision_alpha_0_05"] = (
                "significant" if r["p_value_holm"] < 0.05 else "not significant"
            )
        for r in invalid:
            r["p_value_holm"] = float("nan")
            r["decision_alpha_0_05"] = "undefined"
        corrected.extend(group)
    corrected.sort(key=lambda r: (r["dataset"], r["metric"]))
    return corrected


def _compare(cache, dataset, method_a, method_b, stems):
    rows = []
    for metric in METRICS:
        a_vals, b_vals = [], []
        for stem in stems:
            a_per_seed = [cache[(dataset, method_a, s)][stem][metric] for s in SEEDS]
            b_per_seed = [cache[(dataset, method_b, s)][stem][metric] for s in SEEDS]
            a_vals.append(
                float(np.nanmean(a_per_seed)) if not np.isnan(a_per_seed).all() else float("nan")
            )
            b_vals.append(
                float(np.nanmean(b_per_seed)) if not np.isnan(b_per_seed).all() else float("nan")
            )
        stats = paired_test(np.asarray(a_vals), np.asarray(b_vals))
        # Rename keys to reflect comparison direction
        stats["mean_a"] = stats.pop("mean_unet_paired")
        stats["mean_b"] = stats.pop("mean_bds_lite_full_paired")
        stats["median_a"] = stats.pop("median_unet_paired")
        stats["median_b"] = stats.pop("median_bds_lite_full_paired")
        stats["mean_diff_b_minus_a"] = stats.pop("mean_diff_bds_minus_unet")
        stats["median_diff_b_minus_a"] = stats.pop("median_diff_bds_minus_unet")
        rows.append(
            {
                "dataset": dataset,
                "comparison": f"{method_b}_vs_{method_a}",
                "metric": metric,
                **stats,
            }
        )
    return rows


def _holm_within_comparison(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["dataset"], r["comparison"])].append(r)
    out = []
    for _key, group in grouped.items():
        valid = [r for r in group if not np.isnan(r["p_value_two_sided"])]
        invalid = [r for r in group if np.isnan(r["p_value_two_sided"])]
        m = len(valid)
        for i, r in enumerate(sorted(valid, key=lambda r: r["p_value_two_sided"])):
            r["_rank"] = i
        running = 0.0
        for r in sorted(valid, key=lambda r: r["p_value_two_sided"]):
            adj = (m - r["_rank"]) * r["p_value_two_sided"]
            running = max(running, adj)
            r["p_value_holm"] = min(1.0, running)
            r["decision_alpha_0_05"] = (
                "significant" if r["p_value_holm"] < 0.05 else "not significant"
            )
            r.pop("_rank", None)
        for r in invalid:
            r["p_value_holm"] = float("nan")
            r["decision_alpha_0_05"] = "undefined"
        out.extend(group)
    return out


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    methods = ("unet", "bds_lite_full", "ege_unet")
    cache: dict[tuple[str, str, int], dict[str, dict[str, float]]] = {}
    for dataset in DATASET_CONFIG:
        for method in methods:
            for seed in SEEDS:
                cache[(dataset, method, seed)] = evaluate_pair(dataset, method, seed)
                print(
                    f"  {dataset}/{method}/seed{seed}: {len(cache[(dataset, method, seed)])} rows"
                )

    # Build pairwise comparisons:
    # 1) bds_lite_full vs unet  (primary positive evidence)
    # 2) bds_lite_full vs ege_unet (compare against modern compact baseline)
    # 3) ege_unet vs unet (sanity: does EGE-UNet beat vanilla U-Net?)
    pairs = [("unet", "bds_lite_full"), ("ege_unet", "bds_lite_full"), ("unet", "ege_unet")]

    paired_rows = []
    coverage_rows = []
    for dataset in DATASET_CONFIG:
        present_methods = [
            m for m in methods if cache.get((dataset, m, 1)) and len(cache[(dataset, m, 1)]) > 0
        ]
        sets_per_method = {
            m: [set(cache[(dataset, m, s)].keys()) for s in SEEDS] for m in present_methods
        }
        cov_row = {"dataset": dataset}
        for m in methods:
            cov_row[f"n_{m}_seed1"] = len(cache.get((dataset, m, 1), {}))
        for a, b in pairs:
            if a not in present_methods or b not in present_methods:
                continue
            stems = sorted(set.intersection(*(sets_per_method[a] + sets_per_method[b])))
            cov_row[f"paired_intersection_{a}_vs_{b}"] = len(stems)
            paired_rows.extend(_compare(cache, dataset, a, b, stems))
        coverage_rows.append(cov_row)

    paired_rows = _holm_within_comparison(paired_rows)
    summary_path = OUT_ROOT / "per_image_paired_wilcoxon.csv"
    coverage_path = OUT_ROOT / "per_image_coverage.csv"

    with summary_path.open("w", newline="") as h:
        w = csv.DictWriter(
            h,
            fieldnames=[
                "dataset",
                "comparison",
                "metric",
                "n_pairs",
                "mean_a",
                "mean_b",
                "median_a",
                "median_b",
                "mean_diff_b_minus_a",
                "median_diff_b_minus_a",
                "statistic",
                "p_value_two_sided",
                "p_value_holm",
                "decision_alpha_0_05",
            ],
        )
        w.writeheader()
        for r in paired_rows:
            w.writerow(
                {
                    "dataset": r["dataset"],
                    "comparison": r["comparison"],
                    "metric": r["metric"],
                    "n_pairs": r["n_pairs"],
                    "mean_a": f"{r['mean_a']:.6f}",
                    "mean_b": f"{r['mean_b']:.6f}",
                    "median_a": f"{r['median_a']:.6f}",
                    "median_b": f"{r['median_b']:.6f}",
                    "mean_diff_b_minus_a": f"{r['mean_diff_b_minus_a']:.6f}",
                    "median_diff_b_minus_a": f"{r['median_diff_b_minus_a']:.6f}",
                    "statistic": f"{r['statistic']:.6f}",
                    "p_value_two_sided": f"{r['p_value_two_sided']:.6g}",
                    "p_value_holm": f"{r['p_value_holm']:.6g}",
                    "decision_alpha_0_05": r["decision_alpha_0_05"],
                }
            )

    with coverage_path.open("w", newline="") as h:
        # union of all keys across coverage rows
        keys = []
        for r in coverage_rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        w = csv.DictWriter(h, fieldnames=keys)
        w.writeheader()
        for r in coverage_rows:
            w.writerow({k: r.get(k, "") for k in keys})

    print(f"\nWrote {summary_path.relative_to(ROOT)}")
    print(f"Wrote {coverage_path.relative_to(ROOT)}")
    print("\n--- summary ---")
    for r in paired_rows:
        print(
            f"  {r['dataset']:9s} {r['comparison']:30s} {r['metric']:11s} n={r['n_pairs']:4d} "
            f"a={r['mean_a']:.4f} b={r['mean_b']:.4f} "
            f"diff={r['mean_diff_b_minus_a']:+.4f} "
            f"p_holm={r['p_value_holm']:.3g}  {r['decision_alpha_0_05']}"
        )


if __name__ == "__main__":
    main()
