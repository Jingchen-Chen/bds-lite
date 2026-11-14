#!/usr/bin/env python
"""Per-image paired analysis from saved seed-1 prediction arrays.

For every (dataset, method) pair under ``outputs/evaluations/predictions/``,
this script recomputes per-image Dice, IoU, HD95, ASSD, and Boundary F1
against the local ground-truth masks. It then runs paired Wilcoxon tests
between BDS-Lite full and vanilla U-Net within each dataset/metric and
applies Holm correction within each dataset.

All distances are in pixels because ``data/processed/*/*.npz`` does not
record physical spacing. Results are saved as CSVs under
``results/final_manuscript_audit_v2/per_image_seed1/``.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from bds_lite.data.boundary import mask_to_boundary
from bds_lite.evaluation.metrics import (
    assd,
    boundary_f1,
    dice_score,
    hd95,
    iou_score,
)

ROOT = Path(__file__).resolve().parents[1]
PRED_ROOT = ROOT / "outputs/evaluations/predictions"
OUT_ROOT = ROOT / "results/final_manuscript_audit_v2/per_image_seed1"

# Where to look for ground-truth .npz files for each dataset (order matters).
GT_SEARCH = {
    "isic2018": [
        ROOT / "data/processed/isic2018/val",
        ROOT / "data/processed/isic2018/train",
    ],
    "acdc": [ROOT / "data/processed/acdc/test"],
    "synapse": [ROOT / "data/processed/synapse/test"],
}

# Foreground labels actually evaluable per dataset (matches training configs).
DATASET_LABELS = {
    "isic2018": [1],
    "acdc": [1, 2, 3],
    "synapse": list(range(1, 9)),
}

METRICS = ("dsc", "iou", "hd95", "assd", "boundary_f1")


def find_ground_truth(dataset: str, stem: str) -> Path | None:
    for root in GT_SEARCH[dataset]:
        candidate = root / f"{stem}.npz"
        if candidate.exists():
            return candidate
    return None


def per_image_label_metrics(pred: np.ndarray, target: np.ndarray, label: int, tol: int) -> dict:
    pred_b = (pred == label).astype(np.uint8)
    target_b = (target == label).astype(np.uint8)
    if not pred_b.any() and not target_b.any():
        # Label absent from both — treat as undefined for distance metrics.
        return {m: float("nan") for m in METRICS}
    metrics = {
        "dsc": dice_score(pred, target, label),
        "iou": iou_score(pred, target, label),
        "hd95": hd95(pred, target, label),
        "assd": assd(pred, target, label),
    }
    pred_boundary = mask_to_boundary(pred_b, radius=1)
    target_boundary = mask_to_boundary(target_b, radius=1)
    metrics["boundary_f1"] = boundary_f1(pred_boundary, target_boundary, tolerance=tol)
    return metrics


def evaluate_pair(dataset: str, method: str) -> list[dict]:
    pred_dir = PRED_ROOT / f"{dataset}_{method}_seed1_test"
    if not pred_dir.exists():
        return []
    rows: list[dict] = []
    for pred_path in sorted(pred_dir.iterdir()):
        if pred_path.suffix != ".npy":
            continue
        stem = pred_path.stem
        gt_path = find_ground_truth(dataset, stem)
        if gt_path is None:
            continue
        pred = np.load(pred_path)
        sample = np.load(gt_path)
        target = sample["mask"]
        if pred.shape != target.shape:
            continue
        labels = DATASET_LABELS[dataset]
        per_label: dict[str, list[float]] = {m: [] for m in METRICS}
        for label in labels:
            m = per_image_label_metrics(pred, target, label, tol=2)
            for k, v in m.items():
                per_label[k].append(v)
        # Per-image mean over labels that yielded a defined value.
        agg: dict[str, float] = {}
        for k, vals in per_label.items():
            arr = np.asarray(vals, dtype=np.float64)
            if np.isnan(arr).all():
                agg[k] = float("nan")
            else:
                agg[k] = float(np.nanmean(arr))
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "stem": stem,
                **{f"{m}_per_image": agg[m] for m in METRICS},
            }
        )
    return rows


def write_per_image(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def paired_test(unet: list[float], bds: list[float]) -> dict:
    paired = [(a, b) for a, b in zip(unet, bds, strict=True) if not (np.isnan(a) or np.isnan(b))]
    if len(paired) < 6:
        return {
            "n_pairs": len(paired),
            "mean_unet_paired": float("nan"),
            "mean_bds_lite_full_paired": float("nan"),
            "median_unet_paired": float("nan"),
            "median_bds_lite_full_paired": float("nan"),
            "statistic": float("nan"),
            "p_value_two_sided": float("nan"),
            "median_diff_bds_minus_unet": float("nan"),
            "mean_diff_bds_minus_unet": float("nan"),
        }
    a = np.array([p[0] for p in paired], dtype=np.float64)
    b = np.array([p[1] for p in paired], dtype=np.float64)
    diff = b - a
    nonzero = np.count_nonzero(diff)
    base = {
        "n_pairs": len(paired),
        "mean_unet_paired": float(np.mean(a)),
        "mean_bds_lite_full_paired": float(np.mean(b)),
        "median_unet_paired": float(np.median(a)),
        "median_bds_lite_full_paired": float(np.median(b)),
        "median_diff_bds_minus_unet": float(np.median(diff)),
        "mean_diff_bds_minus_unet": float(np.mean(diff)),
    }
    if nonzero == 0:
        return {
            **base,
            "statistic": float("nan"),
            "p_value_two_sided": 1.0,
            "median_diff_bds_minus_unet": 0.0,
            "mean_diff_bds_minus_unet": 0.0,
        }
    method = "exact" if len(paired) <= 25 else "auto"
    res = wilcoxon(b, a, alternative="two-sided", zero_method="wilcox", method=method)
    return {
        **base,
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
    # Preserve input ordering for output stability.
    corrected.sort(key=lambda r: (r["dataset"], r["metric"]))
    return corrected


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    pair_results: dict[tuple[str, str], list[dict]] = {}
    for dataset in GT_SEARCH:
        for method in ("unet", "bds_lite_full"):
            rows = evaluate_pair(dataset, method)
            pair_results[(dataset, method)] = rows
            write_per_image(rows, OUT_ROOT / f"{dataset}_{method}_per_image.csv")
            print(f"  computed {len(rows):5d} per-image rows for {dataset} {method}")

    paired_rows: list[dict] = []
    coverage_rows: list[dict] = []
    for dataset in GT_SEARCH:
        unet_rows = {r["stem"]: r for r in pair_results[(dataset, "unet")]}
        bds_rows = {r["stem"]: r for r in pair_results[(dataset, "bds_lite_full")]}
        stems = sorted(set(unet_rows) & set(bds_rows))
        coverage_rows.append(
            {
                "dataset": dataset,
                "unet_predictions": len(unet_rows),
                "bds_lite_full_predictions": len(bds_rows),
                "paired_intersection": len(stems),
            }
        )
        for metric in METRICS:
            unet_vals = [unet_rows[s][f"{metric}_per_image"] for s in stems]
            bds_vals = [bds_rows[s][f"{metric}_per_image"] for s in stems]
            stats = paired_test(unet_vals, bds_vals)
            paired_rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    **stats,
                    "mean_unet": stats["mean_unet_paired"],
                    "mean_bds_lite_full": stats["mean_bds_lite_full_paired"],
                    "median_unet": stats["median_unet_paired"],
                    "median_bds_lite_full": stats["median_bds_lite_full_paired"],
                    "mean_unet_all_predictions": float(np.nanmean(unet_vals)),
                    "mean_bds_lite_full_all_predictions": float(np.nanmean(bds_vals)),
                }
            )

    paired_rows = holm_correct(paired_rows)
    summary_path = OUT_ROOT / "per_image_paired_wilcoxon.csv"
    coverage_path = OUT_ROOT / "per_image_coverage.csv"

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "metric",
                "n_pairs",
                "mean_unet",
                "mean_bds_lite_full",
                "median_unet",
                "median_bds_lite_full",
                "mean_diff_bds_minus_unet",
                "median_diff_bds_minus_unet",
                "statistic",
                "p_value_two_sided",
                "p_value_holm",
                "decision_alpha_0_05",
                "mean_unet_all_predictions",
                "mean_bds_lite_full_all_predictions",
            ],
        )
        writer.writeheader()
        for row in paired_rows:
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "metric": row["metric"],
                    "n_pairs": row["n_pairs"],
                    "mean_unet": f"{row['mean_unet']:.6f}",
                    "mean_bds_lite_full": f"{row['mean_bds_lite_full']:.6f}",
                    "median_unet": f"{row['median_unet']:.6f}",
                    "median_bds_lite_full": f"{row['median_bds_lite_full']:.6f}",
                    "mean_diff_bds_minus_unet": f"{row['mean_diff_bds_minus_unet']:.6f}",
                    "median_diff_bds_minus_unet": f"{row['median_diff_bds_minus_unet']:.6f}",
                    "statistic": f"{row['statistic']:.6f}",
                    "p_value_two_sided": f"{row['p_value_two_sided']:.6g}",
                    "p_value_holm": f"{row['p_value_holm']:.6g}",
                    "decision_alpha_0_05": row["decision_alpha_0_05"],
                    "mean_unet_all_predictions": f"{row['mean_unet_all_predictions']:.6f}",
                    "mean_bds_lite_full_all_predictions": (
                        f"{row['mean_bds_lite_full_all_predictions']:.6f}"
                    ),
                }
            )

    with coverage_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "unet_predictions",
                "bds_lite_full_predictions",
                "paired_intersection",
            ],
        )
        writer.writeheader()
        writer.writerows(coverage_rows)

    print(json.dumps(paired_rows, indent=2, default=str))
    print(f"\nWrote {summary_path.relative_to(ROOT)}")
    print(f"Wrote {coverage_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
