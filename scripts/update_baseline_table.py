#!/usr/bin/env python
"""Update baseline_results_month1.csv with newly trained baseline checkpoints.

Scans for eval_val.csv files from known baseline experiments and adds or
overwrites their rows in the consolidated baseline table.

Usage:
    python scripts/update_baseline_table.py
"""

from __future__ import annotations

import csv
from pathlib import Path

CHECKPOINTS_DIR = Path("results/checkpoints")
TABLE_PATH = Path("results/tables/baseline_results_month1.csv")

BASELINE_MAP: dict[str, dict[str, str]] = {
    "isic2018_unet_baseline": {
        "dataset": "ISIC2018",
        "method": "U-Net",
        "split": "val",
    },
    "acdc_unet_baseline": {
        "dataset": "ACDC",
        "method": "U-Net",
        "split": "val",
    },
    "synapse_unet_baseline": {
        "dataset": "Synapse",
        "method": "U-Net",
        "split": "val",
    },
    "isic2018_attention_unet_baseline": {
        "dataset": "ISIC2018",
        "method": "Attention U-Net",
        "split": "val",
    },
    "acdc_attention_unet_baseline": {
        "dataset": "ACDC",
        "method": "Attention U-Net",
        "split": "val",
    },
    "synapse_attention_unet_baseline": {
        "dataset": "Synapse",
        "method": "Attention U-Net",
        "split": "val",
    },
    "isic2018_mobilevit_unet": {
        "dataset": "ISIC2018",
        "method": "MobileViT-UNet",
        "split": "val",
    },
}

METRICS = ["dsc", "iou", "hd95", "assd", "boundary_f1"]
FIELDNAMES = ["dataset", "method", "split", "num_samples", *METRICS, "checkpoint", "notes"]


def _read_mean_metrics(csv_path: Path) -> dict[str, float] | None:
    if not csv_path.exists():
        return None
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    mean_rows = [r for r in rows if r.get("label", "") == "mean"]
    row = mean_rows[0] if mean_rows else rows[-1]
    return {m: float(row[m]) for m in METRICS if m in row}


DATASET_SPLIT_SAMPLES: dict[str, int] = {
    "ISIC2018_val": 519,
    "ACDC_val": 186,
    "Synapse_val": 473,
}


def _num_samples(dataset: str, split: str) -> int:
    return DATASET_SPLIT_SAMPLES.get(f"{dataset}_{split}", 0)


def main() -> None:
    existing: dict[tuple[str, str], dict[str, str]] = {}
    if TABLE_PATH.exists():
        with TABLE_PATH.open() as f:
            for row in csv.DictReader(f):
                key = (row["dataset"], row["method"])
                existing[key] = row

    updated = 0
    for exp_name, meta in BASELINE_MAP.items():
        eval_csv = CHECKPOINTS_DIR / exp_name / "eval_val.csv"
        metrics = _read_mean_metrics(eval_csv)
        if metrics is None:
            print(f"[SKIP] {exp_name}: no eval_val.csv yet")
            continue

        ckpt = CHECKPOINTS_DIR / exp_name / "best.pt"
        n = _num_samples(meta["dataset"], meta["split"])

        key = (meta["dataset"], meta["method"])
        row: dict[str, str] = {
            "dataset": meta["dataset"],
            "method": meta["method"],
            "split": meta["split"],
            "num_samples": str(n),
            **{m: str(metrics[m]) for m in METRICS},
            "checkpoint": str(ckpt),
            "notes": f"best checkpoint from {exp_name}",
        }
        action = "updated" if key in existing else "added"
        existing[key] = row
        print(f"[{action.upper()}] {meta['dataset']} {meta['method']}: DSC={metrics['dsc']:.4f}")
        updated += 1

    if updated == 0:
        print("No new results to update.")
        return

    rows = sorted(existing.values(), key=lambda r: (r["dataset"], r["method"]))
    with TABLE_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nUpdated {TABLE_PATH} with {updated} change(s).")


if __name__ == "__main__":
    main()
