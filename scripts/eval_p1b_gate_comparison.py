#!/usr/bin/env python
"""Evaluate p1b BDS-Lite full vs gate-removed checkpoints on test/val splits.

Datasets / splits:
  isic2018 -> val   (no held-out test masks)
  acdc     -> test
  synapse  -> test

Both variants trained with boundary_gate=True; inference_gate='removed' means
model.boundary_gate is set to None post-load (bypasses gate in forward pass).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from bds_lite.data import NpzSegmentationDataset
from bds_lite.data.boundary import mask_to_boundary
from bds_lite.evaluation.metrics import assd, boundary_f1, dice_score, hd95, iou_score
from bds_lite.training import build_model
from bds_lite.training.runner import inference_forward, move_batch_to_device
from bds_lite.utils import seed_everything

CHECKPOINTS = [
    # ISIC2018 val
    (
        "isic2018",
        "val",
        "bdslite_full",
        1,
        "results/checkpoints/isic2018_p1b_bdslite_full_current_seed1/best.pt",
    ),
    (
        "isic2018",
        "val",
        "bdslite_full",
        2,
        "results/checkpoints/isic2018_p1b_bdslite_full_current_seed2/best.pt",
    ),
    (
        "isic2018",
        "val",
        "bdslite_full",
        3,
        "results/checkpoints/isic2018_p1b_bdslite_full_current_seed3/best.pt",
    ),
    (
        "isic2018",
        "val",
        "bdslite_gate_removed",
        1,
        "results/checkpoints/isic2018_p1b_bdslite_gate_removed_seed1/best.pt",
    ),
    (
        "isic2018",
        "val",
        "bdslite_gate_removed",
        2,
        "results/checkpoints/isic2018_p1b_bdslite_gate_removed_seed2/best.pt",
    ),
    (
        "isic2018",
        "val",
        "bdslite_gate_removed",
        3,
        "results/checkpoints/isic2018_p1b_bdslite_gate_removed_seed3/best.pt",
    ),
    # ACDC test
    (
        "acdc",
        "test",
        "bdslite_full",
        1,
        "results/checkpoints/acdc_p1b_bdslite_full_current_seed1/best.pt",
    ),
    (
        "acdc",
        "test",
        "bdslite_full",
        2,
        "results/checkpoints/acdc_p1b_bdslite_full_current_seed2/best.pt",
    ),
    (
        "acdc",
        "test",
        "bdslite_full",
        3,
        "results/checkpoints/acdc_p1b_bdslite_full_current_seed3/best.pt",
    ),
    (
        "acdc",
        "test",
        "bdslite_gate_removed",
        1,
        "results/checkpoints/acdc_p1b_bdslite_gate_removed_seed1/best.pt",
    ),
    (
        "acdc",
        "test",
        "bdslite_gate_removed",
        2,
        "results/checkpoints/acdc_p1b_bdslite_gate_removed_seed2/best.pt",
    ),
    (
        "acdc",
        "test",
        "bdslite_gate_removed",
        3,
        "results/checkpoints/acdc_p1b_bdslite_gate_removed_seed3/best.pt",
    ),
    # Synapse test
    (
        "synapse",
        "test",
        "bdslite_full",
        1,
        "results/checkpoints/synapse_p1b_bdslite_full_current_seed1/best.pt",
    ),
    (
        "synapse",
        "test",
        "bdslite_full",
        2,
        "results/checkpoints/synapse_p1b_bdslite_full_current_seed2/best.pt",
    ),
    (
        "synapse",
        "test",
        "bdslite_full",
        3,
        "results/checkpoints/synapse_p1b_bdslite_full_current_seed3/best.pt",
    ),
    (
        "synapse",
        "test",
        "bdslite_gate_removed",
        1,
        "results/checkpoints/synapse_p1b_bdslite_gate_removed_seed1/best.pt",
    ),
    (
        "synapse",
        "test",
        "bdslite_gate_removed",
        2,
        "results/checkpoints/synapse_p1b_bdslite_gate_removed_seed2/best.pt",
    ),
    (
        "synapse",
        "test",
        "bdslite_gate_removed",
        3,
        "results/checkpoints/synapse_p1b_bdslite_gate_removed_seed3/best.pt",
    ),
]

OUTPUT_DIR = Path("results/post_recovery_eval/p1b_gate_comparison")
BOUNDARY_TOLERANCE = 2


def _binary_boundary(mask: np.ndarray) -> np.ndarray:
    return mask_to_boundary(mask.astype(np.uint8), radius=1)


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(arr)) if arr.size and not np.isnan(arr).all() else float("nan")


def _foreground_labels(num_classes: int) -> list[int]:
    return list(range(1, num_classes)) if num_classes > 1 else [1]


def evaluate_checkpoint(split: str, checkpoint_path: Path) -> dict:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload["config"]
    seed_everything(int(config["project"].get("seed", 2026)))

    dataset_dir = Path(config["dataset"]["processed_dir"])
    loader_ds = NpzSegmentationDataset(dataset_dir, split=split, require_boundary=False)
    loader = DataLoader(loader_ds, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    model.load_state_dict(payload["model"], strict=True)

    model_cfg = config.get("model", {})
    if model_cfg.get("inference_gate") == "removed" and hasattr(model, "boundary_gate"):
        model.boundary_gate = None

    model.eval()

    num_classes = int(
        config.get("model", {}).get("num_classes", config.get("dataset", {}).get("num_classes", 2))
    )
    foreground_labels = _foreground_labels(num_classes)
    per_label: dict[int, dict[str, list[float]]] = {
        label: {"dsc": [], "iou": [], "hd95": [], "assd": [], "boundary_f1": []}
        for label in foreground_labels
    }

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = inference_forward(model, batch["image"])
            pred = outputs["seg_logits"].argmax(dim=1).cpu().numpy()[0]
            target = batch["mask"].cpu().numpy()[0]
            spacing = (
                tuple(batch["spacing"].cpu().numpy()[0].tolist()) if "spacing" in batch else None
            )
            for label in foreground_labels:
                per_label[label]["dsc"].append(dice_score(pred, target, label))
                per_label[label]["iou"].append(iou_score(pred, target, label))
                per_label[label]["hd95"].append(hd95(pred, target, label, spacing=spacing))
                per_label[label]["assd"].append(assd(pred, target, label, spacing=spacing))
                pred_b = _binary_boundary(pred == label)
                tgt_b = _binary_boundary(target == label)
                per_label[label]["boundary_f1"].append(
                    boundary_f1(pred_b, tgt_b, tolerance=BOUNDARY_TOLERANCE)
                )

    per_label_summary = {
        label: {m: _nanmean(vals) for m, vals in metrics.items()}
        for label, metrics in per_label.items()
    }
    overall = {
        m: _nanmean([per_label_summary[label][m] for label in foreground_labels])
        for m in ("dsc", "iou", "hd95", "assd", "boundary_f1")
    }
    return {
        "split": split,
        "num_samples": len(loader_ds),
        "per_label": per_label_summary,
        "mean": overall,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for ds, split, cond, seed, ckpt_rel in CHECKPOINTS:
        ckpt_path = Path(ckpt_rel)
        print(f"\n--- {ds} {cond} seed{seed}: {ckpt_path} ---")
        report = evaluate_checkpoint(split, ckpt_path)
        m = report["mean"]
        print(json.dumps(m, indent=2))

        out_json = OUTPUT_DIR / f"{ds}_{cond}_seed{seed}.json"
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote {out_json}")

        summary_rows.append(
            {
                "dataset": ds,
                "condition": cond,
                "seed": seed,
                "dice": m["dsc"],
                "bf1": m["boundary_f1"],
                "hd95": m["hd95"],
                "assd": m["assd"],
            }
        )

    summary_path = OUTPUT_DIR / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["dataset", "condition", "seed", "dice", "bf1", "hd95", "assd"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nWrote {summary_path}")

    print("\n=== 3-seed mean/std per dataset ===")
    for ds in ["isic2018", "acdc", "synapse"]:
        for cond in ["bdslite_full", "bdslite_gate_removed"]:
            rows = [r for r in summary_rows if r["dataset"] == ds and r["condition"] == cond]
            for metric in ("dice", "bf1", "hd95", "assd"):
                vals = [r[metric] for r in rows]
                print(f"  {ds} {cond} {metric}: mean={np.mean(vals):.6f} std={np.std(vals):.6f}")


if __name__ == "__main__":
    main()
