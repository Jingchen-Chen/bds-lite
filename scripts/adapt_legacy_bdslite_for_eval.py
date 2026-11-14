#!/usr/bin/env python
"""Evaluate legacy BDS-Lite checkpoints (model.name='bds_lite') against a dataset split.

The historical ablation checkpoints use model.name='bds_lite' and
projection_head.{weight,bias} keys.  Current code registers 'bdslite_unet'
and names the same layer seg_to_boundary_feature.  This adapter:

  1. Patches the config model name to 'bdslite_unet'.
  2. Renames projection_head.* -> seg_to_boundary_feature.* in the state dict.
  3. Loads with strict=True — halts if that fails.
  4. Evaluates on the requested split using the same logic as scripts/evaluate.py.

Do NOT modify committed source code or checkpoints.
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
    "outputs/runs/isic2018_bds_lite_full_seed1/best.pt",
    "outputs/runs/isic2018_bds_lite_full_seed2/best.pt",
    "outputs/runs/isic2018_bds_lite_full_seed3/best.pt",
]

OUTPUT_DIR = Path("results/post_recovery_eval/isic2018_val")
SPLIT = "val"
BOUNDARY_TOLERANCE = 2


def _rename_projection_head(state_dict: dict) -> dict:
    renamed = {}
    for k, v in state_dict.items():
        new_k = (
            k.replace("projection_head.", "seg_to_boundary_feature.", 1)
            if k.startswith("projection_head.")
            else k
        )
        renamed[new_k] = v
    return renamed


def _patch_config(config: dict) -> dict:
    import copy

    cfg = copy.deepcopy(config)
    model_cfg = cfg.get("model", {})
    if model_cfg.get("name") == "bds_lite":
        model_cfg["name"] = "bdslite_unet"
        # map historical keys to current BDSLiteUNet constructor params
        if "use_boundary_gate" in model_cfg and "boundary_gate" not in model_cfg:
            model_cfg["boundary_gate"] = model_cfg["use_boundary_gate"]
    cfg["model"] = model_cfg
    return cfg


def _binary_boundary(mask: np.ndarray) -> np.ndarray:
    return mask_to_boundary(mask.astype(np.uint8), radius=1)


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(arr)) if arr.size and not np.isnan(arr).all() else float("nan")


def _foreground_labels(num_classes: int) -> list[int]:
    return list(range(1, num_classes)) if num_classes > 1 else [1]


def evaluate_checkpoint(checkpoint_path: Path) -> dict:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = _patch_config(payload["config"])
    seed_everything(int(config["project"].get("seed", 2026)))

    dataset = NpzSegmentationDataset(
        Path(config["dataset"]["processed_dir"]),
        split=SPLIT,
        require_boundary=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)

    state_dict = _rename_projection_head(payload["model"])
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    # strict=True would have raised on mismatch; belt-and-suspenders check
    if missing or unexpected:
        raise RuntimeError(
            f"strict=True succeeded but reported missing={missing} unexpected={unexpected}"
        )

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
        "experiment": config.get("experiment", {}).get("name", checkpoint_path.parent.name),
        "split": SPLIT,
        "num_samples": len(dataset),
        "per_label": per_label_summary,
        "mean": overall,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for ckpt_rel in CHECKPOINTS:
        ckpt_path = Path(ckpt_rel)
        seed = ckpt_path.parent.name.split("_seed")[-1]
        print(f"\n--- seed {seed}: {ckpt_path} ---")

        report = evaluate_checkpoint(ckpt_path)
        out_csv = OUTPUT_DIR / f"bdslite_full_seed{seed}_val.json"
        fieldnames = ["label", "dsc", "iou", "hd95", "assd", "boundary_f1"]
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for label, metrics in report["per_label"].items():
                writer.writerow({"label": label, **metrics})
            writer.writerow({"label": "mean", **report["mean"]})

        print(json.dumps(report["mean"], indent=2))
        print(f"Wrote {out_csv}")
        results.append((seed, report["mean"]))

    # Print 3-seed summary
    print("\n=== 3-seed summary ===")
    for metric in ("dsc", "iou", "hd95", "assd", "boundary_f1"):
        vals = [r[metric] for _, r in results]
        print(f"  {metric}: mean={np.mean(vals):.6f} std={np.std(vals):.6f}")


if __name__ == "__main__":
    main()
