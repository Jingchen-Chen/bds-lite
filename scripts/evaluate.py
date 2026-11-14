#!/usr/bin/env python
"""Evaluate a trained BDS-Lite checkpoint on a dataset split.

Reports per-class Dice, IoU, HD95, ASSD, and Boundary F1, plus the
unweighted means over foreground classes. Results are written to a CSV
file alongside the checkpoint and printed to stdout.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi
from torch.utils.data import DataLoader

from bds_lite.data import NpzSegmentationDataset
from bds_lite.evaluation.metrics import assd, boundary_f1, dice_score, hd95, iou_score
from bds_lite.training import build_model
from bds_lite.training.runner import inference_forward, move_batch_to_device
from bds_lite.utils import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained BDS-Lite checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to a `.pt` checkpoint.")
    parser.add_argument(
        "--split", default="test", help="Dataset split to evaluate (defaults to `test`)."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV output path. Defaults to <checkpoint_dir>/eval_<split>.csv.",
    )
    parser.add_argument(
        "--boundary-tolerance",
        type=int,
        default=2,
        help="Pixel tolerance band for the Boundary F1 score.",
    )
    parser.add_argument(
        "--keep-largest-component",
        action="store_true",
        help="Keep only the largest predicted connected component for each foreground class.",
    )
    return parser.parse_args()


def _foreground_labels(num_classes: int) -> list[int]:
    return list(range(1, num_classes)) if num_classes > 1 else [1]


def _binary_boundary(mask: np.ndarray) -> np.ndarray:
    from bds_lite.data.boundary import mask_to_boundary

    return mask_to_boundary(mask.astype(np.uint8), radius=1)


def _keep_largest_components(pred: np.ndarray, labels: Iterable[int]) -> np.ndarray:
    filtered = np.asarray(pred).copy()
    for label in labels:
        label_mask = filtered == label
        if not label_mask.any():
            continue

        components, num_components = ndi.label(label_mask)
        if num_components <= 1:
            continue

        component_sizes = np.bincount(components.ravel())
        component_sizes[0] = 0
        largest_component = int(component_sizes.argmax())
        filtered[np.logical_and(label_mask, components != largest_component)] = 0

    return filtered


def _nanmean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or np.isnan(array).all():
        return float("nan")
    return float(np.nanmean(array))


def evaluate_checkpoint(
    checkpoint_path: Path,
    split: str,
    boundary_tolerance: int,
    keep_largest_component: bool = False,
) -> dict:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload["config"]
    seed_everything(int(config["project"].get("seed", 2026)))

    dataset = NpzSegmentationDataset(
        Path(config["dataset"]["processed_dir"]),
        split=split,
        require_boundary=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    model.load_state_dict(payload["model"])
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
            if keep_largest_component:
                pred = _keep_largest_components(pred, foreground_labels)
            spacing = (
                tuple(batch["spacing"].cpu().numpy()[0].tolist()) if "spacing" in batch else None
            )

            for label in foreground_labels:
                per_label[label]["dsc"].append(dice_score(pred, target, label))
                per_label[label]["iou"].append(iou_score(pred, target, label))
                per_label[label]["hd95"].append(hd95(pred, target, label, spacing=spacing))
                per_label[label]["assd"].append(assd(pred, target, label, spacing=spacing))
                pred_boundary = _binary_boundary(pred == label)
                target_boundary = _binary_boundary(target == label)
                per_label[label]["boundary_f1"].append(
                    boundary_f1(pred_boundary, target_boundary, tolerance=boundary_tolerance)
                )

    per_label_summary: dict[int, dict[str, float]] = {}
    for label, metrics in per_label.items():
        per_label_summary[label] = {
            metric_name: _nanmean(values) for metric_name, values in metrics.items()
        }
    overall = {
        metric_name: _nanmean(per_label_summary[label][metric_name] for label in foreground_labels)
        for metric_name in ("dsc", "iou", "hd95", "assd", "boundary_f1")
    }

    return {
        "experiment": config["experiment"]["name"],
        "split": split,
        "num_samples": len(dataset),
        "postprocess": "largest_component" if keep_largest_component else "none",
        "per_label": per_label_summary,
        "mean": overall,
    }


def write_csv(report: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["label", "dsc", "iou", "hd95", "assd", "boundary_f1"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for label, metrics in report["per_label"].items():
            writer.writerow({"label": label, **metrics})
        writer.writerow({"label": "mean", **report["mean"]})


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    output_path = (
        Path(args.output)
        if args.output
        else checkpoint_path.parent
        / f"eval_{args.split}{'_lcc' if args.keep_largest_component else ''}.csv"
    )

    report = evaluate_checkpoint(
        checkpoint_path,
        args.split,
        args.boundary_tolerance,
        keep_largest_component=args.keep_largest_component,
    )
    write_csv(report, output_path)
    print(json.dumps(report, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
