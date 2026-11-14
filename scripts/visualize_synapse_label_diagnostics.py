#!/usr/bin/env python
"""Create Synapse per-label diagnostic overlays for two checkpoints."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader

from bds_lite.data import NpzSegmentationDataset
from bds_lite.data.boundary import mask_to_boundary
from bds_lite.evaluation.metrics import assd, boundary_f1, dice_score, hd95, iou_score
from bds_lite.training import build_model
from bds_lite.training.runner import inference_forward, move_batch_to_device
from bds_lite.utils import seed_everything


@dataclass(frozen=True)
class CaseRecord:
    sample_id: str
    label: int
    target_pixels: int
    baseline_pixels: int
    candidate_pixels: int
    baseline_dsc: float
    candidate_dsc: float
    delta_dsc: float
    baseline_iou: float
    candidate_iou: float
    delta_iou: float
    baseline_hd95: float
    candidate_hd95: float
    delta_hd95: float
    baseline_assd: float
    candidate_assd: float
    delta_assd: float
    baseline_boundary_f1: float
    candidate_boundary_f1: float
    delta_boundary_f1: float
    image: np.ndarray
    target: np.ndarray
    baseline_pred: np.ndarray
    candidate_pred: np.ndarray

    def to_csv_row(self) -> dict[str, object]:
        row = {
            "sample_id": self.sample_id,
            "label": self.label,
            "target_pixels": self.target_pixels,
            "baseline_pixels": self.baseline_pixels,
            "candidate_pixels": self.candidate_pixels,
        }
        for metric in ("dsc", "iou", "hd95", "assd", "boundary_f1"):
            row[f"baseline_{metric}"] = getattr(self, f"baseline_{metric}")
            row[f"candidate_{metric}"] = getattr(self, f"candidate_{metric}")
            row[f"delta_{metric}"] = getattr(self, f"delta_{metric}")
        return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Synapse label-level failures for U-Net vs BDS-Lite."
    )
    parser.add_argument(
        "--baseline-checkpoint",
        default="results/checkpoints/synapse_unet_baseline/best.pt",
        help="Reference U-Net checkpoint.",
    )
    parser.add_argument(
        "--candidate-checkpoint",
        default="results/checkpoints/synapse_bdslite_boundary_gate_damped_surface_hardclass/best.pt",
        help="BDS-Lite checkpoint to diagnose.",
    )
    parser.add_argument("--split", default="val", help="Dataset split to inspect.")
    parser.add_argument(
        "--labels",
        default="6,8",
        help="Comma-separated foreground labels to diagnose.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="Number of worst cases to export per label.",
    )
    parser.add_argument(
        "--rank-metric",
        choices=["hd95", "assd"],
        default="hd95",
        help="Distance metric used to rank candidate-vs-baseline failures.",
    )
    parser.add_argument(
        "--boundary-tolerance",
        type=int,
        default=2,
        help="Pixel tolerance band for Boundary F1.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device used for inference.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/figures/synapse_label68_diagnostics",
        help="Directory for PNG overlays.",
    )
    parser.add_argument(
        "--table",
        default="results/tables/synapse_label68_diagnostics.csv",
        help="CSV path for per-case diagnostic metrics.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(name)


def parse_labels(raw: str) -> list[int]:
    labels = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not labels:
        raise ValueError("at least one label is required")
    return labels


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload["config"]
    model = build_model(config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, config


def _as_float_delta(candidate_value: float, baseline_value: float) -> float:
    if math.isnan(candidate_value) or math.isnan(baseline_value):
        return float("nan")
    return float(candidate_value - baseline_value)


def _binary_boundary(mask: np.ndarray) -> np.ndarray:
    return mask_to_boundary(mask.astype(np.uint8), radius=1)


def compute_record(
    sample_id: str,
    label: int,
    image: np.ndarray,
    target: np.ndarray,
    baseline_pred: np.ndarray,
    candidate_pred: np.ndarray,
    spacing: tuple[float, float] | None,
    boundary_tolerance: int,
) -> CaseRecord:
    baseline_boundary = _binary_boundary(baseline_pred == label)
    candidate_boundary = _binary_boundary(candidate_pred == label)
    target_boundary = _binary_boundary(target == label)

    baseline_dsc = dice_score(baseline_pred, target, label)
    candidate_dsc = dice_score(candidate_pred, target, label)
    baseline_iou = iou_score(baseline_pred, target, label)
    candidate_iou = iou_score(candidate_pred, target, label)
    baseline_hd95 = hd95(baseline_pred, target, label, spacing=spacing)
    candidate_hd95 = hd95(candidate_pred, target, label, spacing=spacing)
    baseline_assd = assd(baseline_pred, target, label, spacing=spacing)
    candidate_assd = assd(candidate_pred, target, label, spacing=spacing)
    baseline_boundary_f1 = boundary_f1(
        baseline_boundary,
        target_boundary,
        tolerance=boundary_tolerance,
    )
    candidate_boundary_f1 = boundary_f1(
        candidate_boundary,
        target_boundary,
        tolerance=boundary_tolerance,
    )

    return CaseRecord(
        sample_id=sample_id,
        label=label,
        target_pixels=int((target == label).sum()),
        baseline_pixels=int((baseline_pred == label).sum()),
        candidate_pixels=int((candidate_pred == label).sum()),
        baseline_dsc=baseline_dsc,
        candidate_dsc=candidate_dsc,
        delta_dsc=_as_float_delta(candidate_dsc, baseline_dsc),
        baseline_iou=baseline_iou,
        candidate_iou=candidate_iou,
        delta_iou=_as_float_delta(candidate_iou, baseline_iou),
        baseline_hd95=baseline_hd95,
        candidate_hd95=candidate_hd95,
        delta_hd95=_as_float_delta(candidate_hd95, baseline_hd95),
        baseline_assd=baseline_assd,
        candidate_assd=candidate_assd,
        delta_assd=_as_float_delta(candidate_assd, baseline_assd),
        baseline_boundary_f1=baseline_boundary_f1,
        candidate_boundary_f1=candidate_boundary_f1,
        delta_boundary_f1=_as_float_delta(candidate_boundary_f1, baseline_boundary_f1),
        image=image,
        target=target,
        baseline_pred=baseline_pred,
        candidate_pred=candidate_pred,
    )


def _normalize_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3:
        image = image[0]
    image = image.astype(np.float32)
    low, high = np.percentile(image, [1, 99])
    if high <= low:
        low, high = float(image.min()), float(image.max())
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    image = np.clip((image - low) / (high - low), 0.0, 1.0)
    return (image * 255).astype(np.uint8)


def _overlay_color(base: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    output = base.copy()
    mask = np.asarray(mask).astype(bool)
    if mask.any():
        color_array = np.asarray(color, dtype=np.float32)
        output[mask] = (0.35 * output[mask].astype(np.float32) + 0.65 * color_array).astype(
            np.uint8
        )
    return output


def _make_panel(
    base_rgb: np.ndarray,
    title: str,
    overlays: list[tuple[np.ndarray, tuple[int, int, int]]],
) -> Image.Image:
    panel = base_rgb.copy()
    for mask, color in overlays:
        panel = _overlay_color(panel, mask, color)
    image = Image.fromarray(panel)
    canvas = Image.new("RGB", (image.width, image.height + 28), color=(255, 255, 255))
    canvas.paste(image, (0, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 7), title, fill=(0, 0, 0))
    return canvas


def save_overlay(record: CaseRecord, output_path: Path) -> None:
    gray = _normalize_image(record.image)
    base_rgb = np.repeat(gray[:, :, None], 3, axis=2)
    label = record.label

    target_boundary = _binary_boundary(record.target == label)
    baseline_boundary = _binary_boundary(record.baseline_pred == label)
    candidate_boundary = _binary_boundary(record.candidate_pred == label)
    baseline_false_positive = np.logical_and(record.baseline_pred == label, record.target != label)
    candidate_false_positive = np.logical_and(
        record.candidate_pred == label,
        record.target != label,
    )
    candidate_false_negative = np.logical_and(
        record.candidate_pred != label,
        record.target == label,
    )

    panels = [
        _make_panel(base_rgb, "image", []),
        _make_panel(base_rgb, "GT boundary", [(target_boundary, (255, 215, 0))]),
        _make_panel(
            base_rgb,
            "U-Net pred",
            [
                (baseline_false_positive, (0, 160, 255)),
                (baseline_boundary, (0, 40, 255)),
                (target_boundary, (255, 215, 0)),
            ],
        ),
        _make_panel(
            base_rgb,
            "BDS-Lite pred",
            [
                (candidate_false_negative, (255, 140, 0)),
                (candidate_false_positive, (255, 80, 80)),
                (candidate_boundary, (220, 0, 0)),
                (target_boundary, (255, 215, 0)),
            ],
        ),
        _make_panel(
            base_rgb,
            "contours",
            [
                (target_boundary, (255, 215, 0)),
                (baseline_boundary, (0, 40, 255)),
                (candidate_boundary, (220, 0, 0)),
            ],
        ),
    ]

    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height + 42), color=(255, 255, 255))
    offset = 0
    for panel in panels:
        canvas.paste(panel, (offset, 0))
        offset += panel.width

    draw = ImageDraw.Draw(canvas)
    summary = (
        f"{record.sample_id} label {label} | "
        f"dHD95 {record.delta_hd95:.3f}, dASSD {record.delta_assd:.3f}, "
        f"dDSC {record.delta_dsc:.3f}, dBF1 {record.delta_boundary_f1:.3f}"
    )
    draw.text((6, height + 12), summary, fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _rank_value(record: CaseRecord, metric: str) -> float:
    value = getattr(record, f"delta_{metric}")
    if math.isnan(value):
        return -math.inf
    return value


def write_csv(records: list[CaseRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "label",
        "target_pixels",
        "baseline_pixels",
        "candidate_pixels",
        "baseline_dsc",
        "candidate_dsc",
        "delta_dsc",
        "baseline_iou",
        "candidate_iou",
        "delta_iou",
        "baseline_hd95",
        "candidate_hd95",
        "delta_hd95",
        "baseline_assd",
        "candidate_assd",
        "delta_assd",
        "baseline_boundary_f1",
        "candidate_boundary_f1",
        "delta_boundary_f1",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_csv_row())


def main() -> None:
    args = parse_args()
    labels = parse_labels(args.labels)
    baseline_path = Path(args.baseline_checkpoint)
    candidate_path = Path(args.candidate_checkpoint)
    output_dir = Path(args.output_dir)
    table_path = Path(args.table)
    device = resolve_device(args.device)

    baseline_model, baseline_config = load_checkpoint_model(baseline_path, device)
    candidate_model, candidate_config = load_checkpoint_model(candidate_path, device)
    seed_everything(int(candidate_config["project"].get("seed", 2026)))

    dataset_root = Path(baseline_config["dataset"]["processed_dir"])
    dataset = NpzSegmentationDataset(dataset_root, split=args.split, require_boundary=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    records: list[CaseRecord] = []
    with torch.inference_mode():
        for sample_index, batch in enumerate(loader):
            if args.max_samples is not None and sample_index >= args.max_samples:
                break
            batch = move_batch_to_device(batch, device)
            baseline_outputs = inference_forward(baseline_model, batch["image"])
            candidate_outputs = inference_forward(candidate_model, batch["image"])
            baseline_pred = baseline_outputs["seg_logits"].argmax(dim=1).cpu().numpy()[0]
            candidate_pred = candidate_outputs["seg_logits"].argmax(dim=1).cpu().numpy()[0]
            target = batch["mask"].cpu().numpy()[0]
            image = batch["image"].cpu().numpy()[0]
            sample_id_value = batch["id"][0] if isinstance(batch["id"], list) else batch["id"]
            spacing = (
                tuple(batch["spacing"].cpu().numpy()[0].tolist()) if "spacing" in batch else None
            )

            for label in labels:
                record = compute_record(
                    sample_id=str(sample_id_value),
                    label=label,
                    image=image,
                    target=target,
                    baseline_pred=baseline_pred,
                    candidate_pred=candidate_pred,
                    spacing=spacing,
                    boundary_tolerance=args.boundary_tolerance,
                )
                records.append(record)

    write_csv(records, table_path)

    exported = 0
    for label in labels:
        label_records = [
            record
            for record in records
            if record.label == label
            and record.target_pixels > 0
            and _rank_value(record, args.rank_metric) > 0
        ]
        label_records = sorted(
            label_records,
            key=lambda record: _rank_value(record, args.rank_metric),
            reverse=True,
        )
        for rank, record in enumerate(label_records[: args.top_k], start=1):
            output_path = output_dir / f"label{label}" / f"{rank:02d}_{record.sample_id}.png"
            save_overlay(record, output_path)
            exported += 1

    print(
        f"Wrote {len(records)} per-case rows to {table_path} and "
        f"{exported} overlays to {output_dir} using {device}."
    )


if __name__ == "__main__":
    main()
