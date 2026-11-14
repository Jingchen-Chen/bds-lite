#!/usr/bin/env python
"""Generate qualitative manuscript examples from saved seed-1 predictions."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FIGURE = ROOT / "manuscript_v2/figures/qualitative_examples_seed1.pdf"
OUTPUT_MANIFEST = ROOT / "results/final_manuscript_audit_v2/qualitative_examples_manifest.csv"

DATASETS = [
    {
        "dataset": "ISIC2018",
        "split": "val",
        "data_dir": ROOT / "data/processed/isic2018/val",
        "unet_dir": ROOT / "outputs/evaluations/predictions/isic2018_unet_seed1_test",
        "bds_dir": ROOT / "outputs/evaluations/predictions/isic2018_bds_lite_full_seed1_test",
    },
    {
        "dataset": "ACDC",
        "split": "test",
        "data_dir": ROOT / "data/processed/acdc/test",
        "unet_dir": ROOT / "outputs/evaluations/predictions/acdc_unet_seed1_test",
        "bds_dir": ROOT / "outputs/evaluations/predictions/acdc_bds_lite_full_seed1_test",
    },
    {
        "dataset": "Synapse",
        "split": "test",
        "data_dir": ROOT / "data/processed/synapse/test",
        "unet_dir": ROOT / "outputs/evaluations/predictions/synapse_unet_seed1_test",
        "bds_dir": ROOT / "outputs/evaluations/predictions/synapse_bds_lite_full_seed1_test",
    },
]

MASK_CMAP = ListedColormap(
    [
        "#0b0b0b",
        "#e41a1c",
        "#377eb8",
        "#4daf4a",
        "#984ea3",
        "#ff7f00",
        "#ffff33",
        "#a65628",
        "#f781bf",
        "#999999",
    ]
)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Return a display-ready image array."""
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.moveaxis(image, 0, -1)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]

    image = image.astype(np.float32)
    if image.ndim == 2:
        low, high = np.percentile(image, [1, 99])
        if high > low:
            image = (image - low) / (high - low)
        return np.clip(image, 0.0, 1.0)

    if image.max() > 1.0 or image.min() < 0.0:
        low, high = np.percentile(image, [1, 99])
        if high > low:
            image = (image - low) / (high - low)
    return np.clip(image, 0.0, 1.0)


def select_case(item: dict[str, Path | str]) -> dict[str, Path | str]:
    """Select a deterministic readable case with saved predictions."""
    data_dir = Path(item["data_dir"])
    unet_dir = Path(item["unet_dir"])
    bds_dir = Path(item["bds_dir"])

    candidates: list[tuple[int, str, Path, Path, Path, str]] = []
    for sample_path in sorted(data_dir.glob("*.npz")):
        stem = sample_path.stem
        unet_path = unet_dir / f"{stem}.npy"
        bds_path = bds_dir / f"{stem}.npy"
        if not unet_path.exists() or not bds_path.exists():
            continue
        sample = np.load(sample_path)
        mask = sample["mask"]
        unet = np.load(unet_path)
        bds = np.load(bds_path)
        if np.any(mask > 0) and (np.any(unet > 0) or np.any(bds > 0)):
            candidates.append(
                (
                    int(np.count_nonzero(mask)),
                    stem,
                    sample_path,
                    unet_path,
                    bds_path,
                    str(sample["sample_id"]),
                )
            )

    if candidates:
        _, _, sample_path, unet_path, bds_path, sample_id = max(candidates)
        return {
            **item,
            "sample_id": sample_id,
            "sample_path": sample_path,
            "unet_path": unet_path,
            "bds_path": bds_path,
        }

    msg = f"No qualitative case found for {item['dataset']}"
    raise RuntimeError(msg)


def main() -> None:
    selected = [select_case(item) for item in DATASETS]

    fig, axes = plt.subplots(len(selected), 4, figsize=(8.2, 5.9), constrained_layout=True)
    column_titles = ["Image", "Ground truth", "U-Net", "BDS-Lite"]
    for ax, title in zip(axes[0], column_titles, strict=True):
        ax.set_title(title, fontsize=9)

    max_label = 1
    for item in selected:
        mask = np.load(Path(item["sample_path"]))["mask"]
        unet = np.load(Path(item["unet_path"]))
        bds = np.load(Path(item["bds_path"]))
        max_label = max(max_label, int(mask.max()), int(unet.max()), int(bds.max()))

    for row, item in enumerate(selected):
        sample = np.load(Path(item["sample_path"]))
        image = normalize_image(sample["image"])
        mask = sample["mask"]
        unet = np.load(Path(item["unet_path"]))
        bds = np.load(Path(item["bds_path"]))

        axes[row, 0].imshow(image, cmap="gray" if image.ndim == 2 else None)
        for col, mask_like in enumerate([mask, unet, bds], start=1):
            axes[row, col].imshow(mask_like, cmap=MASK_CMAP, vmin=0, vmax=max_label)
        axes[row, 0].set_ylabel(str(item["dataset"]), fontsize=9)
        for col in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIGURE, bbox_inches="tight")
    plt.close(fig)

    with OUTPUT_MANIFEST.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "split",
                "sample_id",
                "sample_path",
                "unet_prediction_path",
                "bdslite_prediction_path",
                "figure_path",
            ],
        )
        writer.writeheader()
        for item in selected:
            writer.writerow(
                {
                    "dataset": item["dataset"],
                    "split": item["split"],
                    "sample_id": item["sample_id"],
                    "sample_path": Path(item["sample_path"]).relative_to(ROOT),
                    "unet_prediction_path": Path(item["unet_path"]).relative_to(ROOT),
                    "bdslite_prediction_path": Path(item["bds_path"]).relative_to(ROOT),
                    "figure_path": OUTPUT_FIGURE.relative_to(ROOT),
                }
            )

    print(f"Wrote {OUTPUT_FIGURE.relative_to(ROOT)}")
    print(f"Wrote {OUTPUT_MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
