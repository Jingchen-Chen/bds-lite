#!/usr/bin/env python
"""Generate Figure 4: Inference resource cost comparison bar chart.

Compares U-Net, MobileViT-UNet, BDS-Lite deploy, and BDS-Lite retained
on Params, FLOPs, and FPS metrics.

Usage:
    python scripts/generate_figure4_resource_cost.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

OUTPUT_SVG = Path("paper/figures/figure4_resource_cost.svg")
OUTPUT_PDF = Path("paper/figures/figure4_resource_cost.pdf")

BSPC_COL_WIDTH_IN = 86 / 25.4

MODELS = [
    "U-Net",
    "MobileViT-UNet",
    "BDS-Lite\ndeploy",
    "BDS-Lite\nretained",
]

PARAMS_M = [1.927, 0.993, 1.928, 2.198]
FLOPS_G = [15.97, 6.98, 16.08, 19.68]
FPS = [561, 455, 565, 405]

COLORS = ["#2171b5", "#238b45", "#fd8d3c", "#d73027"]


def pct_label(val: float, ref: float) -> str:
    pct = 100 * (val - ref) / ref
    if abs(pct) < 0.1:
        return "ref"
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.0f}%"


def main() -> None:
    OUTPUT_SVG.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(BSPC_COL_WIDTH_IN * 2.2, 2.8))

    datasets = [
        ("Parameters (M)", PARAMS_M),
        ("FLOPs (G)", FLOPS_G),
        ("FPS", FPS),
    ]

    for ax, (title, values) in zip(axes, datasets, strict=True):
        x = np.arange(len(MODELS))
        bars = ax.bar(x, values, color=COLORS, width=0.55, edgecolor="black", linewidth=0.5)
        ax.set_title(title, fontsize=8, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ref = values[0]
        for bar, val in zip(bars, values, strict=True):
            label = pct_label(val, ref)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02 * max(values),
                label,
                ha="center",
                va="bottom",
                fontsize=6.5,
                color="black",
            )

    fig.suptitle(
        "ISIC2018 inference cost (CUDA, batch=1)",
        fontsize=9,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_SVG, format="svg", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUTPUT_SVG}")
    print(f"Saved: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
