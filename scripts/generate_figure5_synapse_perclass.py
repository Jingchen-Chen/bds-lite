#!/usr/bin/env python
"""Generate Figure 5: Synapse per-class HD95 delta bar chart (BDS-Lite vs U-Net).

Usage:
    python scripts/generate_figure5_synapse_perclass.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

UNET_CSV = Path("results/checkpoints/synapse_unet_baseline/eval_val.csv")
BDSLITE_CSV = Path("results/checkpoints/synapse_bdslite_boundary_gate_damped_surface/eval_val.csv")
OUTPUT_SVG = Path("paper/figures/figure5_synapse_perclass_hd95.svg")
OUTPUT_PDF = Path("paper/figures/figure5_synapse_perclass_hd95.pdf")

LABEL_NAMES = {
    1: "Aorta",
    2: "Gallbladder",
    3: "Spleen",
    4: "L. Kidney",
    5: "R. Kidney",
    6: "Liver",
    7: "Stomach",
    8: "Pancreas",
}

BSPC_COL_WIDTH_MM = 86
BSPC_COL_WIDTH_IN = BSPC_COL_WIDTH_MM / 25.4


def read_per_class(csv_path: Path) -> dict[int, float]:
    result: dict[int, float] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row["label"] == "mean":
                continue
            result[int(row["label"])] = float(row["hd95"])
    return result


def main() -> None:
    OUTPUT_SVG.parent.mkdir(parents=True, exist_ok=True)

    unet = read_per_class(UNET_CSV)
    bdslite = read_per_class(BDSLITE_CSV)

    label_ids = sorted(LABEL_NAMES.keys())
    names = [LABEL_NAMES[lid] for lid in label_ids]
    deltas = [bdslite[lid] - unet[lid] for lid in label_ids]

    colors = ["#d73027" if d > 1.0 else "#f46d43" if d > 0 else "#4dac26" for d in deltas]

    fig, ax = plt.subplots(figsize=(BSPC_COL_WIDTH_IN * 1.8, 2.6))

    x = np.arange(len(label_ids))
    bars = ax.bar(x, deltas, color=colors, width=0.6, edgecolor="black", linewidth=0.5)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, rotation=25, ha="right")
    ax.set_ylabel("HD95 delta (mm)\nBDS-Lite − U-Net", fontsize=8)
    ax.set_title(
        "Synapse per-class HD95: BDS-Lite vs U-Net",
        fontsize=9,
        fontweight="bold",
    )

    for bar, delta in zip(bars, deltas, strict=True):
        y = bar.get_height()
        va = "bottom" if y >= 0 else "top"
        offset = 0.15 if y >= 0 else -0.15
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y + offset,
            f"{delta:+.2f}",
            ha="center",
            va=va,
            fontsize=7,
        )

    ax.tick_params(axis="both", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTPUT_SVG, format="svg", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUTPUT_SVG}")
    print(f"Saved: {OUTPUT_PDF}")

    print("\nPer-class HD95 deltas (BDS-Lite − U-Net):")
    for lid in label_ids:
        delta = bdslite[lid] - unet[lid]
        mark = "WORSE" if delta > 0 else "BETTER"
        print(
            f"  {LABEL_NAMES[lid]:12s} (label {lid}): "
            f"{unet[lid]:.2f} -> {bdslite[lid]:.2f}  delta={delta:+.2f}  {mark}"
        )
    mean_delta = sum(bdslite[lid] - unet[lid] for lid in label_ids) / len(label_ids)
    mean_unet = sum(unet[lid] for lid in label_ids) / len(label_ids)
    mean_bdslite = sum(bdslite[lid] for lid in label_ids) / len(label_ids)
    print(f"  {'Mean':12s}: {mean_unet:.2f} -> {mean_bdslite:.2f}  delta={mean_delta:+.2f}")


if __name__ == "__main__":
    main()
