#!/usr/bin/env python
"""Render the BDS-Lite training/deployment schematic (basis for Figure 1).

This reproduces the schematic of the train/deploy decoupling: the auxiliary
boundary decoder and training-only losses on the left (training graph) and the
retained segmentation path on the right (inference graph).

Note: the published Figure 1 in the manuscript is a polished, multi-panel
redrawing of this schematic (panels A/B/C). This script regenerates the
structural schematic from code; it is not a pixel-identical copy of the
typeset publication figure. See ``docs/paper_mapping.md``.

Run from the repository root:

    python scripts/figures/generate_architecture_figure.py

Outputs ``figures/figure_architecture_schematic.{png,pdf}``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT_PDF = ROOT / "figures/figure_architecture_schematic.pdf"
OUT_PNG = ROOT / "figures/figure_architecture_schematic.png"

BOX_STYLE = {"linewidth": 1.0, "edgecolor": "black"}
ENCODER_C = "#bcd2e8"
SEG_C = "#cfe8c4"
BND_C = "#f6cbb4"
GATE_C = "#f1e1a6"
TARGET_C = "#e2e2e2"


def box(ax, xy, w, h, text, fill, fontsize=8):
    rect = patches.FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.18",
        facecolor=fill,
        **BOX_STYLE,
    )
    ax.add_patch(rect)
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, src, dst, style="-|>", color="black", lw=1.0, ls="-"):
    ax.annotate(
        "",
        xy=dst,
        xytext=src,
        arrowprops={"arrowstyle": style, "color": color, "lw": lw, "linestyle": ls},
    )


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))

    # --- Left: Training graph -------------------------------------------------
    ax = axes[0]
    ax.set_title("Training graph", fontsize=10)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.5)
    ax.set_axis_off()

    box(ax, (0.2, 2.7), 1.6, 1.1, "Image", TARGET_C)
    box(ax, (2.3, 2.7), 1.8, 1.1, "Shared\nencoder", ENCODER_C)
    box(ax, (4.7, 4.4), 2.3, 1.1, "Segmentation\ndecoder", SEG_C)
    box(ax, (4.7, 1.0), 2.3, 1.1, "Auxiliary\nboundary decoder", BND_C)
    box(ax, (7.4, 4.4), 2.3, 1.1, "Seg. logits", SEG_C)
    box(ax, (7.4, 1.0), 2.3, 1.1, "Bnd. logits", BND_C)

    # Gate placed between encoder features and seg decoder.
    box(ax, (4.0, 3.0), 0.55, 0.45, "g", GATE_C, fontsize=8)

    # Connectors.
    arrow(ax, (1.8, 3.25), (2.3, 3.25))
    arrow(ax, (4.1, 3.5), (4.7, 4.4))
    arrow(ax, (4.1, 3.0), (4.7, 2.1))
    arrow(ax, (7.0, 4.95), (7.4, 4.95))
    arrow(ax, (7.0, 1.55), (7.4, 1.55))

    # Distillation arrow from boundary features into seg path.
    ax.annotate(
        "feature distillation",
        xy=(5.85, 4.4),
        xytext=(5.85, 2.1),
        ha="center",
        fontsize=7,
        color="#444444",
        arrowprops={
            "arrowstyle": "-|>",
            "color": "#444444",
            "lw": 0.9,
            "linestyle": "--",
        },
    )

    # Losses.
    box(ax, (8.4, 5.9), 1.4, 0.5, "L_seg", TARGET_C, fontsize=7)
    box(ax, (8.4, 0.1), 1.4, 0.5, "L_bnd", TARGET_C, fontsize=7)
    arrow(ax, (8.55, 5.6), (8.55, 5.9), color="#444444")
    arrow(ax, (8.55, 0.95), (8.55, 0.6), color="#444444")

    # --- Right: Deployment graph ----------------------------------------------
    ax2 = axes[1]
    ax2.set_title("Inference / deployment graph", fontsize=10)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 6.5)
    ax2.set_axis_off()

    box(ax2, (0.2, 2.7), 1.6, 1.1, "Image", TARGET_C)
    box(ax2, (2.3, 2.7), 1.8, 1.1, "Shared\nencoder", ENCODER_C)
    box(ax2, (4.7, 2.7), 2.3, 1.1, "Segmentation\ndecoder", SEG_C)
    box(ax2, (7.4, 2.7), 2.3, 1.1, "Seg. logits", SEG_C)
    arrow(ax2, (1.8, 3.25), (2.3, 3.25))
    arrow(ax2, (4.1, 3.25), (4.7, 3.25))
    arrow(ax2, (7.0, 3.25), (7.4, 3.25))

    # Faded auxiliary path indicating excluded boundary decoder.
    box(ax2, (4.7, 1.0), 2.3, 1.1, "Auxiliary boundary\ndecoder (excluded)", "#f7f7f7")
    arrow(
        ax2,
        (4.1, 3.0),
        (4.7, 2.1),
        color="#bdbdbd",
        ls="--",
    )

    # Retained projection + gate annotation.
    ax2.text(
        5.0,
        4.4,
        "Projection + bounded gate retained (+1,072 params)",
        fontsize=7,
        color="#444444",
    )

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PNG.relative_to(ROOT)} and {OUT_PDF.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
