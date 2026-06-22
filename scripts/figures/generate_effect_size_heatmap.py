#!/usr/bin/env python
"""Regenerate the cluster-level effect-size heatmap (manuscript Figure 2).

This is a faithful re-visualization of the **artifact of record**
``analysis/outputs/cluster_level_statistics.csv`` (which also backs Table 3). It
recomputes nothing: every value drawn here is read directly from that CSV.

For each dataset (rows: ISIC2018, ACDC, Synapse) and metric (columns grouped as
overlap = DSC, IoU; boundary = Boundary F1; surface distance = HD95, ASSD):

* the cell number is the mean oriented delta (positive = favorable for BDS-Lite);
* the cell color encodes the cluster-level sign-based effect size
  ``(n_favorable - n_unfavorable) / n_nonzero`` (green favorable, red unfavorable);
* a bold black outline marks a Holm-significant signed-rank shift
  (``p_value_holm < 0.05``) -- a reliable distributional difference, not by itself
  a final favorable decision (see Table 3, Decision column).

Run from the repository root:

    python scripts/figures/generate_effect_size_heatmap.py

Outputs ``figures/figure_cluster_effect_size_heatmap.{png,pdf}``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "analysis/outputs/cluster_level_statistics.csv"
OUT_PNG = ROOT / "figures/figure_cluster_effect_size_heatmap.png"
OUT_PDF = ROOT / "figures/figure_cluster_effect_size_heatmap.pdf"

DATASETS = ["isic2018", "acdc", "synapse"]
DATASET_LABEL = {"isic2018": "ISIC2018", "acdc": "ACDC", "synapse": "Synapse"}
# Column order follows the manuscript: overlap | boundary | surface distance.
METRICS = ["dsc", "iou", "boundary_f1", "hd95", "assd"]
METRIC_LABEL = {
    "dsc": "DSC",
    "iou": "IoU",
    "boundary_f1": "Boundary F1",
    "hd95": "HD95",
    "assd": "ASSD",
}


def main() -> None:
    df = pd.read_csv(SRC)
    delta = np.full((len(DATASETS), len(METRICS)), np.nan)
    effect = np.full_like(delta, np.nan)
    holm = np.ones_like(delta)

    for i, ds in enumerate(DATASETS):
        for j, metric in enumerate(METRICS):
            row = df[(df["dataset"] == ds) & (df["metric"] == metric)]
            if row.empty:
                continue
            delta[i, j] = float(row["mean_oriented_delta"].iloc[0])
            effect[i, j] = float(row["sign_effect_size"].iloc[0])
            holm[i, j] = float(row["p_value_holm"].iloc[0])

    fig, ax = plt.subplots(figsize=(9.0, 4.2), constrained_layout=True)
    scale = float(np.nanmax(np.abs(effect))) or 1.0
    im = ax.imshow(effect, cmap="RdYlGn", vmin=-scale, vmax=scale, aspect="auto")

    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels([METRIC_LABEL[m] for m in METRICS])
    ax.set_yticks(range(len(DATASETS)))
    ax.set_yticklabels([DATASET_LABEL[d] for d in DATASETS])

    # Group separators: overlap | boundary | surface distance.
    for x in (1.5, 2.5):
        ax.axvline(x, color="white", linewidth=3)
    sec = ax.secondary_xaxis("top")
    sec.set_xticks([0.5, 2.0, 3.5])
    sec.set_xticklabels(["overlap", "boundary", "surface distance"], fontsize=9)
    sec.tick_params(length=0)

    for i in range(len(DATASETS)):
        for j in range(len(METRICS)):
            if np.isnan(delta[i, j]):
                continue
            ax.text(
                j,
                i,
                f"{delta[i, j]:+.4f}\n(s={effect[i, j]:+.2f})",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
            if holm[i, j] < 0.05:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="black",
                        linewidth=3,
                    )
                )

    ax.set_title(
        "Cluster-level BDS-Lite vs U-Net: mean oriented Δ (cell) and sign effect (color)\n"
        "bold outline = Holm-significant signed-rank shift (p_Holm < 0.05)",
        fontsize=10,
        pad=24,
    )
    fig.colorbar(im, ax=ax, label="sign-based effect size", shrink=0.8)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=180)
    fig.savefig(OUT_PDF)
    plt.close(fig)
    print(f"Wrote {OUT_PNG.relative_to(ROOT)} and {OUT_PDF.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
