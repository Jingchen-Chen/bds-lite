#!/usr/bin/env python
"""Generate a per-seed bar plot summarizing main_test_means.

Uses the already-traced recomputed per-seed means in
``results/final_manuscript_audit_v2/recomputed_main_test_means.csv``
to render a paired per-seed bar plot for ISIC2018, ACDC, and Synapse.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/final_manuscript_audit_v2/recomputed_main_test_means.csv"
OUT = ROOT / "manuscript_v2/figures/per_seed_main_test_means.pdf"

DATASETS = ["isic2018", "acdc", "synapse"]
DATASET_LABEL = {"isic2018": "ISIC2018", "acdc": "ACDC", "synapse": "Synapse"}
METRICS = ["dsc", "assd", "boundary_f1"]
METRIC_LABEL = {"dsc": "Dice", "assd": "ASSD", "boundary_f1": "Boundary F1"}
METHODS = [("unet", "U-Net"), ("bds_lite_full", "BDS-Lite full")]
METHOD_COLORS = {"unet": "#4c72b0", "bds_lite_full": "#dd8452"}


def load_per_seed() -> dict[tuple[str, str, str], list[float]]:
    table: dict[tuple[str, str, str], list[float]] = {}
    with SRC.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (row["dataset"], row["method"], row["metric"])
            seeds = [float(v) for v in row["seed_values"].split("|")]
            table[key] = seeds
    return table


def main() -> None:
    table = load_per_seed()
    fig, axes = plt.subplots(len(METRICS), len(DATASETS), figsize=(8.4, 6.4))

    x = np.arange(3)
    bar_w = 0.36
    for r, metric in enumerate(METRICS):
        for c, dataset in enumerate(DATASETS):
            ax = axes[r, c]
            for offset_idx, (method, label) in enumerate(METHODS):
                seeds = table[(dataset, method, metric)]
                ax.bar(
                    x + (offset_idx - 0.5) * bar_w,
                    seeds,
                    width=bar_w,
                    color=METHOD_COLORS[method],
                    label=label if (r == 0 and c == 0) else None,
                    edgecolor="black",
                    linewidth=0.5,
                )
            ax.set_xticks(x)
            ax.set_xticklabels(["s1", "s2", "s3"], fontsize=9)
            if r == 0:
                ax.set_title(DATASET_LABEL[dataset], fontsize=10)
            if c == 0:
                ax.set_ylabel(METRIC_LABEL[metric], fontsize=10)
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.6)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.01),
        frameon=False,
    )
    fig.suptitle("Per-seed test means (recomputed)", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
