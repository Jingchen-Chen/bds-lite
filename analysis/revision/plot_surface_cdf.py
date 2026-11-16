"""Render exact right-continuous ECDFs from stored distance tables."""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = {
    "U": ("U-Net", "C0", "-"),
    "main_p1b_unet": ("U-Net", "C0", "-"),
    "F": ("Full", "C1", "-"),
    "main_p1b_full": ("Full", "C1", "-"),
    "S": ("SDF only", "C2", "-"),
    "legacy_gsl": ("Legacy GSL", "C3", "--"),
    "attention_reference": ("Attention (s2026)", "C4", ":"),
}


def main():
    directory = ROOT / "results/surface_cdf"
    curves = defaultdict(list)
    source = directory / "full_cdf.csv"
    for row in csv.DictReader(source.open()):
        curves[row["dataset"], row["family"], row["seed"]].append(
            (int(row["distance_squared"]), float(row["equal_image_class_cdf"]))
        )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True, layout="constrained")
    datasets = ["isic2018", "acdc", "synapse"]
    for ax, dataset in zip(axes, datasets, strict=True):
        groups = defaultdict(list)
        for (ds, family, _seed), rows in curves.items():
            if ds != dataset:
                continue
            values = np.array(rows)
            grid = np.arange(10001)
            indices = np.searchsorted(values[:, 0], grid, side="right") - 1
            cdf = np.where(indices >= 0, values[np.maximum(indices, 0), 1], 0)
            groups[family].append(cdf)
        for family in sorted(groups, key=lambda f: FAMILIES[f][1]):
            label, color, style = FAMILIES[family]
            ax.step(
                np.sqrt(np.arange(10001)),
                np.mean(groups[family], axis=0),
                where="post",
                label=label,
                color=color,
                linestyle=style,
            )
        ax.set(
            xlim=(0, 100), ylim=(0, 1.01), title=dataset, xlabel="Surface distance (processed px)"
        )
        ax.set_xscale("symlog", linthresh=2)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Equal valid image/class CDF, mean across seeds")
    for suffix in ["pdf", "png"]:
        path = directory / f"surface_distance_cdf.{suffix}"
        initial = directory / f"initial_linear_render.{suffix}"
        if not initial.exists():
            path.rename(initial)
        fig.savefig(path, dpi=180)
    plt.close(fig)
    receipt = dict(
        source="full_cdf.csv",
        sha256=hashlib.file_digest(source.open("rb"), "sha256").hexdigest(),
        plot="right-continuous steps, consistent colors/labels, zero distance retained",
        previous_render="initial_linear_render.pdf/png preserved; not final figure",
        scope="descriptive; separate seed2026 Attention and legacy GSL references",
    )
    (directory / "plot_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
