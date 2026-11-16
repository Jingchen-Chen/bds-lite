"""Figures that replace every dataset-image panel (docs/LICENSING_AUDIT.md).

No dataset photograph, intensity image, or ACDC/Synapse annotation is read for drawing.
  targets          synthetic masks run through the released target code: class mask, union
                   boundary target, signed-distance target -> paper/figures/synthetic_targets.pdf
  error_geometry   error pixels per image by type and location for four retrained arms, from
                   results/tables/families/error_bands.csv -> paper/figures/error_geometry.pdf
  structures       missed structures by reference area and per-class DSC on ACDC and Synapse, from
                   collapse_by_area.csv and per_class_dsc.csv
                   -> paper/figures/structure_errors.pdf
All inputs except the synthetic masks are aggregate tables of seeds 1-3.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from bds_lite.data.boundary import mask_to_boundary, mask_to_sdf  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FAM = ROOT / "results/tables/families"
FIG = ROOT / "paper/figures"
INK, MUTED, GRID = "#2b2b2b", "#52514e", "#e9e8e4"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
ARMS = {"M0": "U-Net", "M2": "Decoder", "M6": "BDS-Lite", "GSL": "U-Net+GSL"}
ARM_COLORS = {"M0": INK, "M2": BLUE, "M6": ORANGE, "GSL": AQUA}
ARM_MARKERS = {"M0": "o", "M2": "s", "M6": "D", "GSL": "^"}
NAMES = {"isic2018": "ISIC2018", "acdc": "ACDC", "synapse": "Synapse"}
plt.rcParams.update(
    {"font.size": 8, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.linewidth": 0.6}
)


def blob(size: int = 224) -> np.ndarray:
    """One irregular object, like a single lesion."""
    yy, xx = np.mgrid[:size, :size] - size / 2
    angle, radius = np.arctan2(yy, xx), np.hypot(yy, xx)
    edge = 62 + 9 * np.sin(3 * angle + 0.4) + 5 * np.cos(5 * angle) + 3 * np.sin(9 * angle + 1.1)
    return (radius < edge).astype(np.int64)


def adjacent(size: int = 224) -> np.ndarray:
    """Three touching structures: a cavity, a ring around it, and a crescent beside the ring."""
    yy, xx = np.mgrid[:size, :size]
    mask = np.zeros((size, size), np.int64)
    centre = np.hypot(yy - 118, xx - 126)
    side = np.hypot((yy - 112) / 1.25, xx - 70)
    mask[(side < 46) & (centre >= 52)] = 1
    mask[(centre < 52) & (centre >= 36)] = 2
    mask[centre < 36] = 3
    return mask


def separate(size: int = 224) -> np.ndarray:
    """Several separate structures of very different size, one of them tiny."""
    yy, xx = np.mgrid[:size, :size]
    mask = np.zeros((size, size), np.int64)
    shapes = [
        (1, 80, 70, 52, 38),
        (2, 150, 160, 22, 30),
        (3, 70, 170, 20, 26),
        (4, 120, 112, 7, 7),
        (5, 175, 70, 14, 34),
        (6, 40, 120, 4, 4),
    ]
    for label, cy, cx, ry, rx in shapes:
        mask[((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 < 1] = label
    return mask


def targets() -> None:
    rows = [
        (blob(), "One object"),
        (adjacent(), "Adjacent structures"),
        (separate(), "Separate structures,\nsizes differ"),
    ]
    classes = ListedColormap(["#ffffff", BLUE, ORANGE, AQUA, YELLOW, "#e87ba4", "#6250d6"])
    diverging = LinearSegmentedColormap.from_list("sdf", [BLUE, "#f2f1ee", ORANGE])
    fig, axes = plt.subplots(3, 3, figsize=(5.4, 5.5))
    for i, (mask, label) in enumerate(rows):
        boundary = mask_to_boundary(mask)
        sdf = mask_to_sdf(mask).min(axis=0)
        norm = TwoSlopeNorm(vmin=float(sdf.min()), vcenter=0.0, vmax=float(sdf.max()))
        axes[i, 0].imshow(mask, cmap=classes, vmin=0, vmax=6, interpolation="nearest")
        axes[i, 1].imshow(boundary, cmap=ListedColormap(["#ffffff", INK]), interpolation="nearest")
        image = axes[i, 2].imshow(sdf, cmap=diverging, norm=norm)
        axes[i, 0].set_ylabel(label, fontsize=7.5, color=INK)
        colorbar = fig.colorbar(image, ax=axes[i, 2], fraction=0.046, pad=0.03)
        colorbar.ax.tick_params(labelsize=6, colors=MUTED, length=2)
        colorbar.outline.set_linewidth(0.4)
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(GRID)
    for ax, title in zip(
        axes[0],
        ("Class mask", "Boundary target $y_b$", "Signed distance $\\phi$ (px)"),
        strict=True,
    ):
        ax.set_title(title, fontsize=8, color=INK)
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    fig.savefig(FIG / "synthetic_targets.pdf")
    fig.savefig(FIG / "synthetic_targets.png", dpi=220)
    plt.close(fig)


def error_geometry() -> None:
    bands = pd.read_csv(FAM / "error_bands.csv").set_index(["dataset", "method"])
    parts = [
        ("fn_near_per_image", BLUE, "////", "False negative, within 2 px"),
        ("fn_far_per_image", BLUE, None, "False negative, farther"),
        ("fp_near_per_image", ORANGE, "////", "False positive, within 2 px"),
        ("fp_far_per_image", ORANGE, None, "False positive, farther"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
    for ax, dataset in zip(axes, NAMES, strict=True):
        arms = list(ARMS)[::-1]
        for y, arm in enumerate(arms):
            row = bands.loc[(dataset, arm)]
            left = 0.0
            for column, color, hatch, _ in parts:
                width = float(row[column])
                face = "#ffffff" if hatch else color
                ax.barh(
                    y,
                    width,
                    left=left,
                    height=0.62,
                    color=face,
                    edgecolor=color,
                    hatch=hatch,
                    linewidth=0.6,
                )
                left += width
            ax.text(left, y, f" {left:,.0f}", va="center", ha="left", fontsize=6.5, color=MUTED)
        ax.set_yticks(range(len(arms)), [ARMS[a] for a in arms])
        ax.set_title(NAMES[dataset], fontsize=8, color=INK, loc="left")
        ax.set_xlabel("Error pixels per image", color=MUTED)
        ax.set_xlim(0, ax.get_xlim()[1] * 1.22)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, length=2.5, width=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    legend = [
        Patch(facecolor="#ffffff" if h else c, edgecolor=c, hatch=h, label=t)
        for _, c, h, t in parts
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.12, 1, 1), w_pad=1.0)
    fig.savefig(FIG / "error_geometry.pdf")
    fig.savefig(FIG / "error_geometry.png", dpi=220)
    plt.close(fig)


def structures() -> None:
    area = pd.read_csv(FAM / "collapse_by_area.csv")
    per_class = pd.read_csv(FAM / "per_class_dsc.csv")
    bins = ["<1%", "1-2%", "2-5%", ">5%"]
    fig, axes = plt.subplots(
        1, 3, figsize=(7.2, 2.5), gridspec_kw=dict(width_ratios=[1.0, 0.8, 1.6])
    )
    offsets = dict(zip(ARMS, (-0.24, -0.08, 0.08, 0.24), strict=True))
    ax = axes[0]
    for arm in ARMS:
        part = area[(area.dataset == "synapse") & (area.method == arm)].set_index("area_bin")
        values = [100 * float(part.loc[b, "missed_rate"]) for b in bins]
        ax.plot(
            np.arange(4) + offsets[arm],
            values,
            ARM_MARKERS[arm],
            color=ARM_COLORS[arm],
            markersize=4.5,
            markerfacecolor="white" if arm == "M0" else ARM_COLORS[arm],
        )
    ax.set_xticks(range(4), bins)
    ax.set_xlabel("Reference area (share of image)", color=MUTED)
    ax.set_ylabel("Structures missed entirely (%)", color=MUTED)
    ax.set_title("Synapse: missed structures", fontsize=8, color=INK, loc="left")
    for ax, dataset in zip(axes[1:], ("acdc", "synapse"), strict=True):
        labels = sorted(per_class[per_class.dataset == dataset].label.unique())
        for arm in ARMS:
            part = per_class[(per_class.dataset == dataset) & (per_class.method == arm)].set_index(
                "label"
            )
            values = [float(part.loc[k, "dsc_present"]) for k in labels]
            ax.plot(
                np.arange(len(labels)) + offsets[arm],
                values,
                ARM_MARKERS[arm],
                color=ARM_COLORS[arm],
                markersize=4.5,
                markerfacecolor="white" if arm == "M0" else ARM_COLORS[arm],
                label=ARMS[arm],
            )
        ax.set_xticks(range(len(labels)), [str(k) for k in labels])
        ax.set_xlabel("Class ID", color=MUTED)
        ax.set_ylabel("DSC where present", color=MUTED)
        ax.set_title(f"{NAMES[dataset]}: per class", fontsize=8, color=INK, loc="left")
    for ax in axes:
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, length=2.5, width=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles, names = axes[2].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=4, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=1.0)
    fig.savefig(FIG / "structure_errors.pdf")
    fig.savefig(FIG / "structure_errors.png", dpi=220)
    plt.close(fig)


def main() -> None:
    chosen = sys.argv[1:] or ["targets", "error_geometry", "structures"]
    for name in chosen:
        {"targets": targets, "error_geometry": error_geometry, "structures": structures}[name]()
        print("written", name)


if __name__ == "__main__":
    main()
