# ruff: noqa: E501
"""WP7b dataset example figure: image | reference | boundary target | SDF target.

One training sample per dataset chosen by a fixed rule (no visual selection):
the training sample whose number of present foreground classes is maximal and, among
those, whose total foreground area is the median (ties: lexicographic ID).
ACDC/Synapse intensity images are shown only in the `_with_images` variant, pending
licence approval (revision/licensing.md); the default variant shows label maps only.
Targets are the stored training targets (union morphological-gradient boundary,
per-class SDF); for multi-class SDF the minimum over classes is displayed.
Outputs: revision/figures/dataset_examples{,_with_images}.{pdf,png} and a manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATASETS = {"isic2018": "ISIC2018", "acdc": "ACDC", "synapse": "Synapse"}
# colour-blind-safe qualitative palette (Okabe-Ito), background transparent/black
PALETTE = [
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#999999",
]


def choose(dataset: str) -> Path:
    manifest = json.loads((ROOT / f"recovered/followup/splits/{dataset}/train.json").read_text())
    stats = []
    for sid in manifest["sample_ids"]:
        mask = np.load(ROOT / f"data/processed/{dataset}/train/{sid}.npz")["mask"]
        stats.append((len(set(np.unique(mask)) - {0}), int((mask > 0).sum()), sid))
    top = max(s[0] for s in stats)
    candidates = sorted((area, sid) for n, area, sid in stats if n == top)
    return ROOT / f"data/processed/{dataset}/train/{candidates[(len(candidates) - 1) // 2][1]}.npz"


def draw(with_images: bool) -> dict:
    fig, axes = plt.subplots(3, 4, figsize=(10, 7.6))
    manifest = {}
    for row, (dataset, name) in enumerate(DATASETS.items()):
        path = choose(dataset)
        data = np.load(path)
        image, mask = data["image"], data["mask"]
        boundary, sdf = data["boundary"][0], data["sdf"]
        manifest[dataset] = str(path.relative_to(ROOT))
        show_image = with_images or dataset == "isic2018"
        ax = axes[row, 0]
        if show_image:
            img = image.transpose(1, 2, 0) if image.shape[0] == 3 else image[0]
            img = (img - img.min()) / max(float(img.max() - img.min()), 1e-6)
            ax.imshow(img, cmap=None if image.shape[0] == 3 else "gray")
        else:
            ax.text(0.5, 0.5, "intensity image\nnot shown", ha="center", va="center", fontsize=8)
            ax.set_facecolor("#f2f2f2")
        cmap = ListedColormap(PALETTE[: int(mask.max()) + 1])
        axes[row, 1].imshow(
            mask, cmap=cmap, interpolation="nearest", vmin=0, vmax=max(int(mask.max()), 1)
        )
        axes[row, 2].imshow(boundary, cmap="gray_r", interpolation="nearest")
        field = sdf.min(axis=0)
        limit = float(np.abs(field).max()) or 1.0
        im = axes[row, 3].imshow(field, cmap="RdBu_r", vmin=-limit, vmax=limit)
        fig.colorbar(im, ax=axes[row, 3], fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)
        axes[row, 0].set_ylabel(name, fontsize=10)
    for col, title in enumerate(
        ["Image", "Reference mask", "Boundary target $y_b$", "SDF target $\\phi$ (px)"]
    ):
        axes[0, col].set_title(title, fontsize=10)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    stem = ROOT / f"revision/figures/dataset_examples{'_with_images' if with_images else ''}"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return manifest


def main() -> None:
    manifest = draw(False)
    draw(True)
    rule = "max present classes, then median foreground area, lexicographic ties (training split)"
    (ROOT / "revision/figures/dataset_examples_manifest.json").write_text(
        json.dumps(dict(rule=rule, samples=manifest), indent=1) + "\n"
    )
    print(manifest)


if __name__ == "__main__":
    raise SystemExit(
        "Retired: dataset pixels may not be published (revision/LICENSING_AUDIT.md). Use figure_synthetic_targets.py."
    )
    main()
