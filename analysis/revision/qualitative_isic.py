"WP7 draft: deterministic official-test examples from fixed seed1 masks."

import csv
import io
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def main():
    base = ROOT / "results/official_isic/test"
    metrics = {}
    for arm in "UF":
        metrics[arm] = []
        for seed in [1, 2, 3]:
            rows = list(csv.DictReader((base / f"{arm}_seed{seed}/per_case.csv").open()))
            if len(rows) != 1000:
                raise ValueError("incomplete test artifacts")
            metrics[arm].append({r["id"]: r for r in rows})
    ordered = []
    for sid in sorted(metrics["U"][0]):
        values = {arm: [table[sid]["hd95"] for table in metrics[arm]] for arm in "UF"}
        if any("" in values[arm] for arm in "UF"):
            continue
        delta = float(np.mean(np.array(values["U"], float)) - np.mean(np.array(values["F"], float)))
        ordered.append((delta, sid))
    ordered.sort()
    selected = [
        ("largest_negative", ordered[0]),
        ("median", ordered[len(ordered) // 2]),
        ("largest_positive", ordered[-1]),
    ]
    predictions = {}
    for arm in "USF":
        with np.load(base / f"{arm}_seed1/preds_test.npz") as z:
            predictions[arm] = dict(zip(z["ids"], z["predictions"], strict=True))
    directory = ROOT / "paper/figures"
    directory.mkdir(exist_ok=True)
    archives = ROOT / "data/revision/isic2018_official/archives"
    manifest = []
    fig, axes = plt.subplots(3, 6, figsize=(12, 7), layout="constrained")
    with (
        zipfile.ZipFile(archives / "ISIC2018_Task1-2_Test_Input.zip") as images,
        zipfile.ZipFile(archives / "ISIC2018_Task1_Test_GroundTruth.zip") as masks,
    ):
        image_map = {Path(n).stem: n for n in images.namelist() if n.endswith(".jpg")}
        mask_map = {
            Path(n).stem.removesuffix("_segmentation"): n
            for n in masks.namelist()
            if n.endswith(".png")
        }
        for index, (rule, (delta, sid)) in enumerate(selected):
            rgb = np.asarray(
                Image.open(io.BytesIO(images.read(image_map[sid])))
                .convert("RGB")
                .resize((224, 224), Image.Resampling.BILINEAR)
            )
            gt = (
                np.asarray(
                    Image.open(io.BytesIO(masks.read(mask_map[sid])))
                    .convert("L")
                    .resize((224, 224), Image.Resampling.NEAREST)
                )
                > 0
            )
            axes[index, 0].imshow(rgb)
            axes[index, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
            for column, arm in enumerate("USF", 2):
                axes[index, column].imshow(rgb)
                axes[index, column].contour(gt, levels=[0.5], colors=["#009E73"], linewidths=0.7)
                axes[index, column].contour(
                    predictions[arm][sid], levels=[0.5], colors=["#E69F00"], linewidths=0.7
                )
            pred = predictions["F"][sid].astype(bool)
            error = np.ones((224, 224, 3))
            error[pred & ~gt] = [0.835, 0.369, 0]
            error[~pred & gt] = [0, 0.447, 0.698]
            axes[index, 5].imshow(error)
            axes[index, 0].set_ylabel(sid + "\n" + rule, fontsize=8)
            manifest.append(
                dict(
                    id=sid,
                    rule=rule,
                    seed_mean_delta_hd95_u_minus_f=delta,
                    selection="all-three-seed finite paired HD95; lexicographic ties",
                    displayed_seed=1,
                    missing_panels="GSL/strongest comparator pending",
                )
            )
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    for ax, title in zip(
        axes[0], ["Image", "GT", "U-Net", "SDF only", "Full recipe", "Full FP/FN"], strict=True
    ):
        ax.set_title(title, fontsize=10)
    fig.savefig(directory / "official_isic_qualitative_draft.pdf")
    fig.savefig(directory / "official_isic_qualitative_draft.png", dpi=300)
    plt.close(fig)
    (directory / "qualitative_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (directory / "qualitative_caption.md").write_text(
        "Draft for author license/figure review. Official ISIC2018 "
        "test examples selected by deterministic extreme/median seed-"
        "averaged HD95 changes, finite paired cases only. Displayed "
        "masks use fixed seed1. Green contours: GT; orange: "
        "prediction; vermillion: FP; blue: FN. Distances and displays"
        " use the 224 grid. These panels are incomplete for WP7c "
        "until GSL and the strongest required comparator are added. "
        "Source: https://challenge.isic-archive.com/data/ (2018 Task1"
        " CC0). No AI-generated image content.\n"
    )
    print("Saved deterministic draft figure and selection manifest")


if __name__ == "__main__":
    main()
