"""Contour figure for U, D and D+KD-band on the official ISIC2018 test set (main text).

Selection (independent of every model): the 1,000 official test images are sorted by the area
of their reference mask on the 224 x 224 grid, ties broken by image ID, and the images at the
10th, 30th, 50th, 70th and 90th percentile ranks (index round(q * 999)) are shown. One run per
arm, the lowest seed (seed 1), is drawn for every image. No score of any model enters the
selection.

Licensing (revision/LICENSING_AUDIT.md): only ISIC2018 Task 1 test reference masks (CC-0) and
this study's own predicted masks are read and drawn, on a white background. The photograph
archive is never opened. ACDC and Synapse images and annotations are not drawn.

Writes revision/figures/contours_isic2018_kd.{pdf,png} and
revision/figures/qualitative_manifest_kd.json (rule, image IDs, areas, per-panel scores read
from the per-case result files, source paths and hashes).
Usage: PYTHONPATH=src:. .venv/bin/python analysis/revision/figure_qualitative_kd.py
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MASKS = ROOT / "data/revision/isic2018_official/archives/ISIC2018_Task1_Test_GroundTruth.zip"
SEED = 1
QUANTILES = (0.10, 0.30, 0.50, 0.70, 0.90)
# arm label -> run directory holding preds_test.npz and per_case_test.csv (seed 1)
ARMS = {
    "U": ROOT / "revision/official_isic/test/U_seed1",
    "D": ROOT / "results/revision_runs/isic2018_m2_aux_boundary_only_seed1",
    "D+KD-band": ROOT / "results/revision_runs/isic2018_kd_band_seed1",
}
INK, GRID, MUTED = "#222222", "#bbbbbb", "#555555"
TP, FP, FN = (0.86, 0.86, 0.86), (0.90, 0.62, 0.0), (0.0, 0.45, 0.70)  # grey, orange, blue


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_masks() -> dict[str, np.ndarray]:
    out = {}
    with zipfile.ZipFile(MASKS) as archive:
        for name in archive.namelist():
            if name.endswith(".png") and not name.startswith("__MACOSX/"):
                sid = Path(name).stem.removesuffix("_segmentation")
                gt = Image.open(io.BytesIO(archive.read(name))).convert("L")
                resized = gt.resize((224, 224), Image.Resampling.NEAREST)
                out[sid] = (np.asarray(resized) > 0).astype(np.uint8)
    if len(out) != 1000:
        raise ValueError(f"expected 1000 reference masks, found {len(out)}")
    return out


def per_case(run: Path) -> dict[str, dict]:
    path = run / "per_case_test.csv"
    if not path.exists():
        candidates = sorted(run.glob("per_case*.csv"))
        if len(candidates) != 1:
            raise FileNotFoundError(f"no unique per-case file in {run}")
        path = candidates[0]
    with path.open() as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def main() -> None:
    masks = reference_masks()
    ranked = sorted(masks, key=lambda sid: (int(masks[sid].sum()), sid))
    chosen = [ranked[round(q * (len(ranked) - 1))] for q in QUANTILES]
    preds, scores = {}, {}
    for arm, run in ARMS.items():
        with np.load(run / "preds_test.npz") as z:
            table = dict(zip(z["ids"], z["predictions"], strict=True))
        if set(table) != set(masks):
            raise ValueError(f"{arm}: prediction IDs differ from the official test set")
        preds[arm] = table
        scores[arm] = per_case(run)

    cols = 1 + len(ARMS)
    fig, axes = plt.subplots(len(chosen), cols, figsize=(5.74, 1.38 * len(chosen) + 0.4))
    for i, (q, sid) in enumerate(zip(QUANTILES, chosen, strict=True)):
        mask = masks[sid]
        canvas = np.ones(mask.shape + (3,))
        canvas[mask > 0] = TP
        axes[i, 0].imshow(canvas)
        axes[i, 0].contour(mask > 0, levels=[0.5], colors=[INK], linewidths=0.6)
        axes[i, 0].set_ylabel(f"{sid}\narea rank {round(100 * q)}%", fontsize=5.5, color=MUTED)
        for j, arm in enumerate(ARMS, start=1):
            pred = preds[arm][sid]
            canvas = np.ones(mask.shape + (3,))
            canvas[(pred > 0) & (mask > 0)] = TP
            canvas[(pred > 0) & (mask == 0)] = FP
            canvas[(pred == 0) & (mask > 0)] = FN
            axes[i, j].imshow(canvas)
            axes[i, j].contour(mask > 0, levels=[0.5], colors=[INK], linewidths=0.6)
            if pred.any():
                axes[i, j].contour(
                    pred > 0, levels=[0.5], colors=["#d55e00"], linewidths=0.5, linestyles="--"
                )
            bf1 = float(scores[arm][sid]["boundary_f1"])
            axes[i, j].text(
                0.97,
                0.03,
                f"{bf1:.2f}",
                transform=axes[i, j].transAxes,
                ha="right",
                va="bottom",
                fontsize=6,
                color=INK,
            )
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(GRID)
                spine.set_linewidth(0.5)
    axes[0, 0].set_title("Reference", fontsize=6.5, color=INK)
    for j, arm in enumerate(ARMS, start=1):
        axes[0, j].set_title(arm, fontsize=6.5, color=INK)
    legend = [
        Line2D([], [], color=INK, linewidth=0.8, label="Reference contour"),
        Line2D([], [], color="#d55e00", linewidth=0.8, linestyle="--", label="Predicted contour"),
        Patch(facecolor=TP, edgecolor=GRID, label="Agreement"),
        Patch(facecolor=FP, label="False positive"),
        Patch(facecolor=FN, label="False negative"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False, fontsize=6)
    fig.tight_layout(rect=(0, 0.045, 1, 1), h_pad=0.2, w_pad=0.2)
    out = ROOT / "revision/figures"
    fig.savefig(out / "contours_isic2018_kd.pdf", metadata={"CreationDate": None})
    fig.savefig(out / "contours_isic2018_kd.png", dpi=420)
    plt.close(fig)

    manifest = {
        "rule": " ".join(__doc__.split("\n\n")[1].split()),
        "licensing": "revision/LICENSING_AUDIT.md: ISIC2018 Task 1 test reference masks (CC-0) "
        "and this study's predictions only; no photograph is read",
        "reference_archive": str(MASKS.relative_to(ROOT)),
        "reference_archive_sha256": sha(MASKS),
        "seed": SEED,
        "runs": {
            arm: {
                "path": str(run.relative_to(ROOT)),
                "preds_test_sha256": sha(run / "preds_test.npz"),
            }
            for arm, run in ARMS.items()
        },
        "cases": [
            {
                "id": sid,
                "area_quantile": q,
                "area_rank_index": round(q * (len(ranked) - 1)),
                "reference_area_px": int(masks[sid].sum()),
                "boundary_f1": {arm: float(scores[arm][sid]["boundary_f1"]) for arm in ARMS},
                "dsc": {arm: float(scores[arm][sid]["dsc"]) for arm in ARMS},
            }
            for q, sid in zip(QUANTILES, chosen, strict=True)
        ],
    }
    (out / "qualitative_manifest_kd.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print("wrote", out / "contours_isic2018_kd.pdf", [c["id"] for c in manifest["cases"]])


if __name__ == "__main__":
    main()
