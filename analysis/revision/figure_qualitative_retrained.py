# ruff: noqa: E501
"""Qualitative figure from the retrained set: decoder-only versus BDS-Lite, seeds 1-3.

Selection: the fixed rule of figure_qualitative.py (prereg_revision.md), applied to the central
contrast of the paper. Per dataset, the test sample with the largest positive, median, and largest
negative seed-averaged HD95 difference decoder-only minus BDS-Lite (positive = BDS-Lite better),
from the per-case CSVs; lexicographic ID ties; samples with undefined HD95 in any seed of either arm
are excluded. The shown seed is the one whose per-seed difference is closest to the seed mean.

Licensing (docs/LICENSING_AUDIT.md): ISIC2018 Task 1 reference masks (CC-0) and our own
predictions on a white background only; no photograph is read or drawn. ACDC and Synapse labels feed
statistics only and are never drawn.

Writes paper/figures/contours_isic2018_retrained.{pdf,png} and
paper/figures/qualitative_manifest_v3.json (cases, per-seed differences, panel statistics).
Usage: PYTHONPATH=src:. .venv/bin/python analysis/revision/figure_qualitative_retrained.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from analysis.revision import figure_qualitative as fq  # noqa: E402
from analysis.revision.recover_predictions import Cohort  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIRST, SECOND = "M2", "M6"  # decoder-only minus BDS-Lite
SHOWN = ("M0", "M2", "M6", "GSL")
TITLES = {"M0": "U-Net", "M2": "Decoder-only", "M6": "BDS-Lite", "GSL": "U-Net+GSL"}


def select(dataset: str) -> list[tuple[str, str, float, int, list[float]]]:
    first, second = fq.per_seed_hd95(FIRST, dataset), fq.per_seed_hd95(SECOND, dataset)
    rows = []
    for sid in first:
        a, b = np.array(first[sid]), np.array(second.get(sid, [np.nan] * 3))
        if np.isfinite(a).all() and np.isfinite(b).all():
            per_seed = a - b
            delta = float(per_seed.mean())
            seed = int(np.argmin(np.abs(per_seed - delta))) + 1
            rows.append((delta, sid, seed, [float(x) for x in per_seed]))
    by_value = sorted(rows, key=lambda x: (x[0], x[1]))
    lo, med = by_value[0], by_value[(len(by_value) - 1) // 2]
    hi = sorted(rows, key=lambda x: (-x[0], x[1]))[0]
    return [
        ("largest BDS-Lite gain", hi[1], hi[0], hi[2], hi[3]),
        ("median", med[1], med[0], med[2], med[3]),
        ("largest BDS-Lite loss", lo[1], lo[0], lo[2], lo[3]),
    ]


def render(cases: list, preds: dict, masks: dict) -> None:
    cols = 1 + len(SHOWN)
    fig, axes = plt.subplots(len(cases), cols, figsize=(7.2, 1.35 * len(cases) + 0.45))
    for i, (label, sid, delta, seed, _) in enumerate(cases):
        mask = masks[sid]
        row = axes[i]
        reference = np.ones(mask.shape + (3,))
        reference[mask > 0] = fq.TP
        row[0].imshow(reference)
        row[0].contour(mask > 0, levels=[0.5], colors=[fq.INK], linewidths=0.6)
        row[0].set_ylabel(f"{label}\n$\\Delta$HD95 {delta:+.1f} px", fontsize=6, color=fq.MUTED)
        for j, method in enumerate(SHOWN, start=1):
            pred = preds[(method, seed)][sid]
            canvas = np.ones(mask.shape + (3,))
            canvas[(pred > 0) & (mask > 0)] = fq.TP
            canvas[(pred > 0) & (mask == 0)] = fq.FP
            canvas[(pred == 0) & (mask > 0)] = fq.FN
            row[j].imshow(canvas)
            row[j].contour(mask > 0, levels=[0.5], colors=[fq.INK], linewidths=0.6)
            hd95 = fq.panel_stats(method, "isic2018", seed, sid, pred, mask)["hd95"]
            row[j].text(
                0.97,
                0.03,
                f"{hd95:.0f}",
                transform=row[j].transAxes,
                ha="right",
                va="bottom",
                fontsize=6,
                color=fq.INK,
            )
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(fq.GRID)
                spine.set_linewidth(0.5)
        if i == 0:
            row[0].set_title("Reference", fontsize=6.5, color=fq.INK)
            for j, method in enumerate(SHOWN, start=1):
                row[j].set_title(TITLES[method], fontsize=6.5, color=fq.INK)
    legend = [
        Patch(facecolor=fq.TP, edgecolor=fq.GRID, label="Agreement"),
        Patch(facecolor=fq.FP, label="False positive"),
        Patch(facecolor=fq.FN, label="False negative"),
        Line2D([], [], color=fq.INK, linewidth=0.8, label="Reference contour"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.2, w_pad=0.2)
    fig.savefig(
        ROOT / "paper/figures/contours_isic2018_retrained.pdf", metadata={"CreationDate": None}
    )
    fig.savefig(ROOT / "paper/figures/contours_isic2018_retrained.png", dpi=420)
    plt.close(fig)


def main() -> None:
    manifest = {
        "rule": __doc__.split("\n\n")[1].replace("\n", " "),
        "first": FIRST,
        "second": SECOND,
    }
    for dataset in ("isic2018", "acdc", "synapse"):
        methods = [m for m in SHOWN if fq.complete(m, dataset)]
        if methods != list(SHOWN):
            raise ValueError(f"incomplete predictions on {dataset}: {methods}")
        preds = {(m, k): fq.load_seed(m, dataset, k) for m in methods for k in (1, 2, 3)}
        cases = select(dataset)
        if dataset == "isic2018":
            cohort = Cohort(dict(dataset="isic2018", split="official_test"))
            masks = {sid: cohort.get(sid)[1] for _, sid, _, _, _ in cases}
            render(cases, preds, masks)
        else:
            masks = {
                sid: np.load(ROOT / f"data/processed/{dataset}/test/{sid}.npz")["mask"]
                for _, sid, _, _, _ in cases
            }
        manifest[dataset] = dict(
            cases=[
                dict(
                    role=r,
                    id=i,
                    delta_hd95_decoder_minus_bds=d,
                    shown_seed=k,
                    per_seed_delta=ps,
                    panels={
                        m: fq.panel_stats(m, dataset, k, i, preds[(m, k)][i], masks[i])
                        for m in methods
                    },
                )
                for r, i, d, k, ps in cases
            ],
            methods_shown=methods,
        )
    (ROOT / "paper/figures/qualitative_manifest_v3.json").write_text(
        json.dumps(manifest, indent=1) + "\n"
    )
    print("written contours_isic2018_retrained and qualitative_manifest_v3.json")


if __name__ == "__main__":
    main()
