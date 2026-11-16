# ruff: noqa: E501
"""WP7c qualitative comparison with error maps, deterministic case selection.

Selection (prereg_revision.md): per dataset, the test sample with the largest positive,
median, and largest negative seed-averaged HD95 difference U-Net minus BDS-Lite
(positive = BDS-Lite better), computed from the per-case CSVs by
family_statistics.sample_values; lexicographic ID ties; samples with undefined HD95 for
either method are excluded. Shown prediction per method = the
representative seed (per-seed delta closest to the seed mean); cases need defined
HD95 in every seed for both methods (empty-prediction cases are excluded, prereg).

Licensing (docs/LICENSING_AUDIT.md): the figure uses ISIC2018 Task 1 reference masks (CC-0) and our
own predictions on a white background. No dataset photograph is read or drawn. ACDC and Synapse
reference labels have no readable licence, so they feed statistics only and are never drawn.
Methods without complete predictions are omitted and listed in the manifest.
The manifest also records, per case and shown method, the label-macro HD95 of the shown seed
(per-case CSV) and the error decomposition of error_bands.decompose summed over foreground
labels (near/far FP/FN pixels, spurious components), so the text can cite what each panel shows.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import csv  # noqa: E402
from collections import defaultdict  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from analysis.revision.family_statistics import _value, per_case_path
from analysis.revision.recover_predictions import Cohort  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ORDER = ["S0", "S6", "M2", "GSL", "M1", "C1", "C3", "C4"]
SHOWN = ("S0", "S6", "M2", "GSL", "C1", "C3", "C4")
TITLES = {
    "S0": "U-Net (original)",
    "S6": "BDS-Lite (original)",
    "M2": "Decoder",
    "GSL": "U-Net+GSL",
    "C1": "Boundary Loss",
    "C3": "Attention U-Net",
    "C4": "LB-UNet",
}
TP, FP, FN = (0.91, 0.91, 0.89), (0.92, 0.41, 0.20), (0.16, 0.47, 0.84)
INK, MUTED, GRID = "#2b2b2b", "#52514e", "#e9e8e4"


def prediction_file(method: str, dataset: str, seed: int) -> Path | None:
    """Prediction archive stored next to the per-case CSV used by family_statistics."""
    csv_path = per_case_path(method, dataset, seed)
    if csv_path is None:
        return None
    for name in ("preds_test.npz", "predictions.npz"):
        if (csv_path.parent / name).exists():
            return csv_path.parent / name
    return None


def load_seed(method: str, dataset: str, seed: int) -> dict[str, np.ndarray] | None:
    path = prediction_file(method, dataset, seed)
    if path is None:
        return None
    d = np.load(path)
    return dict(zip(d["ids"], d["predictions"], strict=True))


def complete(method: str, dataset: str) -> bool:
    return all(prediction_file(method, dataset, s) is not None for s in (1, 2, 3))


def per_seed_hd95(method: str, dataset: str) -> dict[str, list[float]]:
    """sample -> [label-macro HD95 for seeds 1..3] (NaN if undefined)."""
    out: dict[str, list[float]] = defaultdict(list)
    for seed in (1, 2, 3):
        by_sample: dict[str, list[float]] = defaultdict(list)
        for row in csv.DictReader(per_case_path(method, dataset, seed).open()):
            by_sample[row["id"]].append(_value(row, "hd95"))
        for sid, values in by_sample.items():
            arr = np.array(values, float)
            out[sid].append(float(np.nanmean(arr)) if np.isfinite(arr).any() else math.nan)
    return out


def select(dataset: str) -> list[tuple[str, str, float, int, list[float]]]:
    """Cases with defined HD95 in every seed for both methods; returns
    (role, id, delta, shown seed, per-seed deltas)."""
    unet, bds = per_seed_hd95("S0", dataset), per_seed_hd95("S6", dataset)
    rows = []
    for sid in unet:
        u, b = np.array(unet[sid]), np.array(bds[sid])
        if np.isfinite(u).all() and np.isfinite(b).all():
            per_seed = u - b
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


def panel_stats(method: str, dataset: str, seed: int, sid: str, pred, mask) -> dict:
    from analysis.revision.error_bands import decompose  # local: error_bands imports this module

    values = [
        _value(r, "hd95")
        for r in csv.DictReader(per_case_path(method, dataset, seed).open())
        if r["id"] == sid
    ]
    arr = np.array(values, float)
    out = dict(hd95=float(np.nanmean(arr)) if np.isfinite(arr).any() else math.nan)
    labels = (set(np.unique(mask).tolist()) | set(np.unique(pred).tolist())) - {0}
    for label in sorted(labels):
        for k, v in decompose(pred, mask, int(label)).items():
            out[k] = out.get(k, 0) + v
    for k in ("fp_near", "fp_far", "fn_near", "fn_far", "spurious"):
        out.setdefault(k, 0)
    return out


def render_isic(cases: list, preds: dict, masks: dict, methods: list[str]) -> None:
    """Contour and error maps on white. No photograph is read or drawn (LICENSING_AUDIT.md)."""
    shown = [m for m in methods if m in SHOWN]
    cols = 1 + len(shown)
    fig, axes = plt.subplots(len(cases), cols, figsize=(7.2, 0.93 * len(cases) + 0.55))
    for i, (label, sid, delta, seed, _) in enumerate(cases):
        mask = masks[sid]
        row = axes[i]
        reference = np.ones(mask.shape + (3,))
        reference[mask > 0] = TP
        row[0].imshow(reference)
        row[0].contour(mask > 0, levels=[0.5], colors=[INK], linewidths=0.6)
        row[0].set_ylabel(f"{label}\n$\\Delta$HD95 {delta:+.1f} px", fontsize=6, color=MUTED)
        for j, method in enumerate(shown, start=1):
            pred = preds[(method, seed)][sid]
            canvas = np.ones(mask.shape + (3,))
            canvas[(pred > 0) & (mask > 0)] = TP
            canvas[(pred > 0) & (mask == 0)] = FP
            canvas[(pred == 0) & (mask > 0)] = FN
            row[j].imshow(canvas)
            row[j].contour(mask > 0, levels=[0.5], colors=[INK], linewidths=0.6)
            hd95 = panel_stats(method, "isic2018", seed, sid, pred, mask)["hd95"]
            row[j].text(
                0.97,
                0.03,
                f"{hd95:.0f}",
                transform=row[j].transAxes,
                ha="right",
                va="bottom",
                fontsize=6,
                color=INK,
            )
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(GRID)
                spine.set_linewidth(0.5)
        if i == 0:
            row[0].set_title("Reference", fontsize=6.5, color=INK)
            for j, method in enumerate(shown, start=1):
                row[j].set_title(TITLES[method], fontsize=6.5, color=INK)
    legend = [
        Patch(facecolor=TP, edgecolor=GRID, label="Agreement"),
        Patch(facecolor=FP, label="False positive"),
        Patch(facecolor=FN, label="False negative"),
        Line2D([], [], color=INK, linewidth=0.8, label="Reference contour"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=0.2, w_pad=0.2)
    fig.savefig(ROOT / "paper/figures/contours_isic2018.pdf")
    fig.savefig(ROOT / "paper/figures/contours_isic2018.png", dpi=250)
    plt.close(fig)


def main() -> None:
    manifest = {}
    for dataset in ("isic2018", "acdc", "synapse"):
        methods = [m for m in ORDER if complete(m, dataset)]
        preds = {(m, k): load_seed(m, dataset, k) for m in methods for k in (1, 2, 3)}
        cases = select(dataset)
        if dataset == "isic2018":
            cohort = Cohort(dict(dataset="isic2018", split="official_test"))
            masks = {sid: cohort.get(sid)[1] for _, sid, _, _, _ in cases}
            render_isic(cases, preds, masks, methods)
        else:  # statistics only: ACDC and Synapse labels are never drawn
            masks = {
                sid: np.load(ROOT / f"data/processed/{dataset}/test/{sid}.npz")["mask"]
                for _, sid, _, _, _ in cases
            }
        manifest[dataset] = dict(
            cases=[
                dict(
                    role=r,
                    id=i,
                    delta_hd95_unet_minus_bds=d,
                    shown_seed=k,
                    per_seed_delta=ps,
                    panels={
                        m: panel_stats(m, dataset, k, i, preds[(m, k)][i], masks[i])
                        for m in methods
                    },
                )
                for r, i, d, k, ps in cases
            ],
            methods_shown=methods,
            methods_missing=[m for m in ORDER if m not in methods],
        )
    (ROOT / "paper/figures/qualitative_manifest_v2.json").write_text(
        json.dumps(manifest, indent=1) + "\n"
    )
    print(json.dumps({k: v["methods_shown"] for k, v in manifest.items()}))


if __name__ == "__main__":
    main()
