"""WP5a: gradient interaction between auxiliary and segmentation objectives (prediction P3).

Input: results/revision_runs/{dataset}_wp5_full_seed1/gradient_probes.json (every 50
global steps, FP32, before the AMP update, on shared encoder parameters; see
protocols/ablation/TRAINING_ADAPTER_PROTOCOL.md). Only completed 150-epoch runs are used.
Descriptive, single-seed diagnostic: no inferential test across datasets.
Outputs: results/tables/gradient_summary.csv, paper/figures/gradient_cosine.{pdf,png}.
Null cosines (zero-norm component, e.g. constant SDF on empty batches) are excluded and
counted.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATASETS = {"isic2018": "ISIC2018", "acdc": "ACDC", "synapse": "Synapse"}
TERMS = {"aux": "all auxiliary", "sdf": "SDF", "boundary": "boundary", "distill": "distillation"}
WINDOWS = {"all": (1, 150), "early": (1, 10), "late": (101, 150)}
COLORS = {"isic2018": "#1b6ca8", "acdc": "#d1495b", "synapse": "#3a7d44"}


def load(dataset: str) -> list[dict] | None:
    run = ROOT / f"results/revision_runs/{dataset}_wp5_full_seed1"
    if not (run / "final_metrics.json").exists():
        return None
    return json.loads((run / "gradient_probes.json").read_text())


def summary() -> list[dict]:
    rows = []
    for dataset in DATASETS:
        probes = load(dataset)
        if probes is None:
            continue
        for term in TERMS:
            for window, (lo, hi) in WINDOWS.items():
                sel = [p for p in probes if lo <= p["epoch"] <= hi]
                cos = np.array(
                    [p[f"{term}_cosine"] for p in sel if p[f"{term}_cosine"] is not None]
                )
                ratio = np.array(
                    [p[f"{term}_norm_ratio"] for p in sel if p[f"{term}_norm_ratio"] is not None]
                )
                rows.append(
                    dict(
                        dataset=dataset,
                        term=term,
                        window=window,
                        probes=len(sel),
                        null_cosines=len(sel) - cos.size,
                        cosine_mean=float(cos.mean()) if cos.size else None,
                        cosine_median=float(np.median(cos)) if cos.size else None,
                        conflict_fraction=float((cos < 0).mean()) if cos.size else None,
                        norm_ratio_median=float(np.median(ratio)) if ratio.size else None,
                    )
                )
    return rows


def smooth(values: np.ndarray, width: int = 25) -> np.ndarray:
    if values.size < width:
        return values
    kernel = np.ones(width) / width
    return np.convolve(values, kernel, mode="valid")


def figure() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    for ax, term in zip(axes, TERMS, strict=True):
        for dataset, name in DATASETS.items():
            probes = load(dataset)
            if probes is None:
                continue
            pts = [
                (p["epoch"], p[f"{term}_cosine"]) for p in probes if p[f"{term}_cosine"] is not None
            ]
            epochs = np.array([e for e, _ in pts], float)
            cos = np.array([c for _, c in pts], float)
            ax.plot(epochs, cos, ".", ms=1.5, alpha=0.15, color=COLORS[dataset])
            sm = smooth(cos)
            offset = (epochs.size - sm.size) // 2
            ax.plot(
                epochs[offset : offset + sm.size], sm, color=COLORS[dataset], lw=1.6, label=name
            )
        ax.axhline(0, color="0.4", lw=0.8)
        ax.set_title(
            f"cos(∇L_seg, ∇L_{term})" if term != "aux" else "cos(∇L_seg, ∇L_aux)", fontsize=9
        )
        ax.set_xlabel("epoch")
    axes[0].set_ylabel("cosine on shared encoder")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = ROOT / "paper/figures/gradient_cosine"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)


def main() -> None:
    rows = summary()
    path = ROOT / "results/tables/gradient_summary.csv"
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    figure()
    for r in rows:
        if r["window"] in ("all", "late"):
            print(r)


if __name__ == "__main__":
    main()
