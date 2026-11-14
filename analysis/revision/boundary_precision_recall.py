# ruff: noqa: E501
"""Exploratory: what changes inside Boundary F1 on ISIC2018, using saved test predictions only.

Boundary precision (predicted contour within 2 px of the reference) and recall (reference contour
within 2 px of the prediction), with the training evaluator's definition, plus the number of
connected components not overlapping the reference. Averaged per image, then per run.
Output: revision/tables/families/boundary_precision_recall.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from analysis.revision.recover_predictions import Cohort  # noqa: E402
from bds_lite.data.boundary import mask_to_boundary  # noqa: E402

RUNS = "results/revision_runs/isic2018_{}_seed{}/preds_test.npz"
OFFICIAL = "revision/official_isic/test/{}_seed{}/preds_test.npz"
TEN = range(1, 11)
ARMS = {
    "M0": lambda s: OFFICIAL.format("U", s) if s <= 3 else RUNS.format("m0_unet", s),
    "M2": lambda s: RUNS.format("m2_aux_boundary_only", s),
    "M6": lambda s: OFFICIAL.format("F", s) if s <= 3 else RUNS.format("m6_full", s),
    "GSL": lambda s: RUNS.format("gsl", s),
}
THREE = {"M7": "m7_decoder_surface"}
OUT = ROOT / "revision/tables/families/boundary_precision_recall.csv"


def measures(pred: np.ndarray, target: np.ndarray) -> tuple[float, float, int]:
    pb = mask_to_boundary(pred.astype(np.uint8)).astype(bool)
    tb = mask_to_boundary(target.astype(np.uint8)).astype(bool)
    labels, count = ndi.label(pred)
    spurious = 0
    if count:
        hit = set(np.unique(labels[target])) - {0}
        spurious = count - len(hit)
    if not pb.any() or not tb.any():
        return float("nan"), float("nan"), spurious
    tz = ndi.binary_dilation(tb, iterations=2)
    pz = ndi.binary_dilation(pb, iterations=2)
    return float((pb & tz).sum() / pb.sum()), float((tb & pz).sum() / tb.sum()), spurious


def main() -> None:
    cohort = Cohort(dict(dataset="isic2018", split="official_test"))
    target = {i: np.asarray(cohort.get(i)[1]) > 0 for i in cohort.ids}
    plan = [(arm, s, fn(s)) for arm, fn in ARMS.items() for s in TEN]
    plan += [(arm, s, RUNS.format(name, s)) for arm, name in THREE.items() for s in (1, 2, 3)]
    rows = []
    for arm, seed, path in plan:
        z = np.load(ROOT / path)
        pred = dict(zip([str(x) for x in z["ids"]], z["predictions"], strict=True))
        v = np.array([measures(pred[i] > 0, target[i]) for i in target], float)
        rows.append(
            dict(
                arm=arm,
                seed=seed,
                precision=float(np.nanmean(v[:, 0])),
                recall=float(np.nanmean(v[:, 1])),
                spurious=float(v[:, 2].mean()),
            )
        )
        print(rows[-1], flush=True)
    with OUT.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
