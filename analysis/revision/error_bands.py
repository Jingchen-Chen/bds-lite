# ruff: noqa: E501
"""WP1f error decomposition for every complete matched method (seeds 1-3, test data).

Per test sample and foreground class: false-positive and false-negative pixels within
2 px of the reference class boundary ("near", Euclidean distance to the reference
contour <= 2) versus farther ("far"); predicted 8-connected components without any
overlap with the reference class ("spurious components"). False alarms on classes absent
from the reference are counted as far FP and spurious components. Aggregates: per
method/dataset, pixel totals per image averaged over seeds; far-FP share; spurious
components per image.
Output: results/tables/families/error_bands.csv
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from analysis.revision.error_decomposition import load_targets
from analysis.revision.family_statistics import DATASETS, METHODS, SEEDS, per_case_path
from analysis.revision.figure_qualitative import prediction_file

ROOT = Path(__file__).resolve().parents[2]
EIGHT = np.ones((3, 3), bool)


def classes(dataset: str) -> range:
    return range(1, {"isic2018": 2, "acdc": 4, "synapse": 9}[dataset])


def decompose(pred: np.ndarray, target: np.ndarray, label: int) -> dict:
    p, t = pred == label, target == label
    out = dict(fp_near=0, fp_far=0, fn_near=0, fn_far=0, spurious=0)
    if not p.any() and not t.any():
        return out
    if t.any():
        contour = t ^ ndi.binary_erosion(t)
        dist = ndi.distance_transform_edt(~contour)
        near = dist <= 2
    else:
        near = np.zeros_like(p)
    fp, fn = p & ~t, t & ~p
    out.update(
        fp_near=int((fp & near).sum()),
        fp_far=int((fp & ~near).sum()),
        fn_near=int((fn & near).sum()),
        fn_far=int((fn & ~near).sum()),
    )
    comps, n = ndi.label(p, structure=EIGHT)
    if n:
        hit = np.unique(comps[t & p])
        out["spurious"] = int(n - np.count_nonzero(hit))
    return out


def main() -> None:
    rows = []
    for dataset in DATASETS:
        split = "official_test" if dataset == "isic2018" else "test"
        targets = None
        for method in METHODS:
            files = [prediction_file(method, dataset, s) for s in SEEDS]
            if any(f is None for f in files) or METHODS[method][1] == "replication":
                continue
            per_seed = defaultdict(list)
            for f in files:
                d = np.load(f)
                ids = [str(i) for i in d["ids"]]
                if targets is None:
                    targets = load_targets(dataset, split, ids)
                totals = defaultdict(int)
                for sid, pred in zip(ids, d["predictions"], strict=True):
                    for label in classes(dataset):
                        for key, value in decompose(pred, targets[sid], label).items():
                            totals[key] += value
                n = len(ids)
                for key, value in totals.items():
                    per_seed[key].append(value / n)
                fp = totals["fp_near"] + totals["fp_far"]
                per_seed["far_fp_share"].append(totals["fp_far"] / fp if fp else float("nan"))
            row = dict(dataset=dataset, method=method)
            for key, values in per_seed.items():
                row[f"{key}_per_image"] = float(np.nanmean(values))
            rows.append(row)
            print(json.dumps(row), flush=True)
    out = ROOT / "results/tables/families/error_bands.csv"
    with out.open("w") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    _ = per_case_path  # registry shared with family_statistics (documented dependency)


if __name__ == "__main__":
    main()
