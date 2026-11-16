"""ISIC2018 official-test Jaccard and thresholded Jaccard at original resolution, all methods.

Same procedure as analysis/revision/evaluate_official_isic.py: the 224x224 argmax map is
resized to the original image size with PIL nearest neighbour and compared with the
original binary mask (>0); thresholded Jaccard sets values below 0.65 to zero (ISIC 2018
challenge rule). Methods/seeds are located through family_statistics.per_case_path.
Output: results/tables/families/native_jaccard.csv (per method: seed means).
"""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from analysis.revision.family_statistics import METHODS, SEEDS
from analysis.revision.figure_qualitative import prediction_file

ROOT = Path(__file__).resolve().parents[2]
GT = ROOT / "data/revision/isic2018_official/archives/ISIC2018_Task1_Test_GroundTruth.zip"


def ground_truth() -> dict[str, np.ndarray]:
    out = {}
    with zipfile.ZipFile(GT) as archive:
        for name in archive.namelist():
            if name.endswith(".png") and not name.startswith("__MACOSX/"):
                sid = Path(name).stem.removesuffix("_segmentation")
                out[sid] = np.asarray(Image.open(io.BytesIO(archive.read(name))).convert("L")) > 0
    return out


def main() -> None:
    truth = ground_truth()
    rows = []
    for method in METHODS:
        files = [prediction_file(method, "isic2018", s) for s in SEEDS]
        if any(f is None for f in files):
            continue
        per_seed_j, per_seed_tj = [], []
        for f in files:
            d = np.load(f)
            js = []
            for sid, pred in zip(d["ids"], d["predictions"], strict=True):
                original = truth[str(sid)]
                native = (
                    np.asarray(
                        Image.fromarray(pred.astype(np.uint8)).resize(
                            (original.shape[1], original.shape[0]), Image.Resampling.NEAREST
                        )
                    )
                    > 0
                )
                union = np.logical_or(native, original).sum()
                js.append(float(np.logical_and(native, original).sum() / union) if union else 1.0)
            js = np.array(js)
            if js.size != 1000:
                raise ValueError(f"{method}: expected 1000 official test images, got {js.size}")
            per_seed_j.append(js.mean())
            per_seed_tj.append(np.where(js >= 0.65, js, 0.0).mean())
        rows.append(
            dict(
                method=method,
                jaccard=float(np.mean(per_seed_j)),
                thresholded_jaccard=float(np.mean(per_seed_tj)),
                thresholded_jaccard_seeds=";".join(f"{x:.4f}" for x in per_seed_tj),
            )
        )
        print(rows[-1], flush=True)
    out = ROOT / "results/tables/families/native_jaccard.csv"
    with out.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
