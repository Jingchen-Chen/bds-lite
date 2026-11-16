"""Independent overlap/lineage checks on all official saved label maps."""

import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def main():
    catalog = {
        r["path"]: r["sha256"]
        for r in csv.DictReader((ROOT / "results/audit/checkpoints.csv").open())
    }
    checks = []
    for split, n in [("Validation", 100), ("Test", 1000)]:
        base = ROOT / f"results/official_isic/{split.lower()}"
        path = (
            ROOT
            / f"data/revision/isic2018_official/archives/ISIC2018_Task1_{split}_GroundTruth.zip"
        )
        targets = {}
        with zipfile.ZipFile(path) as masks:
            for name in masks.namelist():
                if not name.endswith(".png") or name.startswith("__MACOSX/"):
                    continue
                sid = Path(name).stem.removesuffix("_segmentation")
                targets[sid] = (
                    np.asarray(
                        Image.open(io.BytesIO(masks.read(name)))
                        .convert("L")
                        .resize((224, 224), Image.Resampling.NEAREST)
                    )
                    > 0
                )
        for arm in "USF":
            for seed in [1, 2, 3]:
                folder = base / f"{arm}_seed{seed}"
                provenance = json.loads((folder / "provenance.json").read_text())
                if provenance["sha256"] != catalog[provenance["checkpoint"]]:
                    raise ValueError("checkpoint differs from preregistered catalog")
                cases = {r["id"]: r for r in csv.DictReader((folder / "per_case.csv").open())}
                max_error = 0.0
                with np.load(folder / f"preds_{split.lower()}.npz") as saved:
                    ids = saved["ids"]
                    pred = saved["predictions"]
                    if len(ids) != n or set(ids) != set(targets) or set(ids) != set(cases):
                        raise ValueError("saved prediction/label/metric ID mismatch")
                    if (
                        pred.shape != (n, 224, 224)
                        or pred.dtype != np.uint8
                        or not np.isin(pred, [0, 1]).all()
                    ):
                        raise ValueError("invalid saved label maps")
                    for sid, prediction in zip(ids, pred, strict=True):
                        p = prediction.astype(bool)
                        gt = targets[sid]
                        intersection = np.count_nonzero(p & gt)
                        dice = (2 * intersection + 1e-6) / (
                            np.count_nonzero(p) + np.count_nonzero(gt) + 1e-6
                        )
                        max_error = max(max_error, abs(dice - float(cases[sid]["dsc"])))
                if max_error > 1e-12:
                    raise ValueError("independent DSC check differs")
                checks.append(
                    dict(
                        split=split,
                        arm=arm,
                        seed=seed,
                        n=n,
                        max_dsc_error=max_error,
                        checkpoint_catalog_match=True,
                    )
                )
    with (ROOT / "results/validation/official_outputs.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(checks[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(checks)
    print("Independent overlap/lineage checks:", sum(r["n"] for r in checks), "prediction maps")


if __name__ == "__main__":
    main()
