"""Verify all five saved metrics without rerunning training or inference."""

import csv
import json
from pathlib import Path

import numpy as np

from bds_lite.data.boundary import mask_to_boundary
from bds_lite.evaluation.metrics import assd, boundary_f1, hd95

ROOT = Path(__file__).resolve().parents[2]


def main():
    source = ROOT / "recovered/followup"
    ids = json.loads((source / "splits/isic2018/val.json").read_text())["sample_ids"]
    masks = {}
    boundaries = {}
    for sid in ids:
        with np.load(source / f"data/followup/isic2018/val/{sid}.npz") as z:
            masks[sid] = z["mask"]
            boundaries[sid] = mask_to_boundary(z["mask"])
    rows = []
    for arm in "USBF":
        for seed in [1, 2, 3]:
            folder = source / f"results/followup_20261018_v2/isic2018_{arm}_seed{seed}"
            max_error = 0.0
            missing = 0
            checked = 0
            for row in csv.DictReader((folder / "per_case.csv").open()):
                sid = row["id"]
                pred = np.load(folder / f"predictions/{sid}.npy")
                values = dict(
                    boundary_f1=boundary_f1(mask_to_boundary(pred), boundaries[sid]),
                    hd95=hd95(pred, masks[sid]),
                    assd=assd(pred, masks[sid]),
                )
                for key, value in values.items():
                    if not np.isfinite(value):
                        if row[key] not in ["", "nan"]:
                            raise ValueError(f"missingness differs {folder} {sid} {key}")
                        missing += 1
                    else:
                        error = abs(value - float(row[key]))
                        max_error = max(error, max_error)
                        if error > 1e-10:
                            raise ValueError(f"metric mismatch: {folder} {sid} {key} {error}")
                    checked += 1
            rows.append(
                dict(
                    run=folder.name,
                    finite_max_error=max_error,
                    metric_cells=checked,
                    missing_distance_cells=missing,
                )
            )
            print(folder.name, max_error, flush=True)
    with (ROOT / "revision/validation/followup_boundary_distance_recomputed.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
