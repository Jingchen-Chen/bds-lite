"""Read-only full frozen-training target audit; no data regeneration on disk."""

import csv
import json
from pathlib import Path

import numpy as np

from bds_lite.data.boundary import mask_to_boundary, mask_to_sdf

ROOT = Path(__file__).resolve().parents[2]


def main():
    rows = []
    for dataset, classes in [("isic2018", 2), ("acdc", 4), ("synapse", 9)]:
        manifest = json.loads(
            (ROOT / f"recovered/followup/splits/{dataset}/train.json").read_text()
        )
        for sid in manifest["sample_ids"]:
            with np.load(ROOT / f"data/processed/{dataset}/train/{sid}.npz") as z:
                boundary = mask_to_boundary(z["mask"])[None]
                sdf = mask_to_sdf(z["mask"], num_classes=classes)
                row = dict(
                    dataset=dataset,
                    id=sid,
                    boundary_exact=np.array_equal(boundary, z["boundary"]),
                    sdf_exact=np.array_equal(sdf, z["sdf"]),
                )
                rows.append(row)
        print(dataset, len(manifest["sample_ids"]), "checked", flush=True)
    with (ROOT / "revision/validation/training_targets.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    if not all(r["boundary_exact"] and r["sdf_exact"] for r in rows):
        raise AssertionError("stored target differs; preserve and investigate before training")
    print(len(rows), "stored training targets exactly match the source definition")


if __name__ == "__main__":
    main()
