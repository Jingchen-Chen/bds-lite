"""Check all native Synapse test labels against frozen processed GT conversion."""

import csv
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def main():
    rows = []
    for path in sorted((ROOT / "data/raw/Synapse/test_vol_h5").glob("*.h5")):
        case = path.name.split(".")[0]
        mismatch = 0
        checked = 0
        with h5py.File(path, "r") as handle:
            for index in range(handle["label"].shape[0]):
                native = handle["label"][index]
                expected = np.asarray(
                    Image.fromarray(native).resize((224, 224), Image.Resampling.NEAREST)
                )
                with np.load(
                    ROOT / f"data/processed/synapse/test/{case}_slice{index:03d}.npz"
                ) as z:
                    actual = z["mask"]
                mismatch += int(np.count_nonzero(actual != expected))
                checked += 1
        rows.append(dict(case=case, slices=checked, pixel_mismatches=mismatch))
        if mismatch:
            raise ValueError(f"native/processed label conversion mismatch: {case}")
    with (ROOT / "results/validation/native_synapse_labels.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(
        sum(r["slices"] for r in rows),
        "slices;",
        sum(r["pixel_mismatches"] for r in rows),
        "pixel mismatches",
    )


if __name__ == "__main__":
    main()
