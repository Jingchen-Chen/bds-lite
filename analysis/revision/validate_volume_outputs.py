"""Real-volume EDT spot checks and full output/protocol integrity validation."""

import csv
import hashlib
import json
import re
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    completions = list((ROOT / "revision/volume_results").rglob("volume_complete.json"))
    protocol = sha(ROOT / "revision/volume_evaluation_protocol.md")
    for path in completions:
        marker = json.loads(path.read_text())
        if marker["protocol_sha256"] != protocol or marker["per_volume_sha256"] != sha(
            path.parent / "per_volume_class.csv"
        ):
            raise ValueError("volume output/protocol hash differs")
    checks = []
    structure = np.zeros((3, 3, 3), bool)
    structure[1, 1, :] = True
    structure[1, :, 1] = True
    structure[:, 1, 1] = True
    for dataset, volume, labels in [
        ("acdc", "patient011_01", [1, 2, 3]),
        ("synapse", "case0001", [1]),
    ]:
        folder = ROOT / f"revision/prediction_recovery/main_p1b_unet/{dataset}_seed1/test"
        with np.load(folder / "predictions.npz") as z:
            available = dict(zip(z["ids"], z["predictions"], strict=True))
        ids = []
        for sid in available:
            pattern = (
                r"(patient\d+)_frame(\d+)_slice_?(\d+)"
                if dataset == "acdc"
                else r"(case\d+)_slice_?(\d+)"
            )
            match = re.fullmatch(pattern, sid)
            if "_".join(match.groups()[:-1]) == volume:
                ids.append((int(match.groups()[-1]), sid))
        ids.sort()
        pred = np.stack([available[sid] for _, sid in ids])
        if dataset == "acdc":
            masks = []
            for _, sid in ids:
                with np.load(ROOT / f"data/processed/acdc/test/{sid}.npz") as z:
                    masks.append(z["mask"])
            target = np.stack(masks)
        else:
            with h5py.File(ROOT / f"data/raw/Synapse/test_vol_h5/{volume}.npy.h5") as raw:
                target = raw["label"][:]
            pred = ndi.zoom(pred, (1, target.shape[1] / 224, target.shape[2] / 224), order=0)
        rows = list(
            csv.DictReader(
                (
                    ROOT
                    / f"revision/volume_results/main_p1b_unet/{dataset}_seed1/per_volume_class.csv"
                ).open()
            )
        )
        for label in labels:
            row = next(r for r in rows if r["volume"] == volume and int(r["label"]) == label)
            p = pred == label
            t = target == label
            if not p.any() or not t.any():
                raise ValueError("fixed real-volume check needs nonempty surfaces")
            dsc = 2 * np.count_nonzero(p & t) / (np.count_nonzero(p) + np.count_nonzero(t))
            coords = np.where(p | t)
            slices = tuple(slice(int(c.min()), int(c.max()) + 1) for c in coords)
            p = np.pad(p[slices], 1)
            t = np.pad(t[slices], 1)
            a = p & ~ndi.binary_erosion(p, structure=structure)
            b = t & ~ndi.binary_erosion(t, structure=structure)
            distances = np.concatenate(
                [ndi.distance_transform_edt(~b)[a], ndi.distance_transform_edt(~a)[b]]
            )
            error = max(
                abs(dsc - float(row["dsc"])),
                abs(np.percentile(distances, 95) - float(row["hd95"])),
                abs(np.mean(distances) - float(row["assd_pooled"])),
            )
            if error > 1e-10:
                raise ValueError("real-volume independent EDT mismatch")
            checks.append(
                dict(dataset=dataset, volume=volume, label=label, max_metric_error=float(error))
            )
    result = dict(
        output_cohorts_verified=len(completions),
        protocol_sha256=protocol,
        real_volume_edt_checks=checks,
        scope="Output hashes and fixed EDT checks; not geometry validation",
    )
    (ROOT / "revision/validation/volume_outputs.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        "Verified",
        len(completions),
        "cohort hashes and",
        len(checks),
        "real-volume EDT metric checks",
    )


if __name__ == "__main__":
    main()
