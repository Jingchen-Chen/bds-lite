"""WP1b stacked/native index-grid evaluation with explicit reference/empty policies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from bds_lite.paths import current

ROOT = Path(__file__).resolve().parents[2]


def surface(mask):
    return mask ^ ndi.binary_erosion(mask)


def distances(pred, target):
    a, b = surface(pred), surface(target)
    if not a.any() or not b.any():
        return None
    pa, pb = np.argwhere(a), np.argwhere(b)
    ab = cKDTree(pb).query(pa, k=1, workers=1)[0]
    ba = cKDTree(pa).query(pb, k=1, workers=1)[0]
    return np.concatenate([ab, ba])


def validate_backend():
    rows = []
    for shift in [0, 1, 3]:
        p = np.zeros((9, 13, 15), bool)
        t = np.zeros_like(p)
        p[0:5, 2:8, 3 + shift : 8 + shift] = True
        t[1:6, 3:9, 4:9] = True
        d = distances(p, t)
        a, b = surface(p), surface(t)
        reference = np.concatenate(
            [ndi.distance_transform_edt(~b)[a], ndi.distance_transform_edt(~a)[b]]
        )
        error = float(np.max(np.abs(d - reference)))
        if error > 1e-12:
            raise ValueError("KDTree/EDT mismatch")
        rows.append(dict(shift=shift, distances=len(d), max_error=error))
    return rows


def metrics(pred, target, label):
    p = pred == label
    t = target == label
    npred, ntarget = int(p.sum()), int(t.sum())
    d = distances(p, t)
    dice = 2 * int((p & t).sum()) / (npred + ntarget) if npred + ntarget else 1.0
    h = float(np.percentile(d, 95)) if d is not None else None
    a = float(np.mean(d)) if d is not None else None
    penalty = float(np.linalg.norm(pred.shape)) if bool(npred) != bool(ntarget) else 0.0
    if npred and ntarget:
        ref_dice, ref_hd = dice, h
    elif npred and not ntarget:
        ref_dice, ref_hd = 1.0, 0.0
    else:
        ref_dice, ref_hd = 0.0, 0.0
    return dict(
        label=label,
        dsc=dice,
        hd95=h,
        assd_pooled=a,
        hd95_penalized=h if h is not None else penalty,
        assd_penalized=a if a is not None else penalty,
        transunet_dsc=ref_dice,
        transunet_hd95=ref_hd,
        pred_voxels=npred,
        gt_voxels=ntarget,
        empty_prediction=not bool(npred),
        empty_target=not bool(ntarget),
        pred_components=int(ndi.label(p)[1]),
        gt_components=int(ndi.label(t)[1]),
    )


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def evaluate(entry):
    folder = ROOT / entry["output"]
    marker = folder / "evaluation_complete.json"
    if not marker.exists():
        return False
    source = json.loads(marker.read_text())
    if source["outputs"]["predictions.npz"] != sha(folder / "predictions.npz"):
        raise ValueError("prediction archive integrity failure")
    output = ROOT / f"results/volume/{entry['family']}/{entry['dataset']}_seed{entry['seed']}"
    output.mkdir(parents=True, exist_ok=True)
    complete = output / "volume_complete.json"
    protocol_sha = sha(ROOT / "protocols/volume/volume_evaluation_protocol.md")
    if complete.exists():
        previous = json.loads(complete.read_text())
        if (
            previous["prediction_sha256"] != source["outputs"]["predictions.npz"]
            or previous["protocol_sha256"] != protocol_sha
        ):
            raise ValueError("existing volume result provenance differs")
        if previous["per_volume_sha256"] != sha(output / "per_volume_class.csv"):
            raise ValueError("existing volume result hash differs")
        return True
    groups = defaultdict(list)
    with np.load(folder / "predictions.npz") as z:
        predictions = dict(zip(z["ids"], z["predictions"], strict=True))
    for sid in predictions:
        pattern = (
            r"(patient\d+)_frame(\d+)_slice_?(\d+)"
            if entry["dataset"] == "acdc"
            else r"(case\d+)_slice_?(\d+)"
        )
        match = re.fullmatch(pattern, sid)
        if match is None:
            raise ValueError("unparsed slice ID")
        groups["_".join(match.groups()[:-1])].append((int(match.groups()[-1]), sid))
    rows = []
    for volume, indices in sorted(groups.items()):
        indices.sort()
        numeric = [i for i, _ in indices]
        if numeric != list(range(numeric[0], numeric[-1] + 1)):
            raise ValueError("missing internal slices")
        pred = np.stack([predictions[sid] for _, sid in indices])
        if entry["dataset"] == "synapse":
            raw = ROOT / f"data/raw/Synapse/test_vol_h5/{volume}.npy.h5"
            with h5py.File(raw, "r") as handle:
                target = handle["label"][:]
            if numeric != list(range(target.shape[0])):
                raise ValueError("incomplete original z range")
            pred = ndi.zoom(pred, (1, target.shape[1] / 224, target.shape[2] / 224), order=0)
            grid = "original H5 native index grid; not mm"
            if pred.shape != target.shape:
                raise ValueError("native prediction/label shapes differ")
        else:
            masks = []
            for _, sid in indices:
                with np.load(ROOT / f"data/processed/acdc/test/{sid}.npz") as z:
                    masks.append(z["mask"])
            target = np.stack(masks)
            grid = "224 processed in-plane index grid; native geometry unverified"
        for label in range(1, entry["classes"]):
            rows.append(
                dict(
                    family=entry["family"],
                    dataset=entry["dataset"],
                    seed=entry["seed"],
                    volume=volume,
                    unit=volume.split("_")[0],
                    grid=grid,
                    shape="x".join(map(str, target.shape)),
                    first_z=numeric[0],
                    **metrics(pred, target, label),
                )
            )
    with (output / "per_volume_class.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    complete.write_text(
        json.dumps(
            dict(
                entry=entry,
                protocol_sha256=protocol_sha,
                prediction_sha256=source["outputs"]["predictions.npz"],
                per_volume_sha256=sha(output / "per_volume_class.csv"),
                volumes=len(groups),
                class_rows=len(rows),
                physical_units=False,
            ),
            indent=2,
        )
        + "\n"
    )
    print("Complete volume evaluation", entry["run_id"], len(groups), "volumes", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", default=None)
    args = parser.parse_args()
    validation = validate_backend()
    (ROOT / "results/validation/volume_distance_backend.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    entries = []
    for name in ["prediction_recovery_queue.json", "main_prediction_queue.json"]:
        entries.extend(json.loads((ROOT / current(f"revision/{name}")).read_text())["runs"])
    count = 0
    for entry in entries:
        if (
            entry["split"] == "test"
            and entry["dataset"] in ["acdc", "synapse"]
            and (not args.family or entry["family"] == args.family)
        ):
            count += evaluate(entry)
    print("Completed/verified", count, "available test cohorts")


if __name__ == "__main__":
    main()
