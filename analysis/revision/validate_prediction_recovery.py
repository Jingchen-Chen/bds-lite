"""Independent label-map/metric/lineage validation for completed recovery cohorts."""

import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from bds_lite.paths import current

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    queues = [
        json.loads((ROOT / current(f"revision/{name}")).read_text())
        for name in [
            "prediction_recovery_queue.json",
            "main_prediction_queue.json",
            "gate_identity_queue.json",
        ]
    ]
    queue = {"runs": [entry for q in queues for entry in q["runs"]]}
    cache = {}
    checks = []
    for entry in queue["runs"]:
        folder = ROOT / entry["output"]
        marker = folder / "evaluation_complete.json"
        if not marker.exists():
            continue
        completion = json.loads(marker.read_text())
        if completion["entry"] != entry:
            raise ValueError("completion identity mismatch")
        for name, digest in completion["outputs"].items():
            if sha(folder / name) != digest:
                raise ValueError("completed output hash mismatch")
        key = (entry["dataset"], entry["split"])
        if key not in cache:
            targets = {}
            if entry["split"].startswith("official"):
                split = "Validation" if entry["split"] == "official_validation" else "Test"
                raw = (
                    ROOT
                    / "data/revision/isic2018_official/archives"
                    / f"ISIC2018_Task1_{split}_GroundTruth.zip"
                )
                with zipfile.ZipFile(raw) as masks:
                    for name in masks.namelist():
                        if name.endswith(".png") and not name.startswith("__MACOSX/"):
                            sid = Path(name).stem.removesuffix("_segmentation")
                            target = (
                                Image.open(io.BytesIO(masks.read(name)))
                                .convert("L")
                                .resize((224, 224), Image.Resampling.NEAREST)
                            )
                            targets[sid] = (np.asarray(target) > 0).astype(np.uint8)
            else:
                ids = json.loads((ROOT / entry["manifest"]).read_text())["sample_ids"]
                for sid in ids:
                    with np.load(
                        ROOT / f"data/processed/{entry['dataset']}/{entry['split']}/{sid}.npz"
                    ) as z:
                        targets[sid] = z["mask"]
            cache[key] = targets
        targets = cache[key]
        records = list(csv.DictReader((folder / "per_case.csv").open()))
        index = {(row["id"], int(row["label"])): row for row in records}
        if len(index) != len(targets) * (entry["classes"] - 1):
            raise ValueError("missing/duplicated image-class rows")
        max_error = 0.0
        with np.load(folder / "predictions.npz") as z:
            ids = z["ids"]
            pred = z["predictions"]
            if set(ids) != set(targets) or len(ids) != len(set(ids)):
                raise ValueError("prediction IDs differ from cohort")
            if (
                pred.shape != (len(ids), 224, 224)
                or pred.dtype != np.uint8
                or not np.isin(pred, np.arange(entry["classes"])).all()
            ):
                raise ValueError("invalid label maps")
            for sid, prediction in zip(ids, pred, strict=True):
                for label in range(1, entry["classes"]):
                    p = prediction == label
                    t = targets[sid] == label
                    intersection = np.count_nonzero(p & t)
                    den = np.count_nonzero(p) + np.count_nonzero(t)
                    dice = (2 * intersection + 1e-6) / (den + 1e-6)
                    iou = (intersection + 1e-6) / (den - intersection + 1e-6)
                    row = index[sid, label]
                    max_error = max(
                        max_error, abs(dice - float(row["dsc"])), abs(iou - float(row["iou"]))
                    )
        if max_error > 1e-12:
            raise ValueError("independent overlap metrics differ")
        checks.append(
            dict(
                run_id=entry["run_id"],
                images=len(targets),
                class_rows=len(records),
                max_overlap_error=max_error,
                output_hashes_valid=True,
            )
        )
    with (ROOT / "results/validation/prediction_recovery.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(checks[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(checks)
    print(
        len(checks),
        "validated cohorts;",
        sum(r["images"] for r in checks),
        "maps;",
        sum(r["class_rows"] for r in checks),
        "image-class rows; planned",
        len(queue["runs"]),
    )


if __name__ == "__main__":
    main()
