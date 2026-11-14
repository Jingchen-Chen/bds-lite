"""WP1a fixed-checkpoint inference on official ISIC archives; never selects models."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from bds_lite.data.boundary import mask_to_boundary
from bds_lite.evaluation.metrics import assd, boundary_f1, dice_score, hd95, iou_score
from bds_lite.training import build_model
from bds_lite.training.runner import inference_forward
from bds_lite.utils import seed_everything

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def metrics(pred, target):
    d, a = hd95(pred, target), assd(pred, target)
    empty_p, empty_t = not pred.any(), not target.any()
    penalty = float(np.hypot(*pred.shape)) if empty_p != empty_t else 0.0
    return dict(
        dsc=dice_score(pred, target, 1),
        iou=iou_score(pred, target, 1),
        boundary_f1=boundary_f1(mask_to_boundary(pred), mask_to_boundary(target)),
        hd95=d if np.isfinite(d) else None,
        assd=a if np.isfinite(a) else None,
        hd95_penalized=d if np.isfinite(d) else penalty,
        assd_penalized=a if np.isfinite(a) else penalty,
        empty_prediction=empty_p,
        empty_target=empty_t,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["Validation", "Test"], required=True)
    args = parser.parse_args()
    # Require the exact prospective document to exist in the recorded commit.
    registered = subprocess.check_output(
        ["git", "show", "a6dc26c:analysis/revision/prereg_revision.md"], cwd=ROOT
    )
    if registered != (ROOT / "analysis/revision/prereg_revision.md").read_bytes():
        raise ValueError("prospective plan differs from committed version")
    parity = json.loads((ROOT / "revision/validation/official_inference_parity.json").read_text())
    if len(parity) != 9 or any(r["pixel_mismatches"] for r in parity):
        raise ValueError("inference parity gate has not passed")
    archives = ROOT / "data/revision/isic2018_official/archives"
    receipt = json.loads((ROOT / "revision/sources/isic_acquisition.json").read_text())
    input_path = archives / f"ISIC2018_Task1-2_{args.split}_Input.zip"
    mask_path = archives / f"ISIC2018_Task1_{args.split}_GroundTruth.zip"
    for path in [input_path, mask_path]:
        record = next(r for r in receipt if r["path"] == str(path.relative_to(ROOT)))
        if sha(path) != record["sha256"]:
            raise ValueError("archive hash mismatch")
    output = ROOT / f"revision/official_isic/{args.split.lower()}"
    output.mkdir(parents=True, exist_ok=True)
    expected_n = 100 if args.split == "Validation" else 1000
    with zipfile.ZipFile(input_path) as images, zipfile.ZipFile(mask_path) as masks:
        image_map = {
            Path(n).stem: n
            for n in images.namelist()
            if n.lower().endswith(".jpg") and not n.startswith("__MACOSX/")
        }
        mask_map = {
            Path(n).stem.removesuffix("_segmentation"): n
            for n in masks.namelist()
            if n.lower().endswith(".png") and not n.startswith("__MACOSX/")
        }
        ids = sorted(image_map)
        if len(ids) != expected_n or set(ids) != set(mask_map):
            raise ValueError("official image/GT count or ID mismatch")
        development = set()
        for split in ["train", "val"]:
            development.update(
                json.loads((ROOT / f"recovered/followup/splits/isic2018/{split}.json").read_text())[
                    "sample_ids"
                ]
            )
        if set(ids) & development:
            raise ValueError("official IDs overlap training/development")
        train_hashes = {
            sha(p)
            for p in (ROOT / "data/raw/isic2018/ISIC2018_Task1-2_Training_Input").glob("*.jpg")
        }
        sample_data = []
        byte_overlap = []
        for sid in ids:
            raw = images.read(image_map[sid])
            if hashlib.sha256(raw).hexdigest() in train_hashes:
                byte_overlap.append(sid)
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            gt_image = Image.open(io.BytesIO(masks.read(mask_map[sid]))).convert("L")
            original = (np.asarray(gt_image) > 0).astype(np.uint8)
            if image.size != gt_image.size:
                raise ValueError("native image/mask size mismatch")
            # Identical converter: PIL bilinear RGB /255 then CHW; no zscore in loader.
            rgb = (
                np.asarray(image.resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32)
                / 255
            )
            gt = (np.asarray(gt_image.resize((224, 224), Image.Resampling.NEAREST)) > 0).astype(
                np.uint8
            )
            sample_data.append((sid, rgb.transpose(2, 0, 1), gt, original))
        identity = dict(
            n=expected_n,
            id_overlap=[],
            exact_jpeg_byte_overlap=byte_overlap,
            patient_independence="not established: incomplete identities",
            scope="official held-out images; no patient-independent claim",
            preprocessing="PIL RGB bilinear 224 /255 CHW; masks nearest >0; no loader zscore",
            preregistration_commit="a6dc26c",
        )
        (output / "identity_and_preprocessing.json").write_text(
            json.dumps(identity, indent=2) + "\n"
        )
        if byte_overlap:
            raise ValueError("exact image overlap with training")
    with (ROOT / "revision/.gpu.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
        ).strip()
        if active:
            raise RuntimeError(f"GPU occupied; no process interrupted: {active}")
        for arm in "USF":
            for seed in [1, 2, 3]:
                seed_everything(seed)
                run = output / f"{arm}_seed{seed}"
                if run.exists():
                    raise FileExistsError(f"refusing overwrite: {run}")
                run.mkdir()
                checkpoint = (
                    ROOT
                    / "recovered/followup/results/followup_20261018_v2"
                    / f"isic2018_{arm}_seed{seed}/best.pt"
                )
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                model = build_model(payload["config"])
                model.load_state_dict(payload["model"], strict=True)
                model = model.cuda().eval()
                preds = []
                with torch.inference_mode():
                    for start in range(0, len(sample_data), 8):
                        batch = (
                            torch.from_numpy(
                                np.ascontiguousarray(
                                    np.stack([x[1] for x in sample_data[start : start + 8]])
                                )
                            )
                            .contiguous()
                            .cuda()
                        )
                        preds.extend(
                            inference_forward(model, batch)["seg_logits"]
                            .argmax(1)
                            .cpu()
                            .numpy()
                            .astype(np.uint8)
                        )
                records = []
                for (sid, _, gt, original), pred in zip(sample_data, preds, strict=True):
                    row = dict(id=sid, **metrics(pred, gt))
                    native = np.asarray(
                        Image.fromarray(pred).resize(
                            (original.shape[1], original.shape[0]), Image.Resampling.NEAREST
                        )
                    )
                    union = np.logical_or(native, original).sum()
                    j = float(np.logical_and(native, original).sum() / union) if union else 1.0
                    row.update(native_jaccard=j, native_thresholded_jaccard=j if j >= 0.65 else 0.0)
                    records.append(row)
                with (run / "per_case.csv").open("w") as stream:
                    writer = csv.DictWriter(
                        stream, fieldnames=list(records[0]), lineterminator="\n"
                    )
                    writer.writeheader()
                    writer.writerows(records)
                np.savez_compressed(
                    run / f"preds_{args.split.lower()}.npz",
                    ids=np.array(ids),
                    predictions=np.stack(preds),
                )
                config = dict(
                    checkpoint=str(checkpoint.relative_to(ROOT)),
                    sha256=sha(checkpoint),
                    epoch=payload["epoch"],
                    preregistration_commit="a6dc26c",
                    fp32=True,
                    batch=8,
                    split=args.split,
                    n=expected_n,
                    config=payload["config"],
                )
                (run / "provenance.json").write_text(json.dumps(config, indent=2) + "\n")
                print("Completed", run.name, args.split, len(records), flush=True)
                del model, payload
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
