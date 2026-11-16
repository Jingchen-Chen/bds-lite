"""Sequential inference-only recovery of separately registered checkpoint families."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import subprocess
import time
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


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class Cohort:
    def __init__(self, entry):
        self.entry = entry
        self.images = None
        self.masks = None
        if entry["split"].startswith("official"):
            split = "Validation" if entry["split"] == "official_validation" else "Test"
            base = ROOT / "data/revision/isic2018_official/archives"
            self.images = zipfile.ZipFile(base / f"ISIC2018_Task1-2_{split}_Input.zip")
            self.masks = zipfile.ZipFile(base / f"ISIC2018_Task1_{split}_GroundTruth.zip")
            self.image_map = {
                Path(n).stem: n
                for n in self.images.namelist()
                if n.endswith(".jpg") and not n.startswith("__MACOSX/")
            }
            self.mask_map = {
                Path(n).stem.removesuffix("_segmentation"): n
                for n in self.masks.namelist()
                if n.endswith(".png") and not n.startswith("__MACOSX/")
            }
            self.ids = sorted(self.image_map)
            if set(self.ids) != set(self.mask_map) or len(self.ids) != (
                100 if split == "Validation" else 1000
            ):
                raise ValueError("official count/identity mismatch")
            identity = (
                ROOT / f"results/official_isic/{split.lower()}/identity_and_preprocessing.json"
            )
            if json.loads(identity.read_text())["id_overlap"]:
                raise ValueError("official overlap gate failed")
        else:
            manifest = ROOT / entry["manifest"]
            if sha(manifest) != entry["manifest_sha256"]:
                raise ValueError("manifest changed")
            self.ids = sorted(json.loads(manifest.read_text())["sample_ids"])
            self.directory = ROOT / f"data/processed/{entry['dataset']}/{entry['split']}"
            if set(self.ids) != {p.stem for p in self.directory.glob("*.npz")}:
                raise ValueError("processed cohort differs from manifest")
        if len(self.ids) != len(set(self.ids)):
            raise ValueError("duplicate sample IDs")

    def get(self, sid):
        if self.images is None:
            with np.load(self.directory / f"{sid}.npz") as z:
                return z["image"].astype(np.float32), z["mask"].astype(np.uint8), None
        rgb = Image.open(io.BytesIO(self.images.read(self.image_map[sid]))).convert("RGB")
        gt = Image.open(io.BytesIO(self.masks.read(self.mask_map[sid]))).convert("L")
        original = (np.asarray(gt) > 0).astype(np.uint8)
        image = (
            np.asarray(rgb.resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32) / 255
        )
        mask = (np.asarray(gt.resize((224, 224), Image.Resampling.NEAREST)) > 0).astype(np.uint8)
        return image.transpose(2, 0, 1), mask, original

    def close(self):
        if self.images:
            self.images.close()
            self.masks.close()


def case_metrics(pred, target, label):
    p = pred == label
    t = target == label
    h, a = hd95(pred, target, label), assd(pred, target, label)
    penalty = float(np.hypot(*pred.shape)) if p.any() != t.any() else 0.0
    return dict(
        label=label,
        dsc=dice_score(pred, target, label),
        iou=iou_score(pred, target, label),
        boundary_f1=boundary_f1(
            mask_to_boundary(p.astype(np.uint8)), mask_to_boundary(t.astype(np.uint8))
        ),
        hd95=h if np.isfinite(h) else None,
        assd=a if np.isfinite(a) else None,
        hd95_penalized=h if np.isfinite(h) else penalty,
        assd_penalized=a if np.isfinite(a) else penalty,
        empty_prediction=not bool(p.any()),
        empty_target=not bool(t.any()),
        target_area=int(t.sum()),
    )


class IdentityGate(torch.nn.Module):
    """Substitute g=.5, yielding exactly unit multiplicative modulation."""

    def forward(self, seg_features, boundary_proxy):
        return seg_features


def run_one(entry, protocol_sha):
    folder = ROOT / entry["output"]
    checkpoint = ROOT / entry["checkpoint"]
    if sha(checkpoint) != entry["checkpoint_sha256"]:
        raise ValueError("checkpoint differs from registered identity")
    completion = folder / "evaluation_complete.json"
    if completion.exists():
        state = json.loads(completion.read_text())
        if state["entry"] != entry or state["protocol_sha256"] != protocol_sha:
            raise ValueError("completion provenance differs")
        for name, digest in state["outputs"].items():
            if sha(folder / name) != digest:
                raise ValueError("completed output hash differs")
        print("Skip verified complete", entry["run_id"], flush=True)
        return
    if folder.exists():
        raise FileExistsError(f"unknown incomplete output, preserved: {folder}")
    folder.mkdir(parents=True)
    start = time.monotonic()
    cohort = Cohort(entry)
    try:
        seed_everything(entry["seed"])
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = build_model(payload["config"])
        model.load_state_dict(payload["model"], strict=True)
        model.cuda().eval()
        gate_parity = None
        if entry.get("force_gate_identity", False):
            if getattr(model, "boundary_gate", None) is None:
                raise ValueError("counterfactual requires an existing full-model gate")
            probe = [cohort.get(sid)[0] for sid in cohort.ids[: entry["batch"]]]
            probe = torch.from_numpy(np.ascontiguousarray(np.stack(probe))).cuda()
            model.boundary_gate.scale = 0.0
            with torch.inference_mode():
                reference = inference_forward(model, probe)["seg_logits"]
            model.boundary_gate = IdentityGate().cuda()
            with torch.inference_mode():
                observed = inference_forward(model, probe)["seg_logits"]
            if not torch.equal(reference, observed):
                raise ValueError("identity gate differs from exact unit-modulation reference")
            gate_parity = dict(reference="gate scale zero equals g=.5 factor", exact_logits=True)
        records = []
        predictions = []
        for start_index in range(0, len(cohort.ids), entry["batch"]):
            ids = cohort.ids[start_index : start_index + entry["batch"]]
            samples = [cohort.get(sid) for sid in ids]
            images = np.ascontiguousarray(np.stack([sample[0] for sample in samples]))
            with torch.inference_mode():
                pred = (
                    inference_forward(model, torch.from_numpy(images).cuda())["seg_logits"]
                    .argmax(1)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )
            predictions.extend(pred)
            for sid, sample, prediction in zip(ids, samples, pred, strict=True):
                for label in range(1, entry["classes"]):
                    row = dict(id=sid, **case_metrics(prediction, sample[1], label))
                    if sample[2] is not None:
                        native = np.asarray(
                            Image.fromarray(prediction).resize(
                                (sample[2].shape[1], sample[2].shape[0]), Image.Resampling.NEAREST
                            )
                        )
                        union = np.count_nonzero((native > 0) | (sample[2] > 0))
                        j = (
                            np.count_nonzero((native > 0) & (sample[2] > 0)) / union
                            if union
                            else 1.0
                        )
                        row.update(
                            native_jaccard=float(j),
                            native_thresholded_jaccard=float(j) if j >= 0.65 else 0.0,
                        )
                    records.append(row)
        np.savez_compressed(
            folder / "predictions.npz", ids=np.array(cohort.ids), predictions=np.stack(predictions)
        )
        with (folder / "per_case.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
        write_json(
            folder / "provenance.json",
            dict(
                entry=entry,
                embedded_config=payload["config"],
                gate_identity_operator_parity=gate_parity,
                epoch=payload["epoch"],
                protocol_sha256=protocol_sha,
                inference="FP32 contiguous NCHW; argmax; no postprocessing",
                distance_units="processed index grid; not verified mm",
            ),
        )
        outputs = {
            name: sha(folder / name)
            for name in ["predictions.npz", "per_case.csv", "provenance.json"]
        }
        write_json(
            completion,
            dict(
                entry=entry,
                protocol_sha256=protocol_sha,
                outputs=outputs,
                images=len(cohort.ids),
                class_rows=len(records),
                seconds=time.monotonic() - start,
            ),
        )
        del model, payload
        torch.cuda.empty_cache()
        print("Complete", entry["run_id"], len(cohort.ids), flush=True)
    finally:
        cohort.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--queue",
        type=Path,
        default=ROOT / "protocols/prediction_recovery/prediction_recovery_queue.json",
    )
    args = parser.parse_args()
    queue_path = args.queue.resolve()
    queue = json.loads(queue_path.read_text())
    protocol = ROOT / queue["protocol"]
    if not args.execute:
        print(len(queue["runs"]), "inference cohorts; no training; use --execute after checks")
        return
    for path in [queue_path, protocol]:
        relative = str(path.relative_to(ROOT))
        committed = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
        if committed != path.read_bytes():
            raise ValueError("prospective recovery matrix/protocol not committed unchanged")
    with (ROOT / "revision/.gpu.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
        ).strip()
        if active:
            raise RuntimeError(f"GPU occupied; no process interrupted: {active}")
        for entry in queue["runs"]:
            run_one(entry, sha(protocol))


if __name__ == "__main__":
    main()
