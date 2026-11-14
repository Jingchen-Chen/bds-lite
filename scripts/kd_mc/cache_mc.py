"""Queue step after a multiclass teacher: diagnostics, quality gate, frozen K-class teacher cache.

revision/KD_MULTICLASS_PROTOCOL.md 3, 5. Training and validation images only. Same steps as
scripts/kd/cache_teacher.py: true fp32 (TF32 off) inference; K-class softmax cached as float16 in
manifest order; per-image, per-class DSC/IoU/Boundary F1/HD95/ASSD on training and validation
images; band diagnostics from the cached values; cached-vs-online check on 20 training images
(numpy default_rng(20261107), < 1e-3); quality gate: teacher validation Boundary F1 and DSC (means
over all image-class rows of the teacher run's own final_metrics.json, i.e. the same evaluator and
definition as the decoder-only record) at least the mean of the dataset's decoder-only seeds 1-3.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from analysis.revision.recover_predictions import case_metrics
from bds_lite.utils import seed_everything
from scripts.kd.cache_teacher import teacher_complete
from scripts.kd.common import CUDA_VERSION, TORCH_VERSION, sha, write_json
from scripts.kd_mc.common import (
    GPU_LOCK,
    ROOT,
    RUNS,
    STOP,
    cache_run,
    decoder_runs,
    kd_sources,
    require_protocol,
    teacher_run,
)
from scripts.kd_mc.data import CACHE_NAME, IDS_NAME
from scripts.kd_mc.loss import class_band
from scripts.kd_mc.teacher import build_teacher
from scripts.revision.training_support import ManifestDataset

CHECK_SEED, CHECK_COUNT, CHECK_TOLERANCE = 20261107, 20, 1e-3
CONSISTENCY_TOLERANCE = 0.005
STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def decoder_reference(dataset: str) -> dict:
    values = [
        json.loads((r / "final_metrics.json").read_text())["metrics"]["val"]
        for r in decoder_runs(dataset)
    ]
    return dict(
        boundary_f1_min=float(np.mean([v["boundary_f1"]["mean"] for v in values])),
        dsc_min=float(np.mean([v["dsc"]["mean"] for v in values])),
        runs=[str(r.relative_to(ROOT)) for r in decoder_runs(dataset)],
    )


@torch.no_grad()
def infer(model, dataset, classes, out: Path, cache, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    rows, band_entropy, band_tv, image_tv, band_sizes = [], [], [], [], []
    index = 0
    for batch in loader:
        if STOP_REQUESTED or STOP.exists():
            write_json(out / "interrupted.json", dict(reason="stop requested"))
            raise SystemExit(130)
        logits = model(batch["image"].cuda().float())["seg_logits"]
        stored = torch.softmax(logits, dim=1).half()
        pred = logits.argmax(1).cpu().numpy().astype(np.uint8)
        band = class_band(stored.float()).cpu().numpy().astype(bool)
        p = stored.float().cpu().numpy()
        for i, sample in enumerate(batch["id"]):
            truth = batch["mask"][i].numpy()
            for label in range(1, classes):
                rows.append(dict(id=sample, **case_metrics(pred[i], truth, label)))
            onehot = np.eye(classes, dtype=np.float32)[truth].transpose(2, 0, 1)
            tv = 0.5 * np.abs(p[i] - onehot).sum(0)  # total-variation distance per pixel
            band_sizes.append(int(band[i].sum()))
            image_tv.append(float(tv.mean()))
            if band[i].any():
                q = np.clip(p[i][:, band[i]].astype(np.float64), 1e-6, 1.0)
                band_entropy.append(-(q * np.log2(q)).sum(0) / np.log2(classes))
                band_tv.append(tv[band[i]])
        if cache is not None:
            cache[index : index + len(pred)] = stored.cpu().numpy()
        index += len(pred)
    summary = {
        metric: float(np.mean([r[metric] for r in rows if r[metric] is not None]))
        for metric in ["dsc", "iou", "boundary_f1", "hd95", "assd"]
    }
    per_class = {
        label: {
            metric: float(
                np.mean([r[metric] for r in rows if r["label"] == label and r[metric] is not None])
            )
            for metric in ["dsc", "boundary_f1"]
        }
        for label in range(1, classes)
    }
    entropy = np.concatenate(band_entropy) if band_entropy else np.zeros(0)
    diagnostics = dict(
        images=index,
        metrics=summary,
        per_class=per_class,
        empty_band_fraction=float(np.mean([s == 0 for s in band_sizes])),
        band_pixels_mean=float(np.mean(band_sizes)),
        band_normalised_entropy_mean=float(entropy.mean()) if entropy.size else None,
        band_normalised_entropy_histogram_10bins=np.histogram(entropy, 10, (0, 1))[0].tolist(),
        band_tv_distance_mean=float(np.concatenate(band_tv).mean()) if band_tv else None,
        image_tv_distance_mean=float(np.mean(image_tv)),
    )
    return rows, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    lock_fd = int(os.environ["BDS_REVISION_GPU_LOCK_FD"])
    if os.fstat(lock_fd).st_ino != GPU_LOCK.stat().st_ino:
        raise RuntimeError("queue lock identity mismatch")
    if torch.__version__ != TORCH_VERSION or torch.version.cuda != CUDA_VERSION:
        raise RuntimeError("environment differs from freeze")
    config = yaml.safe_load(args.config.read_text())
    require_protocol(config)
    dataset = config["dataset"]["name"]
    if config["kd"]["role"] != "teacher":
        raise ValueError("cache step takes the registered teacher config")
    out = args.output.resolve()
    if out != (RUNS / cache_run(dataset)).resolve():
        raise ValueError("cache output path differs from registration")
    if (out / "cache_complete.json").exists():
        raise FileExistsError("cache already complete; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    (out / "interrupted.json").unlink(missing_ok=True)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    torch.set_num_threads(4)
    seed_everything(config["project"]["seed"])
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    teacher_dir = RUNS / teacher_run(dataset)
    teacher_record = teacher_complete(teacher_dir)
    reference = decoder_reference(dataset)
    payload = torch.load(teacher_dir / "best.pt", map_location="cpu", weights_only=False)
    model = build_teacher(config, pretrained=False)
    model.load_state_dict(payload["model"], strict=True)
    model = model.cuda().eval()
    classes = config["dataset"]["num_classes"]
    batch_size = config["trainer"]["batch_size"]
    sets = {}
    for split in ["train", "val"]:
        record = config["revision"]["manifests"][split]
        if sha(ROOT / record["path"]) != record["sha256"]:
            raise ValueError("manifest bytes changed")
        manifest = json.loads((ROOT / record["path"]).read_text())
        sets[split] = (
            ManifestDataset(
                ROOT / config["dataset"]["processed_dir"],
                split,
                manifest,
                classes,
                config["dataset"]["in_channels"],
            ),
            manifest,
        )
    train_set, train_manifest = sets["train"]
    cache = np.zeros((len(train_set), classes, 224, 224), dtype=np.float16)
    train_rows, train_diag = infer(model, train_set, classes, out, cache, batch_size)
    val_rows, val_diag = infer(model, sets["val"][0], classes, out, None, batch_size)
    recorded_val = {k: v["mean"] for k, v in teacher_record["metrics"]["val"].items()}
    consistency = {m: val_diag["metrics"][m] - recorded_val[m] for m in val_diag["metrics"]}
    for metric in ["dsc", "iou", "boundary_f1"]:
        if abs(consistency[metric]) > CONSISTENCY_TOLERANCE:
            raise ValueError(f"teacher validation {metric} differs from its training-run record")
    temporary = out / (CACHE_NAME + ".tmp.npy")
    np.save(temporary, cache)
    temporary.replace(out / CACHE_NAME)
    write_json(out / IDS_NAME, train_manifest["sample_ids"])
    cache_sha = sha(out / CACHE_NAME)
    reloaded = np.load(out / CACHE_NAME, mmap_mode="r")
    picks = np.random.default_rng(CHECK_SEED).choice(len(train_set), CHECK_COUNT, replace=False)
    differences = []
    with torch.no_grad():
        for i in picks.tolist():
            logits = model(train_set[i]["image"][None].cuda().float())["seg_logits"]
            online = torch.softmax(logits, dim=1)[0].cpu().numpy()
            differences.append(float(np.abs(online - reloaded[i].astype(np.float32)).max()))
    check = dict(
        seed=CHECK_SEED,
        indices=picks.tolist(),
        max_abs_difference=max(differences),
        tolerance=CHECK_TOLERANCE,
        passed=max(differences) < CHECK_TOLERANCE,
    )
    if not check["passed"]:
        raise ValueError(f"cache check failed: {check['max_abs_difference']}")
    teacher_val = dict(boundary_f1=recorded_val["boundary_f1"], dsc=recorded_val["dsc"])
    passed = (
        teacher_val["boundary_f1"] >= reference["boundary_f1_min"]
        and teacher_val["dsc"] >= reference["dsc_min"]
    )
    gate = dict(
        status="PASSED" if passed else "FAILED",
        rule="teacher validation Boundary F1 >= decoder-only mean AND DSC >= decoder-only mean",
        teacher=teacher_val,
        thresholds=reference,
    )
    for name, rows in [("per_case_train.csv", train_rows), ("per_case_val.csv", val_rows)]:
        with (out / name).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    write_json(out / "diagnostics.json", dict(train=train_diag, val=val_diag))
    write_json(
        out / "cache_complete.json",
        dict(
            config_sha256=sha(args.config),
            teacher_best_sha256=sha(teacher_dir / "best.pt"),
            teacher_selected_epoch=teacher_record["selected_epoch"],
            cache_sha256=cache_sha,
            cache_dtype="float16",
            cache_shape=list(cache.shape),
            ids_sha256=sha(out / IDS_NAME),
            check=check,
            validation_consistency=dict(
                recomputed_minus_recorded=consistency, tolerance=CONSISTENCY_TOLERANCE
            ),
            gate=gate,
            diagnostics_sha256=sha(out / "diagnostics.json"),
            source_hashes=kd_sources(),
            entropy_definition="categorical entropy in bits divided by log2(K), in [0, 1]",
        ),
    )


if __name__ == "__main__":
    main()
