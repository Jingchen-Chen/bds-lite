"""Queue step after teacher training: diagnostics, quality gate, and the frozen teacher cache.

Protocol 5 and OPEN-1. Uses training and validation images only. Steps:
1. verify the teacher run is complete (150 epochs, artifact hashes) and load its selected state;
2. fp32 inference (TF32 disabled) on the 2,075 training images; foreground probabilities cached
   as float16 in manifest order; per-image DSC/IoU/Boundary F1/HD95/ASSD on training and
   validation images;
3. band diagnostics (computed from the cached float16 values that students will read);
4. cache check: 20 training images drawn with numpy default_rng(20261107), recomputed one at a time,
   maximum absolute difference to the cache < 1e-3;
5. quality gate: teacher validation Boundary F1 and DSC (means over the 519 images, from the
   teacher run's own final_metrics.json) at least the decoder-only means (scripts/kd/common.GATE),
   which are recomputed here from the ten recorded runs and must match. The gate reads the
   teacher's own training-run evaluation (same evaluator and settings as the decoder-only runs);
   the means recomputed here must agree within 0.005 (a consistency check, recorded).
The marker cache_complete.json is written last. A FAILED gate is recorded, not retried.
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
from scripts.kd.common import (
    CUDA_VERSION,
    DECODER_RUNS,
    GATE,
    GPU_LOCK,
    ROOT,
    RUNS,
    STOP,
    TEACHER_RUN,
    TORCH_VERSION,
    kd_sources,
    require_protocol,
    sha,
    write_json,
)
from scripts.kd.data import CACHE_NAME, IDS_NAME
from scripts.kd.kd_loss import teacher_band
from scripts.kd.teacher import TeacherNet
from scripts.revision.training_support import ManifestDataset

CHECK_SEED, CHECK_COUNT, CHECK_TOLERANCE = 20261107, 20, 1e-3
# Recomputed (TF32 off) vs recorded (training-run evaluation, TF32 default) validation means.
CONSISTENCY_TOLERANCE = 0.005
STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def interrupted(out: Path) -> None:
    write_json(out / "interrupted.json", dict(reason="stop requested", stop_file=STOP.exists()))
    raise SystemExit(130)


def teacher_complete(folder: Path) -> dict:
    record = json.loads((folder / "final_metrics.json").read_text())
    if record.get("epochs") != 150 or record.get("test_evaluated") is not False:
        raise ValueError("teacher run is not a complete validation-only run")
    for name, digest in record["artifacts"].items():
        if sha(folder / name) != digest:
            raise ValueError(f"teacher artifact hash mismatch: {name}")
    for name in ["preds_test.npz", "per_case_test.csv"]:
        if (folder / name).exists():
            raise ValueError("teacher folder unexpectedly contains test outputs")
    return record


def decoder_reference() -> dict:
    values = [
        json.loads((r / "final_metrics.json").read_text())["metrics"]["val"] for r in DECODER_RUNS
    ]
    reference = dict(
        boundary_f1_min=float(np.mean([v["boundary_f1"]["mean"] for v in values])),
        dsc_min=float(np.mean([v["dsc"]["mean"] for v in values])),
    )
    for key, value in reference.items():
        if value != GATE[key]:
            raise ValueError(f"recorded decoder-only mean {key} differs from the frozen gate")
    return reference


def entropy_bits(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


@torch.no_grad()
def infer(model, dataset, out: Path, cache: np.ndarray | None):
    """Per-image metrics and band statistics; fills `cache` (float16) when given."""
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    rows, band_p, band_err, all_err, band_sizes = [], [], [], [], []
    index = 0
    for batch in loader:
        if STOP_REQUESTED or STOP.exists():
            interrupted(out)
        logits = model(batch["image"].cuda().float())["seg_logits"]
        prob = torch.softmax(logits, dim=1)[:, 1]
        stored = prob.half()
        pred = logits.argmax(1).cpu().numpy().astype(np.uint8)
        band = teacher_band(stored.float()).cpu().numpy().astype(bool)
        p = stored.float().cpu().numpy()
        for i, sample in enumerate(batch["id"]):
            truth = batch["mask"][i].numpy()
            rows.append(dict(id=sample, **case_metrics(pred[i], truth, 1)))
            y = (truth == 1).astype(np.float64)
            band_sizes.append(int(band[i].sum()))
            band_p.append(p[i][band[i]])
            band_err.append(np.abs(p[i] - y)[band[i]])
            all_err.append(float(np.abs(p[i] - y).mean()))
        if cache is not None:
            cache[index : index + len(pred)] = stored.cpu().numpy()
        index += len(pred)
    band_values = np.concatenate(band_p) if band_p else np.zeros(0)
    summary = {
        metric: float(np.mean([r[metric] for r in rows if r[metric] is not None]))
        for metric in ["dsc", "iou", "boundary_f1", "hd95", "assd"]
    }
    histogram = np.histogram(band_values, bins=10, range=(0.0, 1.0))[0]
    diagnostics = dict(
        images=len(rows),
        metrics=summary,
        empty_band_fraction=float(np.mean([s == 0 for s in band_sizes])),
        band_pixels_mean=float(np.mean(band_sizes)),
        band_entropy_bits_mean=float(entropy_bits(band_values).mean())
        if band_values.size
        else None,
        band_prob_histogram_10bins=histogram.tolist(),
        band_abs_error_mean=float(np.concatenate(band_err).mean()) if band_values.size else None,
        image_abs_error_mean=float(np.mean(all_err)),
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
    if config["kd"]["role"] != "teacher":
        raise ValueError("cache step takes the registered teacher config")
    out = args.output.resolve()
    if out != (RUNS / "isic2018_kd_teacher_cache").resolve():
        raise ValueError("cache output path differs from registration")
    if (out / "cache_complete.json").exists():
        raise FileExistsError("cache already complete; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    (out / "interrupted.json").unlink(missing_ok=True)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    torch.set_num_threads(4)
    seed_everything(config["project"]["seed"])
    # True fp32 for the cache: cuDNN TF32 (PyTorch's default) makes "fp32" convolutions depend on
    # the batch size at about 1e-2 in probability, which would fail the cached-vs-online check.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    teacher_dir = RUNS / TEACHER_RUN
    teacher_record = teacher_complete(teacher_dir)
    reference = decoder_reference()
    payload = torch.load(teacher_dir / "best.pt", map_location="cpu", weights_only=False)
    model = TeacherNet(pretrained=False)
    model.load_state_dict(payload["model"], strict=True)
    model = model.cuda().eval()
    classes = config["dataset"]["num_classes"]
    sets = {}
    for split, record in config["revision"]["manifests"].items():
        path = ROOT / record["path"]
        if sha(path) != record["sha256"]:
            raise ValueError("manifest bytes changed")
        manifest = json.loads(path.read_text())
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
    cache = np.zeros((len(train_set), 224, 224), dtype=np.float16)
    train_rows, train_diag = infer(model, train_set, out, cache)
    val_rows, val_diag = infer(model, sets["val"][0], out, None)
    # Validation metrics recomputed here must equal the teacher run's own evaluation.
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
            item = train_set[i]
            logits = model(item["image"][None].cuda().float())["seg_logits"]
            online = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()
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
    teacher_val = dict(
        boundary_f1=recorded_val["boundary_f1"], dsc=recorded_val["dsc"], images=val_diag["images"]
    )
    passed = (
        teacher_val["boundary_f1"] >= reference["boundary_f1_min"]
        and teacher_val["dsc"] >= reference["dsc_min"]
    )
    gate = dict(
        status="PASSED" if passed else "FAILED",
        rule="teacher validation Boundary F1 >= decoder-only mean AND DSC >= decoder-only mean",
        teacher=teacher_val,
        thresholds=reference,
        source=GATE["source"],
    )
    for name, rows in [("per_case_train.csv", train_rows), ("per_case_val.csv", val_rows)]:
        with (out / name).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    diagnostics = dict(train=train_diag, val=val_diag)
    write_json(out / "diagnostics.json", diagnostics)
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
                recomputed_minus_recorded=consistency,
                tolerance=CONSISTENCY_TOLERANCE,
                note="cache step TF32 off; training-run evaluation TF32 on (PyTorch default)",
            ),
            gate=gate,
            diagnostics_sha256=sha(out / "diagnostics.json"),
            source_hashes=kd_sources(),
        ),
    )


if __name__ == "__main__":
    main()
