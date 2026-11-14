"""Queue step after the E2 compact teacher: frozen softmax cache on the training images.

Registration section 3 (E2). Same steps as the frozen scripts/kd_mc/cache_mc.py, reusing its
unchanged infer(): true fp32 (TF32 off) inference of the selected checkpoint (best.pt, eval mode);
the 4-class softmax cached as float16 in manifest order; per-image, per-class metrics on the
training and validation images; band diagnostics; cached-vs-online check on 20 training images
(numpy default_rng(20261107), < 1e-3); validation consistency with the teacher run's own
final_metrics.json (tolerance 0.005). Differences: the teacher is the compact U-Net trained with
the D recipe (build_model, not the smp teacher), and there is no quality gate (registered as
NOT_APPLICABLE: the E2 teacher is by design not stronger than the student).
Training and validation images only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from bds_lite.training import build_model  # noqa: E402
from bds_lite.utils import seed_everything  # noqa: E402
from scripts.kd.common import CUDA_VERSION, TORCH_VERSION, sha, write_json  # noqa: E402
from scripts.kd_followup.common import (  # noqa: E402
    PEER_CACHE_RUN,
    PEER_TEACHER_RUN,
    RUNS,
    arm_of,
    validate,
)
from scripts.kd_mc import cache_mc  # noqa: E402
from scripts.kd_mc.common import GPU_LOCK, kd_sources  # noqa: E402
from scripts.kd_mc.data import CACHE_NAME, IDS_NAME  # noqa: E402
from scripts.revision.training_support import ManifestDataset  # noqa: E402


def teacher_record(folder: Path) -> dict:
    record = json.loads((folder / "final_metrics.json").read_text())
    if record.get("epochs") != 150 or record.get("test_evaluated") is not False:
        raise RuntimeError("compact teacher run is not complete")
    for name, digest in record["artifacts"].items():
        if sha(folder / name) != digest:
            raise ValueError(f"compact teacher artifact changed: {name}")
    return record


def main() -> None:
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
    validate(config)
    if arm_of(config) != "e2_teacher":
        raise ValueError("cache step takes the registered E2 teacher config")
    out = args.output.resolve()
    if out != (RUNS / PEER_CACHE_RUN).resolve():
        raise ValueError("cache output path differs from registration")
    if (out / "cache_complete.json").exists():
        raise FileExistsError("cache already complete; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    (out / "interrupted.json").unlink(missing_ok=True)
    signal.signal(signal.SIGTERM, cache_mc.request_stop)
    signal.signal(signal.SIGINT, cache_mc.request_stop)
    torch.set_num_threads(4)
    seed_everything(config["project"]["seed"])
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    teacher_dir = RUNS / PEER_TEACHER_RUN
    record = teacher_record(teacher_dir)
    payload = torch.load(teacher_dir / "best.pt", map_location="cpu", weights_only=False)
    model = build_model(config)
    model.load_state_dict(payload["model"], strict=True)
    model = model.cuda().eval()
    classes = config["dataset"]["num_classes"]
    batch_size = config["trainer"]["batch_size"]
    sets = {}
    for split in ["train", "val"]:
        entry = config["revision"]["manifests"][split]
        if sha(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError("manifest bytes changed")
        manifest = json.loads((ROOT / entry["path"]).read_text())
        data = ManifestDataset(
            ROOT / config["dataset"]["processed_dir"],
            split,
            manifest,
            classes,
            config["dataset"]["in_channels"],
        )
        sets[split] = (data, manifest)
    train_set, train_manifest = sets["train"]
    cache = np.zeros((len(train_set), classes, 224, 224), dtype=np.float16)
    train_rows, train_diag = cache_mc.infer(model, train_set, classes, out, cache, batch_size)
    val_rows, val_diag = cache_mc.infer(model, sets["val"][0], classes, out, None, batch_size)
    recorded_val = {k: v["mean"] for k, v in record["metrics"]["val"].items()}
    consistency = {m: val_diag["metrics"][m] - recorded_val[m] for m in val_diag["metrics"]}
    for metric in ["dsc", "iou", "boundary_f1"]:
        if abs(consistency[metric]) > cache_mc.CONSISTENCY_TOLERANCE:
            raise ValueError(f"teacher validation {metric} differs from its training-run record")
    temporary = out / (CACHE_NAME + ".tmp.npy")
    np.save(temporary, cache)
    temporary.replace(out / CACHE_NAME)
    write_json(out / IDS_NAME, train_manifest["sample_ids"])
    cache_sha = sha(out / CACHE_NAME)
    reloaded = np.load(out / CACHE_NAME, mmap_mode="r")
    picks = np.random.default_rng(cache_mc.CHECK_SEED).choice(
        len(train_set), cache_mc.CHECK_COUNT, replace=False
    )
    differences = []
    with torch.no_grad():
        for i in picks.tolist():
            logits = model(train_set[i]["image"][None].cuda().float())["seg_logits"]
            online = torch.softmax(logits, dim=1)[0].cpu().numpy()
            differences.append(float(np.abs(online - reloaded[i].astype(np.float32)).max()))
    check = dict(
        seed=cache_mc.CHECK_SEED,
        indices=picks.tolist(),
        max_abs_difference=max(differences),
        tolerance=cache_mc.CHECK_TOLERANCE,
        passed=max(differences) < cache_mc.CHECK_TOLERANCE,
    )
    if not check["passed"]:
        raise ValueError(f"cache check failed: {check['max_abs_difference']}")
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
            teacher_selected_epoch=record["selected_epoch"],
            cache_sha256=cache_sha,
            cache_dtype="float16",
            cache_shape=list(cache.shape),
            ids_sha256=sha(out / IDS_NAME),
            check=check,
            validation_consistency=dict(
                recomputed_minus_recorded=consistency,
                tolerance=cache_mc.CONSISTENCY_TOLERANCE,
            ),
            gate=dict(
                status="NOT_APPLICABLE",
                rule="none: the E2 teacher is the student architecture trained with the D recipe",
                teacher=dict(boundary_f1=recorded_val["boundary_f1"], dsc=recorded_val["dsc"]),
            ),
            diagnostics_sha256=sha(out / "diagnostics.json"),
            source_hashes=kd_sources(),
            entropy_definition="categorical entropy in bits divided by log2(K), in [0, 1]",
        ),
    )


if __name__ == "__main__":
    main()
