# ruff: noqa: E501
"""Post hoc re-measurement of the ACDC teacher-cache generation time (inference only).

Authorized by the author on 2026-11-09 (restructure revision, "teacher cache generation time").
The original cache step (scripts/kd_mc/cache_mc.py, 2026-11-07) did not record its duration.
This script repeats only its training-set pass, with the same settings as that step:
- the same checkpoint (sha256 checked);
- the registered training manifest (sha256 checked);
- the same ManifestDataset;
- batch size from the teacher config;
- no workers;
- 4 CPU threads;
- TF32 off;
- the same seed.

It calls the frozen `scripts.kd_mc.cache_mc.infer` on an in-memory array. It never writes or
replaces the registered cache. It times three repetitions and compares each regenerated cache
with the stored one.

Output: results/acdc_confirm_eval_posthoc/teacher_cache_timing.json (refuses to overwrite).
    PYTHONPATH=src:. /home/jc/bds-followup/.venv-kd/bin/python \
        analysis/revision/posthoc_teacher_cache_timing.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from bds_lite.utils import seed_everything  # noqa: E402
from scripts.kd.common import sha, write_json  # noqa: E402
from scripts.kd_mc.cache_mc import infer  # noqa: E402
from scripts.kd_mc.teacher import build_teacher  # noqa: E402
from scripts.revision.training_support import ManifestDataset  # noqa: E402

RUN = ROOT / "results/revision_runs/acdc_kd_teacher_seed100"
CACHE = ROOT / "results/revision_runs/acdc_kd_teacher_cache"
OUT = ROOT / "results/acdc_confirm_eval_posthoc/teacher_cache_timing.json"
REPEATS = 3


def registered_config() -> Path:
    queue = yaml.safe_load((ROOT / "protocols/kd_acdc_synapse/queue_kd_mc.yaml").read_text())
    text = json.dumps(queue)
    for path in sorted((ROOT / "configs").rglob("*acdc*teacher*.yaml")):
        if path.name in text or str(path.relative_to(ROOT)) in text:
            return path
    raise FileNotFoundError(
        "registered ACDC teacher config not found in protocols/kd_acdc_synapse/queue_kd_mc.yaml"
    )


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"{OUT} exists")
    record = json.loads((CACHE / "cache_complete.json").read_text())
    config_path = registered_config()
    if sha(config_path) != record["config_sha256"]:
        raise ValueError("teacher config differs from the one used for the cache")
    if sha(RUN / "best.pt") != record["teacher_best_sha256"]:
        raise ValueError("teacher checkpoint differs from the one used for the cache")
    config = yaml.safe_load(config_path.read_text())
    torch.set_num_threads(4)
    seed_everything(config["project"]["seed"])
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    split = config["revision"]["manifests"]["train"]
    if sha(ROOT / split["path"]) != split["sha256"]:
        raise ValueError("training manifest changed")
    manifest = json.loads((ROOT / split["path"]).read_text())
    classes = config["dataset"]["num_classes"]
    data = ManifestDataset(
        ROOT / config["dataset"]["processed_dir"],
        "train",
        manifest,
        classes,
        config["dataset"]["in_channels"],
    )
    model = build_teacher(config, pretrained=False)
    model.load_state_dict(
        torch.load(RUN / "best.pt", map_location="cpu", weights_only=False)["model"], strict=True
    )
    model = model.cuda().eval()
    stored = np.load(CACHE / "train_teacher_prob_fp16.npy", mmap_mode="r")
    runs = []
    with tempfile.TemporaryDirectory() as scratch:
        for repeat in range(REPEATS):
            cache = np.zeros((len(data), classes, 224, 224), dtype=np.float16)
            torch.cuda.synchronize()
            start = time.perf_counter()
            infer(model, data, classes, Path(scratch), cache, config["trainer"]["batch_size"])
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            diff = float(np.abs(cache.astype(np.float32) - stored.astype(np.float32)).max())
            runs.append(
                dict(
                    repeat=repeat,
                    seconds=seconds,
                    max_abs_difference_to_stored_cache=diff,
                    identical_to_stored_cache=bool(np.array_equal(cache, stored)),
                )
            )
            print(runs[-1], flush=True)
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    mtimes = {
        name: datetime.fromtimestamp((CACHE / name).stat().st_mtime, timezone.utc).isoformat()
        for name in ["queue_stdout.log", "train_teacher_prob_fp16.npy", "cache_complete.json"]
    }
    write_json(
        OUT,
        dict(
            label="re-measured post hoc (author authorization 2026-11-09); inference only",
            measured_utc=datetime.now(timezone.utc).isoformat(),
            what=(
                "training-set pass of the frozen cache step (scripts.kd_mc.cache_mc.infer): teacher "
                "forward, softmax, float16 copy into the cache array, and the per-image diagnostics "
                "that the step computes in the same loop; excludes model loading, the validation "
                "pass, saving and the 20-image check"
            ),
            images=len(data),
            batch_size=config["trainer"]["batch_size"],
            config=str(config_path.relative_to(ROOT)),
            checkpoint_sha256=record["teacher_best_sha256"],
            gpu=gpu,
            torch=torch.__version__,
            cuda=torch.version.cuda,
            runs=runs,
            median_seconds=float(np.median([r["seconds"] for r in runs])),
            original_step_file_mtimes_utc=mtimes,
            original_step_note=(
                "file modification times bound the whole original cache step (training and "
                "validation passes, saving, check); they are not a recorded duration"
            ),
        ),
    )


if __name__ == "__main__":
    main()
