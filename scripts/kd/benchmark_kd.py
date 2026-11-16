"""Engineering benchmark before the freeze (protocol 9, step 2). Not part of the experiment.

Seed 0 (not an experimental seed). One training epoch each for the teacher and a D+KD-band
student; the student reads a dummy teacher cache (the ground-truth masks), so no real teacher output
exists. No validation or test metric is computed and no model state is saved. Validation cost is
timed on random tensors of the validation-set size, in eval mode, with no labels. Records only loss
finiteness, epoch time, peak VRAM, and GPU temperature, in results/kd_isic2018/benchmark.json.
"""

from __future__ import annotations

import fcntl
import json
import math
import subprocess
import threading
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from bds_lite.training import BDSLiteLoss, build_model
from bds_lite.training.losses import weights_from_config
from bds_lite.utils import seed_everything
from scripts.kd.common import GPU_LOCK, ROOT, STOP, WORKSPACE, write_json
from scripts.kd.kd_loss import KDCriterion
from scripts.kd.register_kd import student_config, teacher_config
from scripts.kd.teacher import TeacherNet
from scripts.revision.train_run import epoch_train
from scripts.revision.training_support import ManifestDataset

VALIDATION_IMAGES = 519


class DummyCacheDataset(ManifestDataset):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        item["teacher_prob"] = item["mask"].to(torch.float16)
        return item


class Telemetry(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.rows, self.running = [], True

    def run(self):
        while self.running:
            line = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu,memory.used,clocks.sm,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
            self.rows.append([float(x) for x in line.split(",")])
            time.sleep(2)


def one(role: str) -> dict:
    seed_everything(0)
    config = teacher_config("benchmark") if role == "teacher" else student_config(1, "band", "x")
    config["project"]["seed"] = 0
    manifest = json.loads((ROOT / config["revision"]["manifests"]["train"]["path"]).read_text())
    cls = DummyCacheDataset if role == "student" else ManifestDataset
    dataset = cls(
        ROOT / config["dataset"]["processed_dir"],
        "train",
        manifest,
        2,
        3,
        bool(config["loss"]["boundary_weight"]),
        False,
    )
    loader = DataLoader(
        dataset,
        batch_size=config["trainer"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
        generator=torch.Generator().manual_seed(0),
    )
    if role == "teacher":
        model = TeacherNet(pretrained=True).cuda()
        criterion = BDSLiteLoss(weights_from_config(config))
    else:
        model = build_model(config).cuda()
        criterion = KDCriterion(BDSLiteLoss(weights_from_config(config)), "band", 1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["trainer"]["lr"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    began = time.monotonic()
    train, _ = epoch_train(model, loader, criterion, optimizer, scheduler, 1, 0, [], False)
    torch.cuda.synchronize()
    train_seconds = time.monotonic() - began
    model.eval()
    began = time.monotonic()
    with torch.no_grad():
        for start in range(0, VALIDATION_IMAGES, 8):
            n = min(8, VALIDATION_IMAGES - start)
            x = torch.rand(n, 3, 224, 224, device="cuda")
            model(x)["seg_logits"].argmax(1).cpu()
    torch.cuda.synchronize()
    inference_seconds = time.monotonic() - began
    losses = {k: v for k, v in train.items() if isinstance(v, float)}
    return dict(
        role=role,
        batch_size=config["trainer"]["batch_size"],
        parameters=sum(p.numel() for p in model.parameters()),
        train_epoch_seconds=train_seconds,
        validation_size_inference_seconds=inference_seconds,
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
        losses_finite=all(math.isfinite(v) for v in losses.values()),
        loss_keys=sorted(losses),
        amp_skipped_steps=train["amp_skipped_steps"],
    )


def main():
    if STOP.exists():
        raise RuntimeError("STOP_KD_TRAINING exists")
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    with GPU_LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        processes = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
        ).strip()
        if processes:
            raise RuntimeError(f"GPU occupied: {processes}")
        torch.set_num_threads(4)
        results = []
        for role in ["teacher", "student"]:
            while True:
                temperature = int(
                    subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                        text=True,
                    )
                )
                if temperature <= 70:
                    break
                time.sleep(10)
            telemetry = Telemetry()
            telemetry.start()
            try:
                result = one(role)
            finally:
                telemetry.running = False
                telemetry.join()
            rows = np.array(telemetry.rows)
            result |= dict(
                max_temperature_c=float(rows[:, 0].max()),
                max_nvidia_smi_memory_mib=float(rows[:, 1].max()),
                median_sm_clock_mhz=float(np.median(rows[:, 2])),
                samples=len(rows),
            )
            result["fits_batch_8"] = result["batch_size"] == 8
            result["expected_run_hours_without_pauses"] = (
                150
                * (result["train_epoch_seconds"] + result["validation_size_inference_seconds"])
                / 3600
            )
            results.append(result)
            print(json.dumps(result), flush=True)
            torch.cuda.empty_cache()
        write_json(
            WORKSPACE / "benchmark.json",
            dict(
                seed=0,
                note="engineering benchmark; dummy teacher cache (ground-truth masks); "
                "no validation or test metric; no model saved",
                torch=torch.__version__,
                gpu=torch.cuda.get_device_name(0),
                results=results,
            ),
        )


if __name__ == "__main__":
    main()
