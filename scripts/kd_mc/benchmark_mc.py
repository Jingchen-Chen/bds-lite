"""Engineering benchmark for ACDC/Synapse before the freeze (not part of the experiment).

Seed 0. One training epoch each for the teacher and a D+KD-band student per dataset; the student
reads a dummy cache (one-hot ground truth). No validation or test metric; no model saved. Writes
revision/kd_multiclass/benchmark.json (epoch time, peak VRAM, temperature, loss finiteness).
"""

from __future__ import annotations

import fcntl
import json
import math
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from bds_lite.training import BDSLiteLoss, build_model
from bds_lite.training.losses import weights_from_config
from bds_lite.utils import seed_everything
from scripts.kd.benchmark_kd import Telemetry
from scripts.kd.common import write_json
from scripts.kd_mc.common import DATASETS, GPU_LOCK, ROOT, STOP, WORKSPACE
from scripts.kd_mc.loss import KDCriterionMC
from scripts.kd_mc.register_mc import student_config, teacher_config
from scripts.kd_mc.teacher import build_teacher
from scripts.revision.train_run import epoch_train
from scripts.revision.training_support import ManifestDataset


class DummyCacheDataset(ManifestDataset):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        onehot = F.one_hot(item["mask"].long(), self.classes).permute(2, 0, 1)
        item["teacher_prob"] = onehot.to(torch.float16)
        return item


def one(dataset: str, role: str) -> dict:
    seed_everything(0)
    config = teacher_config(dataset, "x") if role == "teacher" else student_config(dataset, 1, "x")
    manifest = json.loads((ROOT / config["revision"]["manifests"]["train"]["path"]).read_text())
    classes, channels = config["dataset"]["num_classes"], config["dataset"]["in_channels"]
    cls = DummyCacheDataset if role == "student" else ManifestDataset
    data = cls(
        ROOT / config["dataset"]["processed_dir"],
        "train",
        manifest,
        classes,
        channels,
        bool(config["loss"]["boundary_weight"]),
        False,
    )
    loader = DataLoader(
        data,
        batch_size=config["trainer"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
        generator=torch.Generator().manual_seed(0),
    )
    if role == "teacher":
        model = build_teacher(config, pretrained=True).cuda()
        criterion = BDSLiteLoss(weights_from_config(config))
    else:
        model = build_model(config).cuda()
        criterion = KDCriterionMC(BDSLiteLoss(weights_from_config(config)), 1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["trainer"]["lr"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    began = time.monotonic()
    train, _ = epoch_train(model, loader, criterion, optimizer, scheduler, 1, 0, [], False)
    torch.cuda.synchronize()
    seconds = time.monotonic() - began
    losses = {k: v for k, v in train.items() if isinstance(v, float)}
    return dict(
        dataset=dataset,
        role=role,
        batch_size=config["trainer"]["batch_size"],
        parameters=sum(p.numel() for p in model.parameters()),
        train_epoch_seconds=seconds,
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
        if subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
        ).strip():
            raise RuntimeError("GPU occupied")
        torch.set_num_threads(4)
        results = []
        for dataset in DATASETS:
            for role in ["teacher", "student"]:
                while (
                    int(
                        subprocess.check_output(
                            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                            text=True,
                        )
                    )
                    > 70
                ):
                    time.sleep(10)
                telemetry = Telemetry()
                telemetry.start()
                try:
                    result = one(dataset, role)
                finally:
                    telemetry.running = False
                    telemetry.join()
                rows = np.array(telemetry.rows)
                result |= dict(
                    max_temperature_c=float(rows[:, 0].max()),
                    max_nvidia_smi_memory_mib=float(rows[:, 1].max()),
                )
                results.append(result)
                print(json.dumps(result), flush=True)
                torch.cuda.empty_cache()
        write_json(
            WORKSPACE / "benchmark.json",
            dict(
                seed=0,
                note="engineering benchmark; dummy one-hot cache; no metric; no model saved",
                results=results,
            ),
        )


if __name__ == "__main__":
    main()
