#!/usr/bin/env python
"""Repeat the Phase 15 no-update GSL profile with the validated EDT."""

from __future__ import annotations

import csv
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from gsl_memory_bounded import install_memory_bounded_gsl
from scipy import ndimage as ndi

from bds_lite.evaluation import profile_model
from bds_lite.training import BDSLiteLoss, build_model
from bds_lite.training.losses import weights_from_config

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/profiling"
WARMUP = 10
ITERATIONS = 40
BATCH_SIZE = 8
INPUT_SIZE = 224


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def config() -> dict:
    return {
        "project": {"seed": 2026},
        "experiment": {"name": "phase16_profile_unet_gsl"},
        "dataset": {
            "name": "isic2018",
            "in_channels": 3,
            "num_classes": 2,
            "input_size": [INPUT_SIZE, INPUT_SIZE],
        },
        "model": {"name": "unet", "base_channels": 32},
        "loss": {
            "dice_weight": 1.0,
            "ce_weight": 1.0,
            "boundary_weight": 0.0,
            "distill_weight": 0.0,
            "surface_weight": 0.0,
            "distance_weight": 0.0,
            "foreground_class_weights": None,
            "use_gsl": True,
            "lambda_gsl": 1.0,
            "gsl_alpha_schedule": "step5",
        },
        "trainer": {"epochs": 150, "batch_size": BATCH_SIZE, "amp": True},
    }


def synthetic_batch(device: torch.device) -> dict[str, torch.Tensor]:
    yy, xx = np.mgrid[:INPUT_SIZE, :INPUT_SIZE]
    masks = []
    for index in range(BATCH_SIZE):
        cy = 76 + (index % 4) * 18
        cx = 78 + (index // 4) * 42
        radius = 35 + (index % 3) * 6
        masks.append(((yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2).astype(np.int64))
    mask = np.stack(masks)
    boundary = np.zeros_like(mask, dtype=np.float32)
    sdf = np.zeros_like(mask, dtype=np.float32)
    for index, sample in enumerate(mask):
        eroded = ndi.binary_erosion(sample, iterations=1)
        boundary[index] = np.logical_xor(sample, eroded)
        inside = ndi.distance_transform_edt(sample)
        outside = ndi.distance_transform_edt(1 - sample)
        sdf[index] = outside - inside
    generator = torch.Generator(device="cpu").manual_seed(20260609)
    image = torch.randn((BATCH_SIZE, 3, INPUT_SIZE, INPUT_SIZE), generator=generator)
    return {
        "image": image.to(device),
        "mask": torch.from_numpy(mask).to(device),
        "boundary": torch.from_numpy(boundary[:, None]).to(device),
        "sdf": torch.from_numpy(sdf[:, None]).to(device),
    }


def backward(
    model: torch.nn.Module,
    criterion: BDSLiteLoss,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> float:
    model.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type=device.type, enabled=True):
        outputs = model(batch["image"])
        loss, _ = criterion(outputs, batch)
    loss.backward()
    return float(loss.detach().cpu())


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the controlled GSL profile")
    install_memory_bounded_gsl()
    device = torch.device("cuda")
    torch.manual_seed(20260609)
    cfg = config()
    model = build_model(cfg).to(device)
    criterion = BDSLiteLoss(weights_from_config(cfg)).to(device)
    criterion.set_epoch(1)
    batch = synthetic_batch(device)

    inference = profile_model(
        model,
        input_shape=(1, 3, INPUT_SIZE, INPUT_SIZE),
        device=device,
        warmup=10,
        iterations=50,
        include_boundary=False,
    )
    model.train()
    for _ in range(WARMUP):
        backward(model, criterion, batch, device)
    torch.cuda.synchronize(device)
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)

    raw = []
    for iteration in range(1, ITERATIONS + 1):
        started = time.perf_counter()
        loss = backward(model, criterion, batch, device)
        torch.cuda.synchronize(device)
        raw.append(
            {
                "method": "unet_gsl",
                "iteration": iteration,
                "step_time_ms": (time.perf_counter() - started) * 1000.0,
                "loss": loss,
            }
        )
    times = [float(row["step_time_ms"]) for row in raw]
    losses = [float(row["loss"]) for row in raw]
    summary = {
        "method": "unet_gsl",
        "parameters": inference.params,
        "inference_flops": inference.flops,
        "inference_flops_g": inference.flops / 1e9 if inference.flops else "",
        "inference_batch_size": 1,
        "inference_latency_ms": inference.latency_ms,
        "inference_fps": inference.fps,
        "inference_peak_memory_mb": inference.peak_memory_mb,
        "training_batch_size": BATCH_SIZE,
        "training_step_mean_ms": statistics.mean(times),
        "training_step_std_ms": statistics.stdev(times),
        "training_step_median_ms": statistics.median(times),
        "training_peak_memory_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
        "loss_mean": statistics.mean(losses),
        "estimated_260_batch_epoch_seconds": statistics.mean(times) * 260 / 1000,
        "weights_updated": False,
        "checkpoints_saved": False,
        "profiling_status": "completed",
        "distance_transform": "SciPy exact Euclidean EDT",
    }
    write_csv(OUT / "raw_gsl_training_step_times.csv", raw)
    write_csv(OUT / "gsl_training_cost_summary.csv", [summary])
    (OUT / "environment.json").write_text(
        json.dumps(
            {
                "generated": "2026-06-09",
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "inference_warmup": 10,
                "inference_iterations": 50,
                "training_warmup": WARMUP,
                "training_iterations": ITERATIONS,
                "input_size": [INPUT_SIZE, INPUT_SIZE],
                "training_batch_size": BATCH_SIZE,
                "optimizer_step": False,
                "weights_saved": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
