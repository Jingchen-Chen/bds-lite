"""WP1h deployment resource profile (F8): params, FLOPs, GPU/CPU latency, kernel counts.

Run only while no training job uses the GPU (checked). Models are randomly initialised
deployment graphs (weights do not affect latency/FLOPs); seed fixed. Deployment graph
of BDS-Lite = segmentation path + projection P + gate (auxiliary boundary decoder not
executed). "BDS-Lite gate-collapsed" removes P and the gate (U-Net-identical graph).
GPU latency: CUDA events, fp32 and AMP fp16 autocast, batch 1/8/16, 50 warm-up and 200
timed forwards, median and IQR per batch. CPU latency: batch 1, 1 thread and all
physical threads, 10 warm-up and 50 timed forwards. Kernel counts: torch.profiler CUDA
kernel launches for one batch-1 forward. Output: revision/tables/resource_profile.csv
and revision/validation/resource_profile_env.json.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.flop_counter import FlopCounterMode

from bds_lite.evaluation.profiling import estimate_flops
from bds_lite.models import AttentionUNet, BDSLiteUNet, UNet
from bds_lite.training.runner import inference_forward
from scripts.comparators.components import LBUNetK

ROOT = Path(__file__).resolve().parents[2]


class GateCollapsed(torch.nn.Module):
    def __init__(self, model: BDSLiteUNet) -> None:
        super().__init__()
        self.encoder, self.decoder = model.encoder, model.seg_decoder

    def forward(self, x):
        logits, _ = self.decoder(self.encoder(x))
        return {"seg_logits": logits}


def models(channels: int, classes: int) -> dict[str, torch.nn.Module]:
    torch.manual_seed(0)
    bds = BDSLiteUNet(channels, classes, 32, 1, True, 0.25)
    return {
        "U-Net": UNet(channels, classes, 32),
        "BDS-Lite (deployed)": bds,
        "BDS-Lite (gate-collapsed)": GateCollapsed(bds),
        "Attention U-Net": AttentionUNet(channels, classes, 32),
        "LB-UNet": LBUNetK(channels, classes),
    }


def deploy_params(name: str, model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    if name == "BDS-Lite (deployed)":
        aux = sum(p.numel() for p in model.boundary_decoder.parameters())
        return total, total - aux
    return total, total


def flops(model, x) -> float:
    with FlopCounterMode(display=False) as counter:
        inference_forward(model, x)
    return counter.get_total_flops() / 1e9


def gpu_latency(model, batch, amp, channels) -> tuple[float, float, float]:
    x = torch.randn(batch, channels, 224, 224, device="cuda")
    times = []
    with torch.inference_mode(), torch.autocast("cuda", enabled=amp):
        for _ in range(50):
            inference_forward(model, x)
        torch.cuda.synchronize()
        for _ in range(200):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            inference_forward(model, x)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    q1, med, q3 = np.percentile(times, [25, 50, 75])
    return float(med), float(q1), float(q3)


def cpu_latency(model, threads, channels) -> float:
    torch.set_num_threads(threads)
    x = torch.randn(1, channels, 224, 224)
    with torch.inference_mode():
        for _ in range(10):
            inference_forward(model, x)
        times = []
        for _ in range(50):
            began = time.perf_counter()
            inference_forward(model, x)
            times.append((time.perf_counter() - began) * 1000)
    return float(np.median(times))


def kernel_count(model, channels) -> tuple[int, float]:
    x = torch.randn(1, channels, 224, 224, device="cuda")
    with torch.inference_mode():
        for _ in range(10):
            inference_forward(model, x)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            inference_forward(model, x)
            torch.cuda.synchronize()
    events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    return len(events), float(sum(e.device_time for e in events) / 1000.0)


def main() -> None:
    busy = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if busy:
        raise RuntimeError(f"GPU busy ({busy}); profiling would be contaminated")
    torch.backends.cudnn.benchmark = False
    rows = []
    physical = len(os.sched_getaffinity(0))
    for dataset, channels, classes in [("isic2018", 3, 2), ("synapse", 1, 9)]:
        for name, model in models(channels, classes).items():
            model.eval()
            total, deployed = deploy_params(name, model)
            row = dict(
                dataset=dataset,
                model=name,
                params_total=total,
                params_deployed=deployed,
                gflops_b1=flops(model, torch.randn(1, channels, 224, 224)),
                # released hook counter (submitted Table 4); counts ConvTranspose2d by
                # output elements, hence larger than FlopCounterMode
                gflops_b1_released_hook=(
                    estimate_flops(model, torch.randn(1, channels, 224, 224)) or 0
                )
                / 1e9,
                cpu_ms_b1_1thread=cpu_latency(model, 1, channels),
                cpu_ms_b1_allthreads=cpu_latency(model, physical, channels),
            )
            model.cuda()
            for amp in (False, True):
                for batch in (1, 8, 16):
                    med, q1, q3 = gpu_latency(model, batch, amp, channels)
                    tag = f"gpu_{'amp' if amp else 'fp32'}_b{batch}"
                    row[f"{tag}_ms_median"], row[f"{tag}_ms_iqr"] = med, q3 - q1
            row["cuda_kernels_b1"], row["cuda_kernel_time_ms_b1"] = kernel_count(model, channels)
            model.cpu()
            rows.append(row)
            print(json.dumps(row), flush=True)
    out = ROOT / "revision/tables/resource_profile.csv"
    with out.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    env = dict(
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),
        cpu=platform.processor() or platform.machine(),
        cpu_model=subprocess.check_output(
            ["bash", "-c", "lscpu | grep 'Model name'"], text=True
        ).strip(),
        threads_all=physical,
        nvidia_smi=subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,temperature.gpu,clocks.sm,power.limit",
                "--format=csv",
            ],
            text=True,
        ),
    )
    (ROOT / "revision/validation/resource_profile_env.json").write_text(
        json.dumps(env, indent=1) + "\n"
    )


if __name__ == "__main__":
    main()
