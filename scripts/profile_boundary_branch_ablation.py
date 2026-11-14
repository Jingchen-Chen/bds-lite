#!/usr/bin/env python
"""Profile boundary-branch disabled vs retained inference cost."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from bds_lite.evaluation import profile_model
from bds_lite.training import build_model
from bds_lite.utils import load_config, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile U-Net and BDS-Lite boundary-branch inference ablation."
    )
    parser.add_argument(
        "--baseline-config",
        default="configs/experiments/isic_unet_baseline.yaml",
        help="Reference baseline experiment config.",
    )
    parser.add_argument(
        "--candidate-config",
        default="configs/experiments/isic_bdslite_boundary_gate_damped_surface.yaml",
        help="BDS-Lite experiment config.",
    )
    parser.add_argument(
        "--output",
        default="results/tables/resource_profile_isic2018_boundary_branch_ablation.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device used for timing and peak-memory profiling.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Profiling batch size.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations.")
    parser.add_argument("--iterations", type=int, default=50, help="Measured iterations.")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(device_name)


def input_shape_from_config(config: dict[str, Any], batch_size: int) -> tuple[int, int, int, int]:
    image_size = tuple(int(value) for value in config.get("data", {}).get("image_size", [224, 224]))
    in_channels = int(
        config.get("model", {}).get(
            "in_channels",
            config.get("dataset", {}).get("in_channels", 1),
        )
    )
    return (batch_size, in_channels, image_size[0], image_size[1])


def count_module_parameters(module: nn.Module | None) -> int:
    if module is None:
        return 0
    return sum(parameter.numel() for parameter in module.parameters())


def removable_boundary_params(model: nn.Module) -> int:
    return count_module_parameters(getattr(model, "boundary_decoder", None))


def format_shape(shape: tuple[int, int, int, int]) -> str:
    return "x".join(str(value) for value in shape)


def percent_delta(value: float | int | None, reference: float | int | None) -> float | None:
    if value is None or reference is None or reference == 0:
        return None
    return (float(value) - float(reference)) * 100.0 / float(reference)


def profile_row(
    *,
    variant: str,
    config: dict[str, Any],
    model: nn.Module,
    input_shape: tuple[int, int, int, int],
    device: torch.device,
    warmup: int,
    iterations: int,
    include_boundary: bool,
    baseline_flops: int | None = None,
    baseline_active_params: int | None = None,
    disabled_latency_ms: float | None = None,
) -> dict[str, Any]:
    profile = profile_model(
        model,
        input_shape=input_shape,
        device=device,
        warmup=warmup,
        iterations=iterations,
        include_boundary=include_boundary,
    )
    removed_params = 0 if include_boundary else removable_boundary_params(model)
    active_params = profile.params - removed_params
    return {
        "variant": variant,
        "experiment": config["experiment"]["name"],
        "model": config.get("model", {}).get("name", "unet"),
        "dataset": config.get("dataset", {}).get("name", "unknown"),
        "device": profile.device,
        "batch_size": profile.batch_size,
        "input_shape": format_shape(profile.input_shape),
        "boundary_branch": profile.boundary_branch,
        "total_params": profile.params,
        "active_params": active_params,
        "removed_boundary_params": removed_params,
        "flops": profile.flops,
        "fps": profile.fps,
        "latency_ms": profile.latency_ms,
        "peak_memory_mb": profile.peak_memory_mb,
        "active_params_over_baseline_pct": percent_delta(active_params, baseline_active_params),
        "flops_over_baseline_pct": percent_delta(profile.flops, baseline_flops),
        "latency_over_disabled_pct": percent_delta(profile.latency_ms, disabled_latency_ms),
        "notes": (
            "main deploy path; boundary decoder disabled"
            if variant == "bdslite_boundary_disabled"
            else "upper-bound retained boundary branch"
            if variant == "bdslite_boundary_retained"
            else "reference baseline"
        ),
    }


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "experiment",
        "model",
        "dataset",
        "device",
        "batch_size",
        "input_shape",
        "boundary_branch",
        "total_params",
        "active_params",
        "removed_boundary_params",
        "flops",
        "fps",
        "latency_ms",
        "peak_memory_mb",
        "active_params_over_baseline_pct",
        "flops_over_baseline_pct",
        "latency_over_disabled_pct",
        "notes",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    baseline_config = load_config(args.baseline_config)
    candidate_config = load_config(args.candidate_config)
    seed_everything(int(candidate_config["project"].get("seed", 2026)))
    device = resolve_device(args.device)

    baseline_shape = input_shape_from_config(baseline_config, args.batch_size)
    candidate_shape = input_shape_from_config(candidate_config, args.batch_size)
    if baseline_shape != candidate_shape:
        raise ValueError(
            f"baseline and candidate input shapes differ: {baseline_shape} vs {candidate_shape}"
        )

    baseline_model = build_model(baseline_config)
    candidate_disabled_model = build_model(candidate_config)
    candidate_retained_model = build_model(candidate_config)

    baseline = profile_row(
        variant="unet_baseline",
        config=baseline_config,
        model=baseline_model,
        input_shape=baseline_shape,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
        include_boundary=False,
    )
    disabled = profile_row(
        variant="bdslite_boundary_disabled",
        config=candidate_config,
        model=candidate_disabled_model,
        input_shape=candidate_shape,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
        include_boundary=False,
        baseline_flops=baseline["flops"],
        baseline_active_params=baseline["active_params"],
    )
    retained = profile_row(
        variant="bdslite_boundary_retained",
        config=candidate_config,
        model=candidate_retained_model,
        input_shape=candidate_shape,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
        include_boundary=True,
        baseline_flops=baseline["flops"],
        baseline_active_params=baseline["active_params"],
        disabled_latency_ms=disabled["latency_ms"],
    )

    rows = [baseline, disabled, retained]
    output_path = Path(args.output)
    write_csv(rows, output_path)
    print(json.dumps(rows, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
