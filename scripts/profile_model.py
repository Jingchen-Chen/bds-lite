#!/usr/bin/env python
"""Profile model size and inference cost for an experiment config."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from bds_lite.evaluation import profile_model
from bds_lite.training import build_model
from bds_lite.utils import load_config, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile a BDS-Lite experiment model.")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML file.")
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional CSV output path. Defaults to "
            "results/tables/resource_profile_<experiment>.csv."
        ),
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
    parser.add_argument(
        "--include-boundary",
        action="store_true",
        help="Force the BDS-Lite boundary branch on to profile retained-branch inference.",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(device_name)


def write_csv(row: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "model",
        "dataset",
        "device",
        "batch_size",
        "input_shape",
        "boundary_branch",
        "params",
        "trainable_params",
        "flops",
        "fps",
        "latency_ms",
        "peak_memory_mb",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["project"].get("seed", 2026)))

    image_size = tuple(int(value) for value in config.get("data", {}).get("image_size", [224, 224]))
    in_channels = int(
        config.get("model", {}).get(
            "in_channels",
            config.get("dataset", {}).get("in_channels", 1),
        )
    )
    input_shape = (args.batch_size, in_channels, image_size[0], image_size[1])
    device = resolve_device(args.device)

    model = build_model(config)
    profile = profile_model(
        model,
        input_shape=input_shape,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
        include_boundary=args.include_boundary,
    )
    profile_row = profile.to_dict()
    profile_row.update(
        {
            "experiment": config["experiment"]["name"],
            "model": config.get("model", {}).get("name", "unet"),
            "dataset": config.get("dataset", {}).get("name", "unknown"),
            "input_shape": "x".join(str(value) for value in profile.input_shape),
        }
    )

    output_path = (
        Path(args.output)
        if args.output
        else Path(config["project"]["output_dir"])
        / "tables"
        / f"resource_profile_{config['experiment']['name']}.csv"
    )
    write_csv(profile_row, output_path)
    print(json.dumps(profile_row, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
