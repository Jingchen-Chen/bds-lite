#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from bds_lite.data.boundary import mask_to_boundary, mask_to_sdf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate boundary and SDF targets for .npz masks."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name under data/processed.")
    parser.add_argument("--split", default="train", help="Split name, e.g. train/val/test.")
    parser.add_argument("--root", default="data/processed", help="Processed data root.")
    parser.add_argument("--radius", type=int, default=1, help="Boundary radius in pixels.")
    parser.add_argument("--num-classes", type=int, default=None, help="Class count for SDF maps.")
    parser.add_argument("--no-sdf", action="store_true", help="Skip signed-distance maps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_dir = Path(args.root) / args.dataset / args.split
    files = sorted(split_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no .npz files found under {split_dir}")

    for path in tqdm(files, desc=f"{args.dataset}/{args.split}"):
        with np.load(path) as sample:
            arrays = {key: sample[key] for key in sample.files}
        mask = arrays["mask"]
        arrays["boundary"] = mask_to_boundary(mask, radius=args.radius)[None, ...].astype(np.uint8)
        if not args.no_sdf:
            arrays["sdf"] = mask_to_sdf(mask, num_classes=args.num_classes).astype(np.float32)
        np.savez_compressed(path, **arrays)


if __name__ == "__main__":
    main()
