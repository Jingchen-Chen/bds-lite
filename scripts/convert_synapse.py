#!/usr/bin/env python3
"""Convert Synapse multi-organ CT data (HiFormer layout) to BDS-Lite NPZ format.

Expected source layout:
  <src-root>/train_npz/*.npz   -- pre-sliced 2D training slices
  <src-root>/test_vol_h5/*.h5  -- 3D test volumes

Usage example:
  python3 scripts/convert_synapse.py \\
      --src /path/to/HiFormer/data/Synapse \\
      --output-root data/processed/synapse \\
      --image-size 224x224 \\
      --val-fraction 0.2 \\
      --seed 2026
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bds_lite.data.converters import convert_synapse


def parse_size(value: str) -> tuple[int, int]:
    try:
        height, width = value.lower().split("x", maxsplit=1)
        return int(width), int(height)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected size formatted as HxW, e.g. 224x224") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Synapse CT data to BDS-Lite format.")
    parser.add_argument(
        "--src",
        required=True,
        help="Root directory containing train_npz/ and test_vol_h5/ sub-directories.",
    )
    parser.add_argument(
        "--output-root",
        default="data/processed/synapse",
        help="Destination root for train/val/test NPZ splits.",
    )
    parser.add_argument("--image-size", type=parse_size, default=None, help="Resize as HxW.")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of training cases held out for validation.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--skip-background-only",
        action="store_true",
        help="Discard slices where the entire label map is background (class 0).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned conversions only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    counts = convert_synapse(
        train_npz_dir=src / "train_npz",
        test_h5_dir=src / "test_vol_h5",
        output_root=args.output_root,
        image_size=args.image_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        skip_background_only=args.skip_background_only,
        dry_run=args.dry_run,
    )
    total = sum(counts.values())
    for split, count in counts.items():
        print(f"  {split}: {count} slices")
    print(f"Total: {total} slices -> {args.output_root}")


if __name__ == "__main__":
    main()
