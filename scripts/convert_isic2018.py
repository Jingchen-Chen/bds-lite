#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from bds_lite.data.converters import convert_isic2018


def parse_size(value: str) -> tuple[int, int]:
    try:
        height, width = value.lower().split("x", maxsplit=1)
        return int(width), int(height)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected size formatted as HxW, e.g. 224x224") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ISIC2018 Task 1 images to .npz samples.")
    parser.add_argument("--image-dir", required=True, help="Directory containing ISIC image files.")
    parser.add_argument("--mask-dir", required=True, help="Directory containing lesion masks.")
    parser.add_argument(
        "--output-root",
        default="data/processed/isic2018",
        help="Output root for processed split folders.",
    )
    parser.add_argument("--split", default="train", help="Output split name when not splitting.")
    parser.add_argument("--image-size", type=parse_size, default=None, help="Resize as HxW.")
    parser.add_argument("--val-fraction", type=float, default=0.0, help="Deterministic val split.")
    parser.add_argument("--seed", type=int, default=2026, help="Seed used for train/val split.")
    parser.add_argument("--limit", type=int, default=None, help="Convert only the first N images.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned conversions only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    converted = convert_isic2018(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        output_root=args.output_root,
        split=args.split,
        image_size=args.image_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        limit=args.limit,
        dry_run=args.dry_run,
    )

    print(f"Prepared {len(converted)} ISIC2018 samples for {Path(args.output_root)}")
    if args.dry_run:
        for sample in converted[:10]:
            print(f"{sample.image_path} + {sample.mask_path} -> {sample.output_path}")
        if len(converted) > 10:
            print(f"... {len(converted) - 10} more")


if __name__ == "__main__":
    main()
