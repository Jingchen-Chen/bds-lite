#!/usr/bin/env python3
"""Convert ACDC cardiac MRI data to BDS-Lite NPZ format.

Supports two source layouts:

1. TransUNet pre-sliced H5 (default, --format transunet):
     <slices-dir>/patient001_frame01_slice_0.h5
     <slices-dir>/patient001_frame01_slice_1.h5
     ...
   Each H5 file contains 'image' [H, W] float32 (z-score normalised) and
   'label' [H, W] uint8. Labels: 0=BG, 1=RV, 2=MYO, 3=LV.

2. Official NIfTI layout (--format nifti):
     <src-root>/patient001/patient001_frame01.nii.gz
     <src-root>/patient001/patient001_frame01_gt.nii.gz
     ...

Usage (TransUNet format):
  python3 scripts/convert_acdc.py \\
      --src data/raw/acdc/ACDC/ACDC_training_slices \\
      --output-root data/processed/acdc \\
      --image-size 224x224 \\
      --val-fraction 0.1 \\
      --test-fraction 0.2 \\
      --seed 2026
"""

from __future__ import annotations

import argparse

from bds_lite.data.converters import convert_acdc, convert_acdc_transunet


def parse_size(value: str) -> tuple[int, int]:
    try:
        height, width = value.lower().split("x", maxsplit=1)
        return int(width), int(height)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected size formatted as HxW, e.g. 224x224") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ACDC cardiac MRI to BDS-Lite format.")
    parser.add_argument(
        "--src",
        required=True,
        help=(
            "For --format transunet: flat directory of patient*.h5 slice files. "
            "For --format nifti: directory containing patient001/, patient002/, ... sub-dirs."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="data/processed/acdc",
        help="Destination root for train/val/test NPZ splits.",
    )
    parser.add_argument(
        "--format",
        choices=["transunet", "nifti"],
        default="transunet",
        help="Source data format (default: transunet).",
    )
    parser.add_argument("--image-size", type=parse_size, default=None, help="Resize as HxW.")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of patients used for validation.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of patients used for testing.",
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
    kwargs = dict(
        output_root=args.output_root,
        image_size=args.image_size,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        skip_background_only=args.skip_background_only,
        dry_run=args.dry_run,
    )

    if args.format == "transunet":
        counts = convert_acdc_transunet(slices_dir=args.src, **kwargs)
    else:
        counts = convert_acdc(src_root=args.src, **kwargs)

    total = sum(counts.values())
    for split, count in counts.items():
        print(f"  {split}: {count} slices")
    print(f"Total: {total} slices -> {args.output_root}")


if __name__ == "__main__":
    main()
