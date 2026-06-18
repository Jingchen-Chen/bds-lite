#!/usr/bin/env python
"""Run the tracked trainer with the isolated memory-bounded GSL DTM."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

from gsl_memory_bounded import install_memory_bounded_gsl

ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    install_memory_bounded_gsl()
    os.chdir(ROOT)
    sys.argv = [
        str(ROOT / "scripts/train.py"),
        "--config",
        args.config,
        "--device",
        args.device,
    ]
    runpy.run_path(str(ROOT / "scripts/train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
