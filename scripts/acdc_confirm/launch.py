"""Training entry point for both arms: the frozen scripts/kd_mc/train_mc.py, unchanged.

The frozen trainer admits student seeds 1-3 and lambda_kd = 1.0 only (scripts/kd_mc/common.py).
After the strict check in scripts/acdc_confirm/common.validate, this launcher widens exactly two
module constants of scripts.kd_mc.common at run time, for this process only:
- STUDENT_SEEDS -> (this run's seed,)
- KD_SPEC["lambda_kd"] -> this run's lambda_kd (1.0 for D+KD-band, 0.0 for decoder-only)
and then calls scripts.kd_mc.train_mc.main() with the same command line. No file is modified.
With lambda_kd = 0, KDCriterionMC skips L_kd entirely (`if self.lambda_kd and ...`), which is the
decoder-only objective; the trainer builds no test dataset and stops on revision/STOP_KD_TRAINING.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.acdc_confirm.common import SEEDS, validate  # noqa: E402


def admit(config: dict, allowed_seeds=SEEDS, check_protocol_file: bool = True) -> None:
    from scripts.kd_mc import common as kd_common

    validate(config, allowed_seeds, check_protocol_file)
    kd_common.STUDENT_SEEDS = (config["project"]["seed"],)
    kd_common.KD_SPEC = dict(kd_common.KD_SPEC, lambda_kd=config["kd"]["lambda_kd"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args, _ = parser.parse_known_args()
    admit(yaml.safe_load(args.config.read_text()))
    from scripts.kd_mc import train_mc

    train_mc.main()


if __name__ == "__main__":
    main()
