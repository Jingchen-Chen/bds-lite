"""Training entry point for every follow-up arm: the frozen scripts/kd_mc/train_mc.py, unchanged.

As in scripts/acdc_confirm/launch.py, the strict check (scripts/kd_followup/common.validate) runs
first, and then module-level names are rebound for this process only; no file is modified.
Always:
- scripts.kd_mc.common.STUDENT_SEEDS -> (this run's seed,); KD_SPEC["lambda_kd"] -> this run's;
- train_mc.require_protocol -> the frozen require_protocol applied to the Stage 3 form of the
  config (kd.mode "band", kd.cache_run the external teacher's), so every frozen check of the
  model, loss, trainer, data and split blocks still runs on the real config.
Per arm (registration section 3):
- kd_full: scripts.kd_mc.loss.class_band -> all-ones, so L_kd averages the same soft
  cross-entropy over every pixel of the batch (the normalisation sum(B) becomes N*H*W);
- kd_band_peer: train_mc.verified_cache -> the compact teacher's cache record
  (results/revision_runs/acdc_followup_e2_teacher_cache), with the same byte check;
- decoder_aug / kd_band_aug: train_mc.TeacherCacheDatasetMC -> a module-level subclass (picklable
  for the forkserver DataLoader workers of Python 3.14) that applies one
  uniformly drawn D4 transform (rot90 k, optional horizontal flip) to image, mask, stored
  boundary target and cached teacher probabilities alike, drawn from the DataLoader worker's
  torch RNG (seeded by the loader generator, so reproducible per seed and new each epoch).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.kd.common import sha  # noqa: E402
from scripts.kd_followup.common import (  # noqa: E402
    ARMS,
    ORIGINAL_CACHE_RUN,
    PEER_CACHE_RUN,
    RUNS,
    validate,
)
from scripts.kd_mc.data import TeacherCacheDatasetMC  # noqa: E402


def full_region(teacher_prob: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """KD-full: every pixel is in the region (same (N, H, W) float layout as class_band)."""
    n, _, h, w = teacher_prob.shape
    return torch.ones((n, h, w), dtype=torch.float32, device=teacher_prob.device)


def d4(value: torch.Tensor, code: int) -> torch.Tensor:
    value = torch.rot90(value, code % 4, dims=(-2, -1))
    if code >= 4:
        value = torch.flip(value, dims=(-1,))
    return value.contiguous()


class AugmentedTeacherCacheDatasetMC(TeacherCacheDatasetMC):
    """Module-level (picklable for forkserver DataLoader workers); one D4 draw per sample."""

    def __getitem__(self, index):
        item = super().__getitem__(index)
        code = int(torch.randint(8, ()))
        for key in ("image", "mask", "boundary", "teacher_prob"):
            if key in item:
                item[key] = d4(item[key], code)
        return item


def peer_cache(dataset: str) -> tuple[Path, str]:
    """Same checks as train_mc.verified_cache, for the compact teacher's cache."""
    from scripts.kd_mc.data import CACHE_NAME

    folder = RUNS / PEER_CACHE_RUN
    marker = folder / "cache_complete.json"
    if dataset != "acdc" or not marker.is_file():
        raise FileNotFoundError("compact-teacher cache step has not completed")
    record = json.loads(marker.read_text())
    if record["gate"]["status"] != "NOT_APPLICABLE" or not record["check"]["passed"]:
        raise RuntimeError("compact-teacher cache record is not the registered one")
    digest = sha(folder / CACHE_NAME)
    if digest != record["cache_sha256"]:
        raise ValueError("compact-teacher cache bytes differ from the completed cache record")
    return folder, digest


def admit(config: dict, check_registration_file: bool = True):
    """Validate, rebind, and return the train_mc module ready to run."""
    from scripts.kd_mc import common as kd_common
    from scripts.kd_mc import loss as kd_loss
    from scripts.kd_mc import train_mc

    arm = validate(config, check_registration_file)
    _, region, cache, augmentation = ARMS[arm]
    kd_common.STUDENT_SEEDS = (config["project"]["seed"],)
    kd_common.KD_SPEC = dict(kd_common.KD_SPEC, lambda_kd=config["kd"]["lambda_kd"])
    frozen_check = kd_common.require_protocol

    def require_protocol(resolved: dict) -> None:
        stage3_form = copy.deepcopy(resolved)
        stage3_form["kd"]["mode"] = "band"
        stage3_form["kd"]["cache_run"] = ORIGINAL_CACHE_RUN
        frozen_check(stage3_form)

    train_mc.require_protocol = require_protocol
    if region == "full":
        kd_loss.class_band = full_region
    if cache == PEER_CACHE_RUN:
        train_mc.verified_cache = peer_cache
    if augmentation == "uniform_D4":
        train_mc.TeacherCacheDatasetMC = AugmentedTeacherCacheDatasetMC
    elif augmentation is not None:
        raise ValueError(augmentation)
    return train_mc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args, _ = parser.parse_known_args()
    admit(yaml.safe_load(args.config.read_text())).main()


if __name__ == "__main__":
    main()
