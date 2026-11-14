#!/usr/bin/env python
"""Step-3 sanity check: run exactly 10 epochs of a GSL config.

Keeps trainer.epochs=150 in the config so that weights_from_config sees the
correct total_epochs for the alpha schedule, but breaks the training loop after
MAX_EPOCHS epochs. Writes to a __rerun_step3 output subdirectory so the
historical results/runs/<name>/ directory is never touched.

Usage:
    python scripts/train_step3_10ep.py --config <yaml> [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# ── GPU cdist workaround ──────────────────────────────────────────────────────
# The Step-2 integration in losses.py calls self.gsl(seg_logits, mask) without
# the precomputed SDF (batch["sdf"]).  That causes distance_transform_maps_torch
# to run torch.cdist on GPU, which OOMs at 224×224 (needs ~10 GB per class/sample).
# The historical run avoided this by passing batch["sdf"] directly (see stash
# 254d4a1). As a read-only workaround, we redirect the module-level function to
# the scipy CPU path before importing BDSLiteLoss so the module cache picks it up.
import bds_lite.losses.gsl as _gsl_mod


def _cpu_dtm(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    arr = _gsl_mod.distance_transform_maps_numpy(target.cpu().numpy(), num_classes)
    return torch.from_numpy(arr)


_gsl_mod.distance_transform_maps_torch = _cpu_dtm  # redirect before BDSLiteLoss import
# ─────────────────────────────────────────────────────────────────────────────

from bds_lite.data import NpzSegmentationDataset  # noqa: E402
from bds_lite.training import BDSLiteLoss, build_model  # noqa: E402
from bds_lite.training.losses import weights_from_config  # noqa: E402
from bds_lite.training.runner import train_one_epoch, validate_one_epoch  # noqa: E402
from bds_lite.utils import load_config, seed_everything  # noqa: E402

MAX_EPOCHS = 10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None)
    return p.parse_args()


def configure_logging(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("bds_lite.train_step3")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(run_dir / "train.log")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_device(requested: str | None) -> torch.device:
    name = requested or "auto"
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config["project"].get("seed", 2026))
    seed_everything(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    experiment_name = config["experiment"]["name"]
    # Write to a fresh __rerun_step3 subdirectory; never touch the historical dir.
    run_dir = Path(config["project"]["output_dir"]) / "runs" / (experiment_name + "__rerun_step3")
    checkpoint_dir = (
        Path(config["project"]["output_dir"]) / "checkpoints" / (experiment_name + "__rerun_step3")
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger = configure_logging(run_dir)
    with (run_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2, default=str)

    trainer_cfg = config["trainer"]
    device = resolve_device(args.device or trainer_cfg.get("device"))

    dataset_cfg = config["dataset"]
    data_root = Path(dataset_cfg["processed_dir"])
    num_workers = int(config.get("data", {}).get("num_workers", 4))

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_set = NpzSegmentationDataset(data_root, split=dataset_cfg.get("train_split", "train"))
    val_set = NpzSegmentationDataset(data_root, split=dataset_cfg.get("val_split", "val"))
    train_loader = DataLoader(
        train_set,
        batch_size=int(trainer_cfg["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(trainer_cfg["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

    model = build_model(config).to(device)
    # weights_from_config sees trainer.epochs=150 → correct total_epochs for alpha schedule
    criterion = BDSLiteLoss(weights_from_config(config))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(trainer_cfg["lr"]),
        weight_decay=float(trainer_cfg.get("weight_decay", 0.0)),
    )
    # Cosine scheduler with T_max = full 150 epochs so LR matches historical trajectory
    full_epochs = int(trainer_cfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=full_epochs)

    grad_clip = trainer_cfg.get("grad_clip", None)
    grad_clip = float(grad_clip) if grad_clip is not None else None
    num_classes = int(config.get("model", {}).get("num_classes", dataset_cfg.get("num_classes", 2)))

    logger.info(
        "step3_sanity_check experiment=%s device=%s total_epochs=%d max_epochs=%d "
        "batch_size=%d lr=%.2e seed=%d deterministic=True",
        experiment_name,
        device,
        full_epochs,
        MAX_EPOCHS,
        int(trainer_cfg["batch_size"]),
        float(trainer_cfg["lr"]),
        seed,
    )

    t0 = time.time()
    best_dsc = -float("inf")
    for epoch in range(1, MAX_EPOCHS + 1):
        if criterion.gsl is not None:
            criterion.gsl.set_epoch(epoch)

        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            amp=bool(trainer_cfg.get("amp", True)),
            grad_clip=grad_clip,
            scheduler=scheduler,
        )
        val_metrics = validate_one_epoch(
            model, val_loader, criterion, device, num_classes=num_classes
        )

        logger.info("epoch=%d train=%s val=%s", epoch, train_metrics, val_metrics)
        with (run_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"epoch": epoch, "train": train_metrics, "val": val_metrics}) + "\n")

        val_dsc = val_metrics.get("val_dsc", float("-inf"))
        if val_dsc > best_dsc:
            best_dsc = val_dsc
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "config": config},
                checkpoint_dir / "best.pt",
            )
            logger.info("new best val_dsc=%.4f at epoch=%d", val_dsc, epoch)

    elapsed = (time.time() - t0) / 60.0
    logger.info("done epochs=%d wall_clock=%.1f min", MAX_EPOCHS, elapsed)


if __name__ == "__main__":
    main()
