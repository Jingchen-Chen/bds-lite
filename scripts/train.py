#!/usr/bin/env python
"""Train a BDS-Lite experiment from a YAML config file."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from bds_lite.data import NpzSegmentationDataset
from bds_lite.training import BDSLiteLoss, build_model
from bds_lite.training.losses import weights_from_config
from bds_lite.training.runner import train_one_epoch, validate_one_epoch
from bds_lite.utils import load_config, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BDS-Lite experiment.")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML file.")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default=None,
        help=(
            "Device to train on. Defaults to trainer.device from config, or `auto`. "
            "Use `cuda` for full baseline runs to avoid accidental CPU fallback."
        ),
    )
    return parser.parse_args()


def configure_logging(run_dir: Path) -> logging.Logger:
    """Send logs to both stdout and a per-run `train.log` file."""
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("bds_lite.train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(run_dir / "train.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def seed_worker(worker_id: int) -> None:
    """Reseed numpy/random in each DataLoader worker for deterministic augmentations."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloaders(
    config: dict,
    seed: int,
    require_boundary: bool,
    require_sdf: bool,
) -> tuple[DataLoader, DataLoader]:
    dataset_cfg = config["dataset"]
    trainer_cfg = config["trainer"]
    data_root = Path(dataset_cfg["processed_dir"])

    train_set = NpzSegmentationDataset(
        data_root,
        split=dataset_cfg.get("train_split", "train"),
        require_boundary=require_boundary,
        require_sdf=require_sdf,
    )
    val_set = NpzSegmentationDataset(
        data_root,
        split=dataset_cfg.get("val_split", "val"),
        require_boundary=False,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)
    num_workers = int(config.get("data", {}).get("num_workers", 4))

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
    return train_loader, val_loader


def save_checkpoint(path: Path, epoch: int, model, optimizer, scheduler, config: dict) -> None:
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)


def _maybe_init_from_checkpoint(
    model: torch.nn.Module,
    trainer_cfg: dict,
    device: torch.device,
    logger: logging.Logger,
) -> None:
    """Load encoder (and optionally decoder) weights from a U-Net checkpoint."""
    init_from = trainer_cfg.get("init_from_checkpoint")
    if not init_from:
        return
    ckpt_path = Path(init_from)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"init_from_checkpoint: {ckpt_path} not found")
    ckpt = torch.load(ckpt_path, map_location=device)
    src_state = ckpt.get("model", ckpt)

    # Copy matching sub-module weights with strict=False so extra keys are ignored.
    dst_state = model.state_dict()
    # Remap U-Net decoder → BDS-Lite seg_decoder if present
    remapped: dict[str, torch.Tensor] = {}
    for k, v in src_state.items():
        remapped[k] = v
        if k.startswith("decoder."):
            remapped["seg_" + k] = v
    loaded, skipped = 0, 0
    for k in dst_state:
        if k in remapped and remapped[k].shape == dst_state[k].shape:
            dst_state[k] = remapped[k]
            loaded += 1
        else:
            skipped += 1
    model.load_state_dict(dst_state)
    logger.info("init_from_checkpoint=%s loaded=%d skipped=%d", ckpt_path, loaded, skipped)


def resolve_device(requested: str | None) -> torch.device:
    device_name = requested or "auto"
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(device_name)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config["project"].get("seed", 2026))
    seed_everything(seed)

    experiment_name = config["experiment"]["name"]
    run_dir = Path(config["project"]["output_dir"]) / "runs" / experiment_name
    checkpoint_dir = Path(config["project"]["output_dir"]) / "checkpoints" / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(run_dir)
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, default=str)

    trainer_cfg = config["trainer"]
    loss_cfg = config.get("loss", {})
    require_boundary = float(loss_cfg.get("boundary_weight", 0.0)) > 0
    require_sdf = (
        float(loss_cfg.get("surface_weight", 0.0)) > 0
        or float(loss_cfg.get("distance_weight", 0.0)) > 0
        or float(loss_cfg.get("far_fp_weight", 0.0)) > 0
    )
    device = resolve_device(args.device or trainer_cfg.get("device"))
    train_loader, val_loader = build_dataloaders(
        config,
        seed=seed,
        require_boundary=require_boundary,
        require_sdf=require_sdf,
    )

    model = build_model(config).to(device)
    _maybe_init_from_checkpoint(model, trainer_cfg, device, logger)
    criterion = BDSLiteLoss(weights_from_config(config))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(trainer_cfg["lr"]),
        weight_decay=float(trainer_cfg.get("weight_decay", 0.0)),
    )

    epochs = int(trainer_cfg["epochs"])
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    if str(trainer_cfg.get("scheduler", "cosine")).lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    grad_clip = trainer_cfg.get("grad_clip", None)
    grad_clip = float(grad_clip) if grad_clip is not None else None
    save_every = int(trainer_cfg.get("save_every", 25))
    num_classes = int(
        config.get("model", {}).get("num_classes", config.get("dataset", {}).get("num_classes", 2))
    )

    logger.info(
        "experiment=%s device=%s epochs=%d batch_size=%d lr=%.2e",
        experiment_name,
        device,
        epochs,
        int(trainer_cfg["batch_size"]),
        float(trainer_cfg["lr"]),
    )

    freeze_encoder_epochs = int(trainer_cfg.get("freeze_encoder_epochs", 0))

    best_dsc = -float("inf")
    metrics_log: list[dict] = []
    for epoch in range(1, epochs + 1):
        criterion.set_epoch(epoch)
        # Two-stage training: freeze encoder for first N epochs.
        if freeze_encoder_epochs > 0 and hasattr(model, "encoder"):
            frozen = epoch <= freeze_encoder_epochs
            for p in model.encoder.parameters():
                p.requires_grad = not frozen
            if epoch == 1:
                logger.info(
                    "freeze_encoder_epochs=%d; encoder frozen for epochs 1-%d",
                    freeze_encoder_epochs,
                    freeze_encoder_epochs,
                )
            if epoch == freeze_encoder_epochs + 1:
                logger.info("unfreezing encoder at epoch=%d", epoch)
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

        epoch_record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        metrics_log.append(epoch_record)
        logger.info("epoch=%d train=%s val=%s", epoch, train_metrics, val_metrics)
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_record) + "\n")

        save_checkpoint(checkpoint_dir / "last.pt", epoch, model, optimizer, scheduler, config)
        val_dsc = val_metrics.get("val_dsc", float("-inf"))
        if val_dsc > best_dsc:
            best_dsc = val_dsc
            save_checkpoint(checkpoint_dir / "best.pt", epoch, model, optimizer, scheduler, config)
            logger.info("new best val_dsc=%.4f at epoch=%d", val_dsc, epoch)

        if save_every > 0 and epoch % save_every == 0:
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch:04d}.pt",
                epoch,
                model,
                optimizer,
                scheduler,
                config,
            )

    logger.info("training complete; best val_dsc=%.4f", best_dsc)


if __name__ == "__main__":
    main()
