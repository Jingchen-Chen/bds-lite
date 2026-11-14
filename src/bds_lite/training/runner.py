"""Training and validation loops shared by the experiment scripts."""

from __future__ import annotations

from collections.abc import Iterable
from inspect import signature

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from bds_lite.evaluation.metrics import dice_score, iou_score


def move_batch_to_device(
    batch: dict[str, torch.Tensor | list[str] | str],
    device: torch.device,
) -> dict:
    """Move every tensor in `batch` to `device`; leave non-tensors untouched."""
    output = {}
    for key, value in batch.items():
        output[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return output


def inference_forward(model: nn.Module, image: torch.Tensor) -> dict[str, torch.Tensor]:
    """Run the model in inference mode, disabling the boundary branch when supported."""
    if "return_boundary" in signature(model.forward).parameters:
        return model(image, return_boundary=False)
    return model(image)


def _accumulate_segmentation_metrics(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    accumulator: dict[str, float],
    counter: dict[str, int],
) -> None:
    """Accumulate per-sample mean foreground Dice and IoU into `accumulator`."""
    pred = pred_logits.argmax(dim=1).detach().cpu().numpy()
    truth = target.detach().cpu().numpy()
    foreground_labels = list(range(1, num_classes)) if num_classes > 1 else [1]

    for sample_idx in range(pred.shape[0]):
        per_class_dice = [
            dice_score(pred[sample_idx], truth[sample_idx], label) for label in foreground_labels
        ]
        per_class_iou = [
            iou_score(pred[sample_idx], truth[sample_idx], label) for label in foreground_labels
        ]
        accumulator["val_dsc"] = accumulator.get("val_dsc", 0.0) + float(np.mean(per_class_dice))
        accumulator["val_iou"] = accumulator.get("val_iou", 0.0) + float(np.mean(per_class_iou))
        counter["samples"] = counter.get("samples", 0) + 1


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[dict],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: bool = True,
    grad_clip: float | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict[str, float]:
    """Train for one epoch and return averaged loss scalars."""
    model.train()
    amp_enabled = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    totals: dict[str, float] = {}
    count = 0

    for batch in tqdm(loader, desc="train", leave=False):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(batch["image"])
            loss, scalars = criterion(outputs, batch)

        scaler.scale(loss).backward()
        if grad_clip is not None and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()

        count += 1
        for key, value in scalars.items():
            totals[key] = totals.get(key, 0.0) + value

    if scheduler is not None:
        scheduler.step()

    averaged = {key: value / max(count, 1) for key, value in totals.items()}
    averaged["lr"] = float(optimizer.param_groups[0]["lr"])
    return averaged


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: Iterable[dict],
    criterion: nn.Module,
    device: torch.device,
    num_classes: int | None = None,
) -> dict[str, float]:
    """Evaluate one epoch; report loss scalars plus mean Dice/IoU when `num_classes` is set."""
    model.eval()
    totals: dict[str, float] = {}
    metric_totals: dict[str, float] = {}
    metric_counter: dict[str, int] = {}
    count = 0

    for batch in tqdm(loader, desc="val", leave=False):
        batch = move_batch_to_device(batch, device)
        outputs = inference_forward(model, batch["image"])
        loss, scalars = criterion(outputs, batch)
        scalars["loss"] = float(loss.detach().cpu())
        count += 1
        for key, value in scalars.items():
            totals[key] = totals.get(key, 0.0) + value

        if num_classes is not None:
            _accumulate_segmentation_metrics(
                outputs["seg_logits"],
                batch["mask"],
                num_classes,
                metric_totals,
                metric_counter,
            )

    averaged = {key: value / max(count, 1) for key, value in totals.items()}
    sample_count = metric_counter.get("samples", 0)
    if sample_count > 0:
        for key, value in metric_totals.items():
            averaged[key] = value / sample_count
    return averaged
