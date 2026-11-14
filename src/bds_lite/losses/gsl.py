"""Generalized Surface Loss.

Ported from Celaya, A., Riviere, B., and Fuentes, D. "A Generalized Surface
Loss for Reducing the Hausdorff Distance in Medical Imaging Segmentation."
arXiv:2302.03868v3 (2024), reference implementation
https://github.com/aecelaya/gen-surf-loss at commit
2115ccabe289b223f52475482ae70b273a9ac809.

The paper combines Dice-CE region loss with the GSL distance-transform term and
a decreasing alpha schedule. The default here is the step schedule with step
length 5 epochs, matching the paper's best LiTS setting.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy import ndimage as ndi
from torch import nn
from torch.nn import functional as F


class GeneralizedSurfaceLoss(nn.Module):
    """Dice-CE plus GSL boundary term for 2D multi-class segmentation."""

    def __init__(
        self,
        num_classes: int,
        total_epochs: int,
        alpha_schedule: str = "step5",
        class_weights: tuple[float, ...] | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        self.num_classes = int(num_classes)
        self.total_epochs = int(total_epochs)
        self.alpha_schedule = str(alpha_schedule)
        self.eps = float(eps)
        self.current_epoch = 1
        if class_weights is not None:
            weights = tuple(float(value) for value in class_weights)
            if len(weights) == num_classes - 1:
                weights = (1.0, *weights)
            if len(weights) != num_classes:
                raise ValueError("class_weights must match num_classes or foreground classes")
            self.register_buffer("class_weights", torch.tensor(weights, dtype=torch.float32))
        else:
            self.register_buffer("class_weights", None)

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        dtm: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if logits.ndim != 4:
            raise ValueError(f"expected logits with shape [N,C,H,W], got {tuple(logits.shape)}")
        if logits.shape[1] != self.num_classes:
            raise ValueError(f"expected {self.num_classes} channels, got {logits.shape[1]}")

        region = dice_ce_region_loss(logits, target, class_weights=self._class_weights(logits))
        if dtm is None:
            dtm = distance_transform_maps_torch(target, self.num_classes).to(logits.device)
        else:
            dtm = _coerce_dtm(dtm, self.num_classes, logits.device, logits.dtype)

        boundary = generalized_surface_term(
            logits,
            target,
            dtm,
            class_weights=self._class_weights(logits),
            eps=self.eps,
        )
        alpha = logits.new_tensor(
            alpha_value(self.current_epoch, self.total_epochs, self.alpha_schedule)
        )
        total = alpha * region + (1.0 - alpha) * boundary
        return total, {"gsl_region": region, "gsl_boundary": boundary, "gsl_alpha": alpha}

    def _class_weights(self, logits: torch.Tensor) -> torch.Tensor | None:
        return self.class_weights.to(logits.device) if self.class_weights is not None else None


def dice_ce_region_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    target = target.long()
    ce = F.cross_entropy(logits, target, weight=class_weights)
    target_oh = F.one_hot(target, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
    probs = torch.softmax(logits, dim=1)
    axes = (2, 3)
    numerator = torch.sum((target_oh - probs).square(), dim=axes)
    denominator = torch.sum(target_oh.square(), dim=axes) + torch.sum(probs.square(), dim=axes)
    dice = (numerator / denominator.clamp_min(eps)).mean(dim=1).mean()
    return ce + dice


def generalized_surface_term(
    logits: torch.Tensor,
    target: torch.Tensor,
    dtm: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    target_oh = F.one_hot(target.long(), num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
    probs = torch.softmax(logits, dim=1)
    if dtm.shape != probs.shape:
        raise ValueError(
            f"dtm shape {tuple(dtm.shape)} does not match logits {tuple(logits.shape)}"
        )

    if class_weights is None:
        class_weights = 1.0 / (torch.sum(target_oh, dim=(2, 3)).square() + 1.0)
    else:
        class_weights = class_weights[None].expand(target.shape[0], -1)

    y_worst = (1.0 - target_oh).square()
    numerator = torch.sum((dtm * (y_worst - probs)).square(), dim=(2, 3)) * class_weights
    denominator = torch.sum((dtm * (y_worst - target_oh)).square(), dim=(2, 3)) * class_weights
    boundary_loss = torch.sum(numerator, dim=1) / torch.sum(denominator, dim=1).clamp_min(eps)
    return 1.0 - boundary_loss.mean()


def alpha_value(epoch: int, total_epochs: int, schedule: str) -> float:
    if total_epochs <= 1:
        return 0.0
    epoch0 = max(0, int(epoch) - 1)
    schedule = str(schedule)
    if schedule.startswith("step"):
        step_length = int(schedule.removeprefix("step") or 5)
        return _step_alpha(epoch0, total_epochs, step_length)
    if schedule == "linear":
        return max(0.0, 1.0 - epoch0 / float(total_epochs - 1))
    if schedule == "cosine":
        return float((1.0 + math.cos(math.pi * epoch0 / float(total_epochs - 1))) / 2.0)
    raise ValueError("gsl_alpha_schedule must be one of: step5, step25, step50, linear, cosine")


def _step_alpha(epoch0: int, total_epochs: int, step_length: int) -> float:
    if step_length <= 0:
        raise ValueError("step_length must be positive")
    num_steps = max(1, (total_epochs - step_length) // step_length)
    if epoch0 >= total_epochs - step_length:
        return 0.0
    step = epoch0 // step_length
    return max(0.0, 1.0 - step / float(num_steps))


def _coerce_dtm(
    dtm: torch.Tensor,
    num_classes: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    dtm = dtm.to(device=device, dtype=dtype).abs()
    if dtm.ndim == 3:
        dtm = dtm[:, None]
    if dtm.shape[1] == num_classes:
        return dtm
    if dtm.shape[1] != num_classes - 1:
        raise ValueError(
            f"dtm must have {num_classes} or {num_classes - 1} channels, got {dtm.shape[1]}"
        )
    background = torch.amax(dtm, dim=1, keepdim=True)
    return torch.cat([background, dtm], dim=1)


def distance_transform_maps_torch(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Exact 2D classwise DTM using torch ops; practical for tests and fallback batches."""
    if target.ndim != 3:
        raise ValueError(f"expected target with shape [N,H,W], got {tuple(target.shape)}")
    device = target.device
    h, w = int(target.shape[-2]), int(target.shape[-1])
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    grid = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)
    maps: list[torch.Tensor] = []
    diagonal = math.sqrt(float(h * h + w * w))
    for sample in target.long():
        sample_maps = []
        for class_id in range(num_classes):
            mask = sample == class_id
            if not bool(mask.any()):
                sample_maps.append(torch.full((h, w), diagonal, device=device))
                continue
            points = torch.nonzero(mask, as_tuple=False).float()
            distances = torch.cdist(grid, points).amin(dim=1).reshape(h, w)
            sample_maps.append(distances)
        maps.append(torch.stack(sample_maps, dim=0))
    return torch.stack(maps, dim=0)


def distance_transform_maps_numpy(target: np.ndarray, num_classes: int) -> np.ndarray:
    """SciPy reference DTM used to verify the torch implementation in tests."""
    target_arr = np.asarray(target)
    if target_arr.ndim == 2:
        target_arr = target_arr[None]
    if target_arr.ndim != 3:
        raise ValueError(f"expected target with shape [N,H,W], got {target_arr.shape}")
    output = np.zeros((target_arr.shape[0], num_classes, *target_arr.shape[1:]), dtype=np.float32)
    diagonal = float(np.sqrt(target_arr.shape[-2] ** 2 + target_arr.shape[-1] ** 2))
    for sample_idx, sample in enumerate(target_arr):
        for class_id in range(num_classes):
            mask = sample == class_id
            if mask.any():
                output[sample_idx, class_id] = ndi.distance_transform_edt(~mask).astype(np.float32)
            else:
                output[sample_idx, class_id] = diagonal
    return output
