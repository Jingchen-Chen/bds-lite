"""Composite segmentation losses used by the reproducibility pipeline."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from bds_lite.losses.gsl import GeneralizedSurfaceLoss


class CompositeSegmentationLoss(nn.Module):
    """Dice + CE segmentation loss with optional BDS-Lite components."""

    def __init__(
        self,
        num_classes: int,
        lambda_boundary: float,
        lambda_surface: float,
        lambda_distill: float,
        lambda_deep_supervision: float,
        class_weights: Sequence[float] | None = None,
        boundary_dice_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        self.num_classes = int(num_classes)
        self.lambda_boundary = float(lambda_boundary)
        self.lambda_surface = float(lambda_surface)
        self.lambda_distill = float(lambda_distill)
        self.lambda_deep_supervision = float(lambda_deep_supervision)
        self.boundary_dice_weight = float(boundary_dice_weight)

        weights = _normalize_class_weights(class_weights, num_classes)
        self.register_buffer(
            "class_weights",
            torch.as_tensor(weights, dtype=torch.float32) if weights is not None else None,
        )

    def forward(
        self,
        outputs: torch.Tensor | dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        output_dict = _as_output_dict(outputs)
        seg_logits = output_dict["seg_logits"]
        target = batch["mask"].long()
        zero = seg_logits.new_tensor(0.0)

        seg_ce = F.cross_entropy(seg_logits, target, weight=self._ce_weights(seg_logits.device))
        seg_dice = _multiclass_dice_loss(
            seg_logits,
            target,
            class_weights=self._foreground_weights(seg_logits.device),
        )

        boundary_bce = zero
        boundary_dice = zero
        if self.lambda_boundary != 0.0 and "boundary_logits" in output_dict and "boundary" in batch:
            boundary_target = batch["boundary"].float()
            if boundary_target.ndim == 3:
                boundary_target = boundary_target[:, None]
            boundary_logits = output_dict["boundary_logits"]
            boundary_bce = F.binary_cross_entropy_with_logits(boundary_logits, boundary_target)
            boundary_dice = _binary_dice_loss(boundary_logits, boundary_target)

        surface = zero
        if self.lambda_surface != 0.0 and "sdf" in batch:
            surface = _surface_loss(seg_logits, batch["sdf"].float())

        distill_smooth_l1 = zero
        if self.lambda_distill != 0.0 and "boundary_features" in output_dict:
            seg_features = output_dict.get("seg_boundary_features", output_dict.get("seg_features"))
            boundary_features = output_dict["boundary_features"].detach()
            if seg_features is not None and seg_features.shape == boundary_features.shape:
                distill_smooth_l1 = F.smooth_l1_loss(seg_features, boundary_features)

        deep_supervision = zero
        if self.lambda_deep_supervision != 0.0:
            deep_supervision = self._deep_supervision_loss(output_dict, target)

        total = seg_dice + seg_ce
        total = total + self.lambda_boundary * (
            boundary_bce + self.boundary_dice_weight * boundary_dice
        )
        total = total + self.lambda_surface * surface
        total = total + self.lambda_distill * distill_smooth_l1
        total = total + self.lambda_deep_supervision * deep_supervision

        loss_dict = {
            "seg_dice": seg_dice,
            "seg_ce": seg_ce,
            "boundary_bce": boundary_bce,
            "boundary_dice": boundary_dice,
            "surface": surface,
            "distill_smooth_l1": distill_smooth_l1,
            "deep_supervision": deep_supervision,
            "total": total,
        }
        return total, loss_dict

    def _ce_weights(self, device: torch.device) -> torch.Tensor | None:
        return self.class_weights.to(device) if self.class_weights is not None else None

    def _foreground_weights(self, device: torch.device) -> torch.Tensor | None:
        if self.class_weights is None:
            return None
        return self.class_weights.to(device)[1:]

    def _deep_supervision_loss(
        self,
        outputs: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        aux_outputs = outputs.get("deep_supervision_logits", outputs.get("aux_seg_logits"))
        if aux_outputs is None:
            return target.new_tensor(0.0, dtype=torch.float32)
        if isinstance(aux_outputs, torch.Tensor):
            aux_outputs = [aux_outputs]

        losses = []
        for logits in aux_outputs:
            if logits.shape[2:] != target.shape[1:]:
                logits = F.interpolate(
                    logits,
                    size=target.shape[1:],
                    mode="bilinear",
                    align_corners=False,
                )
            losses.append(
                F.cross_entropy(logits, target, weight=self._ce_weights(logits.device))
                + _multiclass_dice_loss(
                    logits,
                    target,
                    class_weights=self._foreground_weights(logits.device),
                )
            )
        if not losses:
            return target.new_tensor(0.0, dtype=torch.float32)
        return torch.stack(losses).mean()


def _as_output_dict(outputs: torch.Tensor | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if isinstance(outputs, torch.Tensor):
        return {"seg_logits": outputs}
    if "seg_logits" not in outputs:
        raise KeyError("outputs must contain seg_logits")
    return outputs


def _normalize_class_weights(
    class_weights: Sequence[float] | None,
    num_classes: int,
) -> tuple[float, ...] | None:
    if class_weights is None:
        return None
    weights = tuple(float(value) for value in class_weights)
    if len(weights) == num_classes - 1:
        weights = (1.0, *weights)
    if len(weights) != num_classes:
        raise ValueError(
            "class_weights must contain either num_classes values or one value per foreground class"
        )
    if any(value < 0.0 for value in weights):
        raise ValueError("class_weights must be non-negative")
    if sum(weights) <= 0.0:
        raise ValueError("class_weights must contain at least one positive value")
    return weights


def _one_hot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    if weights is None:
        return values.mean()
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def _multiclass_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    target_oh = _one_hot(target, logits.shape[1])
    if logits.shape[1] > 1:
        probs = probs[:, 1:]
        target_oh = target_oh[:, 1:]
    dims = (0, 2, 3)
    intersection = torch.sum(probs * target_oh, dim=dims)
    denominator = torch.sum(probs + target_oh, dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - _weighted_mean(dice, class_weights)


def _binary_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(0, probs.ndim))
    intersection = torch.sum(probs * target, dim=dims)
    denominator = torch.sum(probs + target, dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice


def _surface_loss(logits: torch.Tensor, sdf: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] <= 1:
        return logits.new_tensor(0.0)
    probs = torch.softmax(logits, dim=1)[:, 1:]
    if sdf.ndim == 3:
        sdf = sdf[:, None]
    if probs.shape != sdf.shape:
        raise ValueError(
            f"surface loss expected sdf shape {tuple(probs.shape)}, got {tuple(sdf.shape)}"
        )
    distance = sdf.abs()
    dims = tuple(range(2, distance.ndim))
    scale = distance.amax(dim=dims, keepdim=True).clamp_min(1e-6)
    valid = (distance.amax(dim=dims, keepdim=True) > 0).float()
    if not bool(valid.any()):
        return logits.new_tensor(0.0)
    signed_distance = sdf / scale
    return (probs * signed_distance * valid).sum() / (
        valid.sum().clamp_min(1e-6) * probs.shape[-2] * probs.shape[-1]
    )


__all__ = [
    "CompositeSegmentationLoss",
    "GeneralizedSurfaceLoss",
]
