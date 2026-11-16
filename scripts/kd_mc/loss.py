"""Multiclass boundary-band probability distillation.

Protocol: protocols/kd_acdc_synapse/KD_MULTICLASS_PROTOCOL.md, section 4.

Generalisation of scripts/kd/kd_loss.py chosen so that it reduces to the ISIC2018 loss for K = 2:
- teacher hard map M_t = argmax_k p_t (for K = 2 this equals 1[p_fg > 0.5], ties going to
  background as with the strict threshold);
- band B = union over foreground classes c of [dilate(1[M_t = c]) - erode(1[M_t = c])], 3x3 square
  element applied once, zero-valued border; background is excluded, as in the repository's boundary
  targets (union over labels != 0) and as for ISIC2018;
- per-pixel loss: soft-target cross-entropy -sum_k p_t,k log p_s,k with both distributions clamped
  to [1e-6, 1 - 1e-6] (for K = 2 this is exactly the binary BCE of the ISIC2018 arm);
- L_kd = sum_batch B * CE / (sum_batch B + eps), eps = 1e-6; an empty band contributes nothing.
Computed in fp32 with autocast disabled.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from scripts.kd_mc.common import KD_SPEC


def class_band(teacher_prob: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """teacher_prob (N, K, H, W) -> band (N, H, W) float in {0, 1}."""
    labels = teacher_prob.float().argmax(1)
    classes = teacher_prob.shape[1]
    onehot = F.one_hot(labels, classes).permute(0, 3, 1, 2)[:, 1:].float()
    pad = kernel // 2
    padded = F.pad(onehot, (pad, pad, pad, pad), value=0.0)
    dilated = F.max_pool2d(padded, kernel, stride=1)
    eroded = -F.max_pool2d(-padded, kernel, stride=1)
    return (dilated - eroded).amax(1)


def kd_loss_mc(
    seg_logits: torch.Tensor,
    teacher_prob: torch.Tensor,
    eps: float = KD_SPEC["eps"],
    clamp: float = KD_SPEC["clamp"],
    kernel: int = KD_SPEC["band_kernel"],
) -> torch.Tensor:
    with torch.autocast(device_type=seg_logits.device.type, enabled=False):
        p_s = torch.softmax(seg_logits.float(), dim=1).clamp(clamp, 1.0 - clamp)
        p_t = teacher_prob.float().clamp(clamp, 1.0 - clamp)
        ce = -(p_t * torch.log(p_s)).sum(1)
        band = class_band(teacher_prob, kernel)
        return (band * ce).sum() / (band.sum() + eps)


class KDCriterionMC(nn.Module):
    """Unchanged decoder-only objective plus lambda_kd * L_kd on batches carrying teacher_prob."""

    def __init__(self, base: nn.Module, lambda_kd: float = KD_SPEC["lambda_kd"]):
        super().__init__()
        self.base = base
        self.lambda_kd = float(lambda_kd)

    @property
    def weights(self):
        return self.base.weights

    def set_epoch(self, epoch: int) -> None:
        self.base.set_epoch(epoch)

    def forward(self, outputs, batch):
        total, scalars = self.base(outputs, batch)
        if self.lambda_kd and "teacher_prob" in batch:
            value = kd_loss_mc(outputs["seg_logits"], batch["teacher_prob"])
            total = total + self.lambda_kd * value
            scalars["kd"] = float(value.detach().cpu())
            scalars["total"] = float(total.detach().cpu())
        return total, scalars
