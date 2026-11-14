"""Boundary-band probability distillation loss (protocol 4.2, OPEN-5).

Implementation choices fixed in the protocol, not specified by BDKD-Net:
- w = 3 is read as the side of a 3x3 square structuring element, applied once.
- Dilation and erosion use a zero-valued border (as scipy.ndimage), so a region touching the image
  edge has a band along that edge, as in the repository's boundary targets and Boundary F1 contours.
- Normalisation is over the whole batch: sum of band-weighted BCE over all pixels of all images,
  divided by the total band pixel count plus eps. An image with an empty band adds nothing to
  either sum; a batch whose band is empty everywhere gives a loss of exactly zero.
- eps = 1e-6; student and teacher probabilities are clamped to [1e-6, 1 - 1e-6].
The loss is computed in fp32 with autocast disabled.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from scripts.kd.common import KD_SPEC


def teacher_band(
    teacher_prob: torch.Tensor, threshold: float = 0.5, kernel: int = 3
) -> torch.Tensor:
    """B = dilate(M_t) - erode(M_t) with a kernel x kernel square, zero border; (N, H, W) float."""
    if kernel % 2 != 1:
        raise ValueError("kernel must be odd")
    mask = (teacher_prob.float() > threshold).float()[:, None]
    pad = kernel // 2
    padded = F.pad(mask, (pad, pad, pad, pad), value=0.0)
    dilated = F.max_pool2d(padded, kernel, stride=1)
    eroded = -F.max_pool2d(-padded, kernel, stride=1)
    return (dilated - eroded)[:, 0]


def kd_loss(
    seg_logits: torch.Tensor,
    teacher_prob: torch.Tensor,
    mode: str,
    threshold: float = KD_SPEC["threshold"],
    kernel: int = KD_SPEC["band_kernel"],
    eps: float = KD_SPEC["eps"],
    clamp: float = KD_SPEC["clamp"],
) -> torch.Tensor:
    """sum(B * BCE(p_s, p_t)) / (sum(B) + eps) over the batch; B = 1 everywhere for mode='full'."""
    with torch.autocast(device_type=seg_logits.device.type, enabled=False):
        p_s = torch.softmax(seg_logits.float(), dim=1)[:, 1].clamp(clamp, 1.0 - clamp)
        p_t = teacher_prob.float().clamp(clamp, 1.0 - clamp)
        bce = -(p_t * torch.log(p_s) + (1.0 - p_t) * torch.log(1.0 - p_s))
        if mode == "band":
            band = teacher_band(teacher_prob, threshold, kernel)
        elif mode == "full":
            band = torch.ones_like(bce)
        else:
            raise ValueError(f"unknown kd mode: {mode}")
        return (band * bce).sum() / (band.sum() + eps)


class KDCriterion(nn.Module):
    """The unchanged decoder-only objective plus lambda_kd * L_kd when a teacher cache is present.

    Validation batches carry no teacher probabilities; the term is then skipped, which affects only
    the logged validation loss (checkpoint selection uses DSC).
    """

    def __init__(self, base: nn.Module, mode: str, lambda_kd: float = KD_SPEC["lambda_kd"]):
        super().__init__()
        self.base = base
        self.mode = mode
        self.lambda_kd = float(lambda_kd)

    @property
    def weights(self):
        return self.base.weights

    def set_epoch(self, epoch: int) -> None:
        self.base.set_epoch(epoch)

    def forward(self, outputs, batch):
        total, scalars = self.base(outputs, batch)
        if self.lambda_kd and "teacher_prob" in batch:
            value = kd_loss(outputs["seg_logits"], batch["teacher_prob"], self.mode)
            total = total + self.lambda_kd * value
            scalars["kd"] = float(value.detach().cpu())
            scalars["total"] = float(total.detach().cpu())
        return total, scalars
