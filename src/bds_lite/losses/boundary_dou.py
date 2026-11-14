"""Boundary Difference over Union loss.

Implementation follows Sun, F., Luo, Z., and Li, S. "Boundary Difference over
Union Loss for Medical Image Segmentation." MICCAI 2023, LNCS 14223,
pp. 292-302. DOI/proceedings: https://doi.org/10.1007/978-3-031-43901-8_28.

The target-size-adaptive boundary attention term is ported from the public
reference implementation at https://github.com/sunfan-bvb/BoundaryDoULoss.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BoundaryDoULoss(nn.Module):
    """Multi-class Boundary DoU loss with the paper's adaptive alpha schedule."""

    def __init__(self, num_classes: int, eps: float = 1e-5, alpha_max: float = 0.8) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        self.num_classes = int(num_classes)
        self.eps = float(eps)
        self.alpha_max = float(alpha_max)
        kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]])
        self.register_buffer("kernel", kernel.view(1, 1, 3, 3))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return the mean Boundary DoU loss over classes, including background."""
        if logits.ndim != 4:
            raise ValueError(f"expected logits with shape [N,C,H,W], got {tuple(logits.shape)}")
        if logits.shape[1] != self.num_classes:
            raise ValueError(f"expected {self.num_classes} channels, got {logits.shape[1]}")
        if target.shape != logits.shape[:1] + logits.shape[2:]:
            raise ValueError(
                f"target shape {tuple(target.shape)} does not match logits {tuple(logits.shape)}"
            )

        probs = torch.softmax(logits, dim=1)
        target_oh = F.one_hot(target.long(), num_classes=self.num_classes).permute(0, 3, 1, 2)
        target_oh = target_oh.to(dtype=probs.dtype, device=probs.device)

        losses = [
            self._adaptive_class_loss(probs[:, class_id], target_oh[:, class_id])
            for class_id in range(self.num_classes)
        ]
        return torch.stack(losses).mean()

    def adaptive_alpha(self, target: torch.Tensor) -> torch.Tensor:
        """Compute the target-size-adaptive alpha used by the MICCAI paper."""
        target = target.to(dtype=torch.float32)
        if target.ndim == 2:
            target = target[None]
        if target.ndim != 3:
            raise ValueError(f"expected target with shape [N,H,W], got {tuple(target.shape)}")
        kernel = self.kernel.to(device=target.device, dtype=target.dtype)
        neighbor_count = F.conv2d(target[:, None], kernel, padding=1)[:, 0]
        boundary_attention = neighbor_count * target
        boundary_attention = torch.where(
            boundary_attention == 5,
            torch.zeros_like(boundary_attention),
            boundary_attention,
        )
        c_count = torch.count_nonzero(boundary_attention).to(dtype=target.dtype)
        s_count = torch.count_nonzero(target).to(dtype=target.dtype)
        alpha = 1.0 - (c_count + self.eps) / (s_count + self.eps)
        alpha = 2.0 * alpha - 1.0
        return torch.minimum(alpha, alpha.new_tensor(self.alpha_max))

    def _adaptive_class_loss(self, score: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        alpha = self.adaptive_alpha(target)
        intersect = torch.sum(score * target)
        target_sum = torch.sum(target * target)
        score_sum = torch.sum(score * score)
        numerator = score_sum + target_sum - 2.0 * intersect + self.eps
        denominator = score_sum + target_sum - (1.0 + alpha) * intersect + self.eps
        return numerator / denominator.clamp_min(self.eps)
