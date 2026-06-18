from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from bds_lite.losses.gsl import GeneralizedSurfaceLoss


@dataclass(frozen=True)
class LossWeights:
    """Scalar weights for the BDS-Lite training objective."""

    dice: float = 1.0
    ce: float = 1.0
    boundary: float = 0.0
    distill: float = 0.0
    surface: float = 0.0
    distance: float = 0.0
    far_fp: float = 0.0
    foreground_class_weights: tuple[float, ...] | None = None
    use_gsl: bool = False
    lambda_gsl: float = 1.0
    gsl_alpha_schedule: str = "step5"
    num_classes: int = 2
    total_epochs: int = 1


def _one_hot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()


def _foreground_weights(
    weights: tuple[float, ...] | None,
    num_foreground_classes: int,
    device: torch.device,
) -> torch.Tensor | None:
    if weights is None:
        return None
    if len(weights) != num_foreground_classes:
        raise ValueError(
            "foreground_class_weights must contain one value per foreground class: "
            f"expected {num_foreground_classes}, got {len(weights)}"
        )
    tensor = torch.as_tensor(weights, dtype=torch.float32, device=device)
    if torch.any(tensor < 0):
        raise ValueError("foreground_class_weights must be non-negative")
    if float(tensor.sum()) <= 0:
        raise ValueError("foreground_class_weights must contain at least one positive value")
    return tensor


def _weighted_channel_mean(values: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    if weights is None:
        return values.mean()
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


class DiceLoss(nn.Module):
    def __init__(
        self,
        include_background: bool = False,
        eps: float = 1e-6,
        foreground_class_weights: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()
        self.include_background = include_background
        self.eps = eps
        self.foreground_class_weights = foreground_class_weights

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        target_oh = _one_hot(target, num_classes)
        class_weights = None

        if not self.include_background and num_classes > 1:
            probs = probs[:, 1:]
            target_oh = target_oh[:, 1:]
            class_weights = _foreground_weights(
                self.foreground_class_weights,
                probs.shape[1],
                logits.device,
            )

        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_oh, dim=dims)
        denominator = torch.sum(probs + target_oh, dim=dims)
        dice = (2.0 * intersection + self.eps) / (denominator + self.eps)
        return 1.0 - _weighted_channel_mean(dice, class_weights)


class BinaryDiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = (0, 2, 3)
        intersection = torch.sum(probs * target, dim=dims)
        denominator = torch.sum(probs + target, dim=dims)
        dice = (2.0 * intersection + self.eps) / (denominator + self.eps)
        return 1.0 - dice.mean()


class SdfWeightedSurfaceLoss(nn.Module):
    """Signed-distance surface loss for foreground probabilities.

    The dataset stores one signed-distance map for each foreground class. This
    loss follows the surface-loss idea: foreground probability is multiplied by
    the ground-truth signed distance map. Outside probabilities are penalized by
    positive distances, while inside probabilities receive negative cost.
    """

    def __init__(
        self,
        eps: float = 1e-6,
        foreground_class_weights: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.foreground_class_weights = foreground_class_weights

    def forward(self, logits: torch.Tensor, sdf: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] <= 1:
            raise ValueError("surface loss requires at least one foreground class")

        probs = torch.softmax(logits, dim=1)[:, 1:]
        if sdf.ndim == 3:
            sdf = sdf[:, None]
        if probs.shape != sdf.shape:
            raise ValueError(
                f"surface loss expected sdf shape {tuple(probs.shape)}, got {tuple(sdf.shape)}"
            )

        distance = sdf.abs()
        dims = tuple(range(2, distance.ndim))
        scale = distance.amax(dim=dims, keepdim=True).clamp_min(self.eps)
        valid = distance.amax(dim=dims, keepdim=True) > 0
        if not bool(valid.any()):
            return logits.new_tensor(0.0)

        signed_distance = sdf / scale
        class_weights = _foreground_weights(
            self.foreground_class_weights,
            probs.shape[1],
            logits.device,
        )
        if class_weights is not None:
            valid_class = valid.any(dim=0).flatten()
            weighted_values = (probs * signed_distance * valid.float()).sum(dim=(0, 2, 3)) / (
                valid.float().sum(dim=(0, 2, 3)).clamp_min(self.eps)
                * probs.shape[-2]
                * probs.shape[-1]
            )
            return _weighted_channel_mean(weighted_values[valid_class], class_weights[valid_class])

        return (probs * signed_distance * valid.float()).sum() / (
            valid.float().sum() * probs.shape[-2] * probs.shape[-1]
        )


class SdfDistanceErrorLoss(nn.Module):
    """Distance-weighted foreground probability error.

    Unlike the signed surface loss, this term is nonnegative. It penalizes false
    foreground and missed foreground more when the error is far from the true
    class surface, giving HD95/ASSD outliers a stronger training signal.
    """

    def __init__(
        self,
        eps: float = 1e-6,
        foreground_class_weights: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.foreground_class_weights = foreground_class_weights

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor, sdf: torch.Tensor
    ) -> torch.Tensor:
        if logits.shape[1] <= 1:
            raise ValueError("distance error loss requires at least one foreground class")

        probs = torch.softmax(logits, dim=1)[:, 1:]
        target_oh = _one_hot(target, logits.shape[1])[:, 1:]
        if sdf.ndim == 3:
            sdf = sdf[:, None]
        if probs.shape != sdf.shape:
            raise ValueError(
                f"distance error loss expected sdf shape {tuple(probs.shape)}, "
                f"got {tuple(sdf.shape)}"
            )

        distance = sdf.abs()
        dims = tuple(range(2, distance.ndim))
        max_distance = distance.amax(dim=dims, keepdim=True)
        valid = max_distance > 0
        if not bool(valid.any()):
            return logits.new_tensor(0.0)

        normalized_distance = distance / max_distance.clamp_min(self.eps)
        error = (probs - target_oh).abs()
        class_weights = _foreground_weights(
            self.foreground_class_weights,
            probs.shape[1],
            logits.device,
        )
        if class_weights is not None:
            valid_class = valid.any(dim=0).flatten()
            weighted_values = (error * normalized_distance * valid.float()).sum(dim=(0, 2, 3)) / (
                valid.float().sum(dim=(0, 2, 3)).clamp_min(self.eps)
                * probs.shape[-2]
                * probs.shape[-1]
            )
            return _weighted_channel_mean(weighted_values[valid_class], class_weights[valid_class])

        return (error * normalized_distance * valid.float()).sum() / (
            valid.float().sum() * probs.shape[-2] * probs.shape[-1]
        )


class SdfFarFalsePositiveLoss(nn.Module):
    """Penalize confident foreground probability far outside the target class.

    This narrower SDF loss targets HD95/ASSD outliers caused by remote false
    positives without adding extra penalty to missed foreground pixels.
    """

    def __init__(
        self,
        eps: float = 1e-6,
        foreground_class_weights: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.foreground_class_weights = foreground_class_weights

    def forward(self, logits: torch.Tensor, sdf: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] <= 1:
            raise ValueError("far false-positive loss requires at least one foreground class")

        probs = torch.softmax(logits, dim=1)[:, 1:]
        if sdf.ndim == 3:
            sdf = sdf[:, None]
        if probs.shape != sdf.shape:
            raise ValueError(
                f"far false-positive loss expected sdf shape {tuple(probs.shape)}, "
                f"got {tuple(sdf.shape)}"
            )

        outside_distance = torch.relu(sdf)
        dims = tuple(range(2, outside_distance.ndim))
        max_distance = outside_distance.amax(dim=dims, keepdim=True)
        valid = max_distance > 0
        if not bool(valid.any()):
            return logits.new_tensor(0.0)

        normalized_distance = outside_distance / max_distance.clamp_min(self.eps)
        weighted_false_positive = probs.square() * normalized_distance
        class_weights = _foreground_weights(
            self.foreground_class_weights,
            probs.shape[1],
            logits.device,
        )
        if class_weights is not None:
            valid_class = valid.any(dim=0).flatten()
            weighted_values = (weighted_false_positive * valid.float()).sum(dim=(0, 2, 3)) / (
                valid.float().sum(dim=(0, 2, 3)).clamp_min(self.eps)
                * probs.shape[-2]
                * probs.shape[-1]
            )
            return _weighted_channel_mean(weighted_values[valid_class], class_weights[valid_class])

        return (weighted_false_positive * valid.float()).sum() / (
            valid.float().sum() * probs.shape[-2] * probs.shape[-1]
        )


class BDSLiteLoss(nn.Module):
    """Composite segmentation + boundary + distillation loss for BDS-Lite."""

    def __init__(self, weights: LossWeights | None = None) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        self.dice = DiceLoss(
            include_background=False,
            foreground_class_weights=self.weights.foreground_class_weights,
        )
        self.boundary_dice = BinaryDiceLoss()
        self.surface = SdfWeightedSurfaceLoss(
            foreground_class_weights=self.weights.foreground_class_weights,
        )
        self.distance = SdfDistanceErrorLoss(
            foreground_class_weights=self.weights.foreground_class_weights,
        )
        self.far_fp = SdfFarFalsePositiveLoss(
            foreground_class_weights=self.weights.foreground_class_weights,
        )
        if self.weights.use_gsl:
            self.gsl: GeneralizedSurfaceLoss | None = GeneralizedSurfaceLoss(
                num_classes=self.weights.num_classes,
                total_epochs=self.weights.total_epochs,
                alpha_schedule=self.weights.gsl_alpha_schedule,
            )
        else:
            self.gsl = None

    def set_epoch(self, epoch: int) -> None:
        if self.gsl is not None:
            self.gsl.set_epoch(epoch)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        seg_logits = outputs["seg_logits"]
        mask = batch["mask"]

        losses: dict[str, torch.Tensor] = {}
        if self.weights.ce:
            losses["ce"] = F.cross_entropy(seg_logits, mask)
        if self.weights.dice:
            losses["dice"] = self.dice(seg_logits, mask)

        if self.weights.boundary and "boundary_logits" in outputs and "boundary" in batch:
            boundary = batch["boundary"].float()
            boundary_logits = outputs["boundary_logits"]
            losses["boundary_bce"] = F.binary_cross_entropy_with_logits(boundary_logits, boundary)
            losses["boundary_dice"] = self.boundary_dice(boundary_logits, boundary)

        if (
            self.weights.distill
            and "seg_boundary_features" in outputs
            and "boundary_features" in outputs
        ):
            losses["distill"] = F.smooth_l1_loss(
                outputs["seg_boundary_features"],
                outputs["boundary_features"].detach(),
            )

        if self.weights.surface and "sdf" in batch:
            losses["surface"] = self.surface(seg_logits, batch["sdf"].float())

        if self.weights.distance and "sdf" in batch:
            losses["distance"] = self.distance(seg_logits, mask, batch["sdf"].float())

        if self.weights.far_fp and "sdf" in batch:
            losses["far_fp"] = self.far_fp(seg_logits, batch["sdf"].float())

        if self.gsl is not None:
            gsl_total, gsl_log = self.gsl(seg_logits, mask)
            losses["gsl"] = self.weights.lambda_gsl * gsl_total
            losses["gsl_region"] = gsl_log["gsl_region"]
            losses["gsl_boundary"] = gsl_log["gsl_boundary"]
            losses["gsl_alpha"] = gsl_log["gsl_alpha"]

        total = seg_logits.new_tensor(0.0)
        total = total + self.weights.ce * losses.get("ce", total.new_tensor(0.0))
        total = total + self.weights.dice * losses.get("dice", total.new_tensor(0.0))
        boundary_loss = losses.get("boundary_bce", total.new_tensor(0.0)) + losses.get(
            "boundary_dice",
            total.new_tensor(0.0),
        )
        total = total + self.weights.boundary * boundary_loss
        total = total + self.weights.distill * losses.get("distill", total.new_tensor(0.0))
        total = total + self.weights.surface * losses.get("surface", total.new_tensor(0.0))
        total = total + self.weights.distance * losses.get("distance", total.new_tensor(0.0))
        total = total + self.weights.far_fp * losses.get("far_fp", total.new_tensor(0.0))
        total = total + losses.get("gsl", total.new_tensor(0.0))

        scalars = {name: float(value.detach().cpu()) for name, value in losses.items()}
        scalars["total"] = float(total.detach().cpu())
        return total, scalars


def weights_from_config(config: dict) -> LossWeights:
    loss_cfg = config.get("loss", {})
    dataset_cfg = config.get("dataset", {})
    trainer_cfg = config.get("trainer", {})
    model_cfg = config.get("model", {})
    class_weights = loss_cfg.get("foreground_class_weights")
    num_classes = int(model_cfg.get("num_classes", dataset_cfg.get("num_classes", 2)))
    total_epochs = int(trainer_cfg.get("epochs", 1))
    return LossWeights(
        dice=float(loss_cfg.get("dice_weight", 1.0)),
        ce=float(loss_cfg.get("ce_weight", 1.0)),
        boundary=float(loss_cfg.get("boundary_weight", 0.0)),
        distill=float(loss_cfg.get("distill_weight", 0.0)),
        surface=float(loss_cfg.get("surface_weight", 0.0)),
        distance=float(loss_cfg.get("distance_weight", 0.0)),
        far_fp=float(loss_cfg.get("far_fp_weight", 0.0)),
        foreground_class_weights=tuple(float(value) for value in class_weights)
        if class_weights is not None
        else None,
        use_gsl=bool(loss_cfg.get("use_gsl", False)),
        lambda_gsl=float(loss_cfg.get("lambda_gsl", 1.0)),
        gsl_alpha_schedule=str(loss_cfg.get("gsl_alpha_schedule", "step5")),
        num_classes=num_classes,
        total_epochs=total_epochs,
    )
