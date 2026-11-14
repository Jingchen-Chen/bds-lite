"""Models and objectives for the revision comparators.

C1  U-Net + Boundary Loss (Kervadec et al.). Official reference:
    https://github.com/LIVIAETS/boundary-loss @ 171c32d88a4ce59af8be46fb88b96d3637b9515b
    - SurfaceLoss: mean(probs[:, idc] * dist_maps[:, idc]) (losses.py)
    - dist maps:   one_hot2dist: edt(neg)*neg - (edt(pos)-1)*pos, unit resolution,
                   classes absent from the image left at zero (utils.py)
    - schedule:    gdl_surface_steal: weights start [1, 0.01]; after every epoch
                   [max(0.1, a - 0.01), b + 0.01] (scheduler.py StealWeight)
    Matched-protocol adaptation: the region term is this study's CE + soft Dice
    (identical to the U-Net baseline) instead of GDL, and idc = all foreground
    classes (the official binary setting uses idc=[1]).

C4  LB-UNet (Xu & Tong, MICCAI 2024), official model file fetched at a pinned commit
    (scripts/comparators/fetch_lbunet.py). Architecture and official widths
    c_list=[8,16,24,32,48,64] unchanged. Multi-class adaptation: the final 1x1 conv
    has K channels and its pre-sigmoid output (captured by a forward hook, then the
    same x2 bilinear align_corners=True upsampling as upstream) is used as softmax
    logits with CE + soft Dice; the five foreground prediction maps and three
    boundary maps keep the official deep-supervision weights (0.1..0.5 and
    0.1/0.2/0.3, BCE+Dice with weights 0.5/1) and are supervised with the foreground
    union and this study's union morphological boundary target (the official
    genetic-algorithm key-point boundary maps are ISIC-specific and not reproduced).

GSL U-Net + GSL, identical to the Phase16 matched runs: the tracked
    bds_lite.losses.gsl objective with its distance transform replaced in-process by
    the exact SciPy EDT (bds-followup scripts/gsl/gsl_memory_bounded.py).
"""

from __future__ import annotations

import importlib.util
import sys
import types

import numpy as np
import torch
from scipy import ndimage as ndi
from torch import nn
from torch.nn import functional as F

from bds_lite.training.losses import BDSLiteLoss, DiceLoss

# ---------------------------------------------------------------- C1 Boundary Loss


def kervadec_dist_maps(mask: np.ndarray, num_classes: int) -> np.ndarray:
    """Foreground-class signed distance maps exactly as official one_hot2dist."""
    out = np.zeros((num_classes - 1, *mask.shape), dtype=np.float32)
    for k in range(1, num_classes):
        posmask = mask == k
        if posmask.any():
            negmask = ~posmask
            out[k - 1] = (
                ndi.distance_transform_edt(negmask) * negmask
                - (ndi.distance_transform_edt(posmask) - 1) * posmask
            )
    return out


def kervadec_weights(epoch: int) -> tuple[float, float]:
    """(region, boundary) weights used during 1-indexed `epoch` under StealWeight(0.01)."""
    region, boundary = 1.0, 0.01
    for _ in range(epoch - 1):
        region, boundary = max(0.1, region - 0.01), boundary + 0.01
    return region, boundary


class KervadecBoundaryObjective(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dice = DiceLoss(include_background=False)
        self.region_w, self.boundary_w = kervadec_weights(1)

    def set_epoch(self, epoch: int) -> None:
        self.region_w, self.boundary_w = kervadec_weights(epoch)

    def forward(self, outputs, batch):
        logits = outputs["seg_logits"]
        mask = batch["mask"]
        ce = F.cross_entropy(logits, mask)
        dice = self.dice(logits, mask)
        if "kervadec_dist" in batch:
            probs = torch.softmax(logits.float(), dim=1)[:, 1:]
            boundary = (probs * batch["kervadec_dist"].float()).mean()
        else:  # validation batches: region loss only (selection uses DSC, not loss)
            boundary = logits.new_zeros(())
        total = self.region_w * (ce + dice) + self.boundary_w * boundary
        scalars = dict(
            ce=float(ce.detach()),
            dice=float(dice.detach()),
            boundary_loss=float(boundary.detach()),
            region_weight=self.region_w,
            boundary_weight=self.boundary_w,
            total=float(total.detach()),
        )
        return total, scalars


# ---------------------------------------------------------------- C4 LB-UNet


def _load_official_lbunet():
    from scripts.comparators.fetch_lbunet import ensure

    path = ensure()
    if "timm.models.layers" not in sys.modules:
        # Upstream imports timm only for trunc_normal_ (applied to nn.Linear, of which
        # LB-UNet has none); timm is not in the frozen environment.
        timm = types.ModuleType("timm")
        models = types.ModuleType("timm.models")
        layers = types.ModuleType("timm.models.layers")
        layers.trunc_normal_ = nn.init.trunc_normal_
        timm.models, models.layers = models, layers
        sys.modules.update({"timm": timm, "timm.models": models, "timm.models.layers": layers})
    spec = importlib.util.spec_from_file_location("lbunet_official_913ee68", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LBUNetK(nn.Module):
    """Official LB-UNet graph with K-class logits."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        official = _load_official_lbunet()
        self.net = official.LBUNet(num_classes=num_classes, input_channels=in_channels)
        self._final = None
        self.net.final.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self._final = output

    def forward(self, x: torch.Tensor, return_boundary: bool | None = None):
        if return_boundary is None:
            return_boundary = self.training
        gt_pre, boundary_pre, _ = self.net(x)
        logits = F.interpolate(
            self._final, scale_factor=(2, 2), mode="bilinear", align_corners=True
        )
        self._final = None
        output = {"seg_logits": logits}
        if return_boundary:
            output["lb_gt_pre"] = gt_pre
            output["lb_boundary_pre"] = boundary_pre
        return output


def _bce_dice_prob(pred: torch.Tensor, target: torch.Tensor, wb: float, wd: float):
    """Official BceDiceLoss on probabilities (BCELoss mean + per-sample smooth-1 Dice)."""
    size = pred.size(0)
    p, t = pred.reshape(size, -1).float(), target.reshape(size, -1).float()
    bce = F.binary_cross_entropy(p.clamp(1e-7, 1 - 1e-7), t)
    dice = 1 - ((2 * (p * t).sum(1) + 1) / (p.sum(1) + t.sum(1) + 1)).sum() / size
    return wd * dice + wb * bce


class LBUNetObjective(nn.Module):
    GT_WEIGHTS = (0.1, 0.2, 0.3, 0.4, 0.5)  # gt_pre5..gt_pre1
    BOUNDARY_WEIGHTS = (0.1, 0.2, 0.3)  # weight1..weight3

    def __init__(self) -> None:
        super().__init__()
        self.dice = DiceLoss(include_background=False)

    def set_epoch(self, epoch: int) -> None:
        pass

    def forward(self, outputs, batch):
        logits, mask = outputs["seg_logits"], batch["mask"]
        ce = F.cross_entropy(logits, mask)
        dice = self.dice(logits, mask)
        total = ce + dice
        scalars = dict(ce=float(ce.detach()), dice=float(dice.detach()))
        if "lb_gt_pre" in outputs:
            with torch.autocast(logits.device.type, enabled=False):
                foreground = (mask > 0).float()[:, None]
                boundary = batch["boundary"].float()
                gt = sum(
                    w * _bce_dice_prob(p, foreground, 0.5, 1)
                    for w, p in zip(self.GT_WEIGHTS, outputs["lb_gt_pre"], strict=True)
                )
                bnd = sum(
                    w * _bce_dice_prob(p, boundary, 0.5, 1)
                    for w, p in zip(self.BOUNDARY_WEIGHTS, outputs["lb_boundary_pre"], strict=True)
                )
            total = total + gt + bnd
            scalars.update(deep_gt=float(gt.detach()), deep_boundary=float(bnd.detach()))
        scalars["total"] = float(total.detach())
        return total, scalars


# ---------------------------------------------------------------- GSL (Phase16)


def distance_transform_maps_scipy(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Copy of the Phase16 linear-memory exact EDT wrapper (bds-followup)."""
    if target.ndim != 3:
        raise ValueError(f"expected target with shape [N,H,W], got {tuple(target.shape)}")
    target_array = target.detach().cpu().numpy()
    batch, height, width = target_array.shape
    output = np.empty((batch, int(num_classes), height, width), dtype=np.float32)
    diagonal = float(np.sqrt(height**2 + width**2))
    for sample_index, sample in enumerate(target_array):
        for class_id in range(int(num_classes)):
            mask = sample == class_id
            if mask.any():
                output[sample_index, class_id] = ndi.distance_transform_edt(~mask).astype(
                    np.float32
                )
            else:
                output[sample_index, class_id].fill(diagonal)
    return torch.from_numpy(output).to(device=target.device)


def install_phase16_gsl() -> None:
    import bds_lite.losses.gsl as gsl_module

    gsl_module.distance_transform_maps_torch = distance_transform_maps_scipy


# ---------------------------------------------------------------- factories


def build(config):
    """Return (model, criterion) for a comparator config."""
    from bds_lite.training import build_model
    from bds_lite.training.losses import weights_from_config

    kind = config["comparator"]["kind"]
    classes = config["dataset"]["num_classes"]
    if kind == "c1_boundary_loss":
        return build_model(config), KervadecBoundaryObjective(classes)
    if kind == "c4_lbunet":
        return LBUNetK(config["dataset"]["in_channels"], classes), LBUNetObjective()
    if kind == "gsl":
        install_phase16_gsl()
        return build_model(config), BDSLiteLoss(weights_from_config(config))
    raise ValueError(f"unknown comparator kind: {kind}")
