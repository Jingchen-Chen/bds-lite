"""Multiclass external teacher: smp U-Net / ImageNet ResNet-34 (multiclass protocol 3).

Single-channel inputs are replicated to three channels so that the pretrained first convolution is
used unchanged. Synapse (windowed CT in [0, 1]) then receives ImageNet mean/std normalisation, as
ISIC2018 does; ACDC is stored per-volume z-scored (mean 0, SD about 1), which is already the scale
of ImageNet-normalised inputs, so it is passed unchanged.
"""

from __future__ import annotations

import os

import torch
from torch import nn

from scripts.kd.common import IMAGENET_MEAN, IMAGENET_STD, SMP_VERSION
from scripts.kd.teacher import verify_weights


class TeacherNetMC(nn.Module):
    def __init__(self, classes: int, input_norm: str, pretrained: bool = True) -> None:
        super().__init__()
        os.environ["HF_HUB_OFFLINE"] = "1"
        import segmentation_models_pytorch as smp

        if smp.__version__ != SMP_VERSION:
            raise RuntimeError(f"segmentation_models_pytorch {smp.__version__} != {SMP_VERSION}")
        if input_norm not in {"imagenet", "none"}:
            raise ValueError(input_norm)
        if pretrained:
            verify_weights()
        self.net = smp.Unet(
            "resnet34",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=classes,
        )
        self.input_norm = input_norm
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        if self.input_norm == "imagenet":
            x = (x - self.mean) / self.std
        return {"seg_logits": self.net(x)}


def build_teacher(config: dict, pretrained: bool) -> TeacherNetMC:
    return TeacherNetMC(
        config["model"]["num_classes"], config["model"]["input_norm"], pretrained=pretrained
    )
