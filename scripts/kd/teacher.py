"""External teacher: smp U-Net with an ImageNet ResNet-34 encoder (protocol 5).

ImageNet mean/std normalisation is applied inside the wrapper; stored images are in [0, 1], which
is the input range declared by the pinned encoder weights. The weight file is hash-checked before
use and is read from the local Hugging Face cache only (HF_HUB_OFFLINE=1).
"""

from __future__ import annotations

import os

import torch
from torch import nn

from scripts.kd.common import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    SMP_VERSION,
    WEIGHTS_REPO,
    WEIGHTS_REVISION,
    WEIGHTS_SHA256,
    sha,
)


def verify_weights() -> str:
    os.environ["HF_HUB_OFFLINE"] = "1"
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(WEIGHTS_REPO, filename="model.safetensors", revision=WEIGHTS_REVISION)
    if sha(path) != WEIGHTS_SHA256:
        raise ValueError("teacher encoder weights differ from the pinned file")
    return path


class TeacherNet(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        os.environ["HF_HUB_OFFLINE"] = "1"
        import segmentation_models_pytorch as smp

        if smp.__version__ != SMP_VERSION:
            raise RuntimeError(f"segmentation_models_pytorch {smp.__version__} != {SMP_VERSION}")
        if pretrained:
            verify_weights()
        self.net = smp.Unet(
            "resnet34",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=2,
        )
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"seg_logits": self.net((x - self.mean) / self.std)}
