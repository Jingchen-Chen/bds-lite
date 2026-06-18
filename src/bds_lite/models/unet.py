from __future__ import annotations

import torch
from torch import nn

from bds_lite.models.blocks import DoubleConv, DownBlock, UpBlock


class UNetEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.inc = DoubleConv(in_channels, c)
        self.down1 = DownBlock(c, c * 2)
        self.down2 = DownBlock(c * 2, c * 4)
        self.down3 = DownBlock(c * 4, c * 8)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        return [x1, x2, x3, x4]


class UNetDecoder(nn.Module):
    def __init__(self, num_classes: int, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.up1 = UpBlock(c * 8, c * 4, c * 4)
        self.up2 = UpBlock(c * 4, c * 2, c * 2)
        self.up3 = UpBlock(c * 2, c, c)
        self.out = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, features: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        x1, x2, x3, x4 = features
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        decoder_feature = self.up3(x, x1)
        return self.out(decoder_feature), decoder_feature


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self.encoder = UNetEncoder(in_channels, base_channels)
        self.decoder = UNetDecoder(num_classes, base_channels)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(x)
        logits, decoder_feature = self.decoder(features)
        return {"seg_logits": logits, "seg_features": decoder_feature}
