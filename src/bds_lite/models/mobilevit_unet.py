from __future__ import annotations

import torch
from torch import nn

from bds_lite.models.unet import UNetDecoder


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MobileConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels),
            DepthwiseSeparableConv(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MobileDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), MobileConvBlock(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MobileViTBlock(nn.Module):
    """Small bottleneck transformer block with local convolutional context."""

    def __init__(
        self,
        channels: int,
        transformer_dim: int,
        depth: int = 1,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        if transformer_dim % num_heads != 0:
            raise ValueError("transformer_dim must be divisible by num_heads")

        self.local = DepthwiseSeparableConv(channels, channels)
        self.to_tokens = nn.Conv2d(channels, transformer_dim, kernel_size=1, bias=False)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=transformer_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.from_tokens = nn.Conv2d(transformer_dim, channels, kernel_size=1, bias=False)
        self.fuse = DepthwiseSeparableConv(channels * 2, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_features = self.local(x)
        tokens = self.to_tokens(local_features)
        batch, channels, height, width = tokens.shape
        tokens = tokens.flatten(2).transpose(1, 2)
        tokens = self.transformer(tokens)
        tokens = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        global_features = self.from_tokens(tokens)
        return self.fuse(torch.cat([x, global_features], dim=1))


class MobileViTEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        transformer_dim: int | None = None,
        transformer_depth: int = 1,
        transformer_heads: int = 4,
    ) -> None:
        super().__init__()
        c = base_channels
        bottleneck_channels = c * 8
        if transformer_dim is None:
            transformer_dim = bottleneck_channels

        self.stem = MobileConvBlock(in_channels, c)
        self.down1 = MobileDownBlock(c, c * 2)
        self.down2 = MobileDownBlock(c * 2, c * 4)
        self.down3 = MobileDownBlock(c * 4, bottleneck_channels)
        self.mobilevit = MobileViTBlock(
            channels=bottleneck_channels,
            transformer_dim=transformer_dim,
            depth=transformer_depth,
            num_heads=transformer_heads,
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x1 = self.stem(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.mobilevit(self.down3(x3))
        return [x1, x2, x3, x4]


class MobileViTUNet(nn.Module):
    """MobileViT-style lightweight U-Net baseline."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 32,
        transformer_dim: int | None = None,
        transformer_depth: int = 1,
        transformer_heads: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = MobileViTEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            transformer_dim=transformer_dim,
            transformer_depth=transformer_depth,
            transformer_heads=transformer_heads,
        )
        self.decoder = UNetDecoder(num_classes=num_classes, base_channels=base_channels)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(x)
        logits, decoder_feature = self.decoder(features)
        return {"seg_logits": logits, "seg_features": decoder_feature}
