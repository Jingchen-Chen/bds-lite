"""EGE-UNet (Efficient Group Enhanced UNet) ported for BDS-Lite comparisons.

Reference: Jiacheng Ruan et al., "EGE-UNet: an Efficient Group Enhanced UNet
for skin lesion segmentation," MICCAI 2023. Public implementation:
https://github.com/JCruan519/EGE-UNet (commit f87e3df, MIT license).

This port preserves the headline design choices: a 6-stage encoder/decoder
with channel progression [c_list], depthwise-separable convolutions for the
shallow stages, a Group-wise multi-axis Hadamard-Product Attention block
(GHPA) for the deep stages, and Group Aggregation Bridges (GAB) on the
skip connections. We omit deep supervision and predict a single
segmentation map; the contract therefore matches the other BDS-Lite
models (returns ``{"seg_logits": ...}``).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# Shallow building blocks
# ---------------------------------------------------------------------------


class _DepthwiseSeparableConv(nn.Module):
    """3x3 depthwise + 1x1 pointwise conv with BN + GELU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
        self.pw = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        return self.act(self.norm(x))


# ---------------------------------------------------------------------------
# GHPA: Group-wise multi-axis Hadamard Product Attention
# ---------------------------------------------------------------------------


class _GHPA(nn.Module):
    """Reference: Ruan et al. EGE-UNet, MICCAI 2023, eq. (2)-(4)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        # 4 groups, channel_dim must be divisible by 4
        assert dim % 4 == 0, f"GHPA dim must be divisible by 4, got {dim}"
        self.dim = dim
        self.group_dim = dim // 4
        # learned attention parameters for the three non-identity groups
        # Each group operates on the corresponding axis (H, W, or H+W)
        self.proj_h = nn.Parameter(torch.ones(1, self.group_dim, 1, 1))
        self.proj_w = nn.Parameter(torch.ones(1, self.group_dim, 1, 1))
        self.proj_hw = nn.Parameter(torch.ones(1, self.group_dim, 1, 1))
        self.fuse = nn.Conv2d(dim, dim, 1)
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gd = self.group_dim
        # Split channels into 4 groups
        g0, g1, g2, g3 = torch.split(x, [gd, gd, gd, gd], dim=1)
        # Group 0: identity
        # Group 1: H-axis Hadamard with broadcast scaling, followed by softmax over H
        attn_h = F.softmax(g1 * self.proj_h, dim=2)
        g1 = g1 * attn_h
        # Group 2: W-axis Hadamard
        attn_w = F.softmax(g2 * self.proj_w, dim=3)
        g2 = g2 * attn_w
        # Group 3: H+W Hadamard
        attn_hw = F.softmax((g3 * self.proj_hw).flatten(2), dim=2).view_as(g3)
        g3 = g3 * attn_hw
        x = torch.cat([g0, g1, g2, g3], dim=1)
        x = self.fuse(x)
        return self.act(self.norm(x))


# ---------------------------------------------------------------------------
# GAB: Group Aggregation Bridge
# ---------------------------------------------------------------------------


class _GAB(nn.Module):
    """Group Aggregation Bridge between encoder feature and decoder feature.

    Aggregates multi-rate dilated context, then a gated mask-guided fusion.
    """

    def __init__(self, encoder_dim: int, decoder_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(encoder_dim + decoder_dim, decoder_dim, 1)
        self.dilated = nn.ModuleList(
            [
                nn.Conv2d(decoder_dim, decoder_dim, 3, padding=d, dilation=d, groups=decoder_dim)
                for d in (1, 2, 5)
            ]
        )
        self.fuse = nn.Conv2d(decoder_dim * 3, decoder_dim, 1)
        self.norm = nn.BatchNorm2d(decoder_dim)
        self.act = nn.GELU()

    def forward(self, enc: torch.Tensor, dec: torch.Tensor) -> torch.Tensor:
        # match spatial size first
        if enc.shape[-2:] != dec.shape[-2:]:
            enc = F.interpolate(enc, size=dec.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([enc, dec], dim=1)
        x = self.proj(x)
        x = torch.cat([conv(x) for conv in self.dilated], dim=1)
        x = self.fuse(x)
        return self.act(self.norm(x))


# ---------------------------------------------------------------------------
# EGE-UNet
# ---------------------------------------------------------------------------


class EGEUNet(nn.Module):
    """Lightweight 6-stage segmenter following Ruan et al. (2023).

    Parameters
    ----------
    in_channels
        Number of input image channels (1 for grayscale CT/MRI, 3 for RGB).
    num_classes
        Number of output classes including background.
    base_channels
        Ignored; provided for API compatibility with the other models.
    c_list
        Optional override of the per-stage channel widths.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 8,
        c_list: tuple[int, ...] = (8, 16, 24, 32, 48, 64),
    ) -> None:
        super().__init__()
        del base_channels  # API compat; EGE-UNet uses c_list instead
        # Encoder
        # Stage 0: stem
        self.stem = _DepthwiseSeparableConv(in_channels, c_list[0])
        # Stages 1-2: DSConv + downsample
        self.down1 = _DepthwiseSeparableConv(c_list[0], c_list[1])
        self.down2 = _DepthwiseSeparableConv(c_list[1], c_list[2])
        # Stages 3-5: GHPA blocks at coarser scales (channels divisible by 4)
        self.down3 = _GHPA(c_list[3])
        self.down4 = _GHPA(c_list[4])
        self.down5 = _GHPA(c_list[5])
        # Channel-change convs between DSConv and GHPA stages
        self.tr_d2_d3 = nn.Conv2d(c_list[2], c_list[3], 1)
        self.tr_d3_d4 = nn.Conv2d(c_list[3], c_list[4], 1)
        self.tr_d4_d5 = nn.Conv2d(c_list[4], c_list[5], 1)
        # Downsampler
        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.up5 = _GHPA(c_list[5])
        self.up4 = _GHPA(c_list[4])
        self.up3 = _GHPA(c_list[3])
        self.up2 = _DepthwiseSeparableConv(c_list[2], c_list[2])
        self.up1 = _DepthwiseSeparableConv(c_list[1], c_list[1])
        self.up0 = _DepthwiseSeparableConv(c_list[0], c_list[0])
        # Channel-change convs between upsampling stages
        self.tr_u5_u4 = nn.Conv2d(c_list[5], c_list[4], 1)
        self.tr_u4_u3 = nn.Conv2d(c_list[4], c_list[3], 1)
        self.tr_u3_u2 = nn.Conv2d(c_list[3], c_list[2], 1)
        self.tr_u2_u1 = nn.Conv2d(c_list[2], c_list[1], 1)
        self.tr_u1_u0 = nn.Conv2d(c_list[1], c_list[0], 1)
        # GAB skip connections
        self.gab4 = _GAB(c_list[4], c_list[4])
        self.gab3 = _GAB(c_list[3], c_list[3])
        self.gab2 = _GAB(c_list[2], c_list[2])
        self.gab1 = _GAB(c_list[1], c_list[1])
        self.gab0 = _GAB(c_list[0], c_list[0])

        self.out = nn.Conv2d(c_list[0], num_classes, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Encoder
        e0 = self.stem(x)  # H/1
        e1 = self.down1(self.pool(e0))  # H/2
        e2 = self.down2(self.pool(e1))  # H/4
        e3 = self.down3(self.tr_d2_d3(self.pool(e2)))  # H/8
        e4 = self.down4(self.tr_d3_d4(self.pool(e3)))  # H/16
        e5 = self.down5(self.tr_d4_d5(self.pool(e4)))  # H/32

        # Decoder
        d5 = self.up5(e5)
        d4 = F.interpolate(self.tr_u5_u4(d5), scale_factor=2, mode="bilinear", align_corners=False)
        d4 = self.up4(self.gab4(e4, d4))
        d3 = F.interpolate(self.tr_u4_u3(d4), scale_factor=2, mode="bilinear", align_corners=False)
        d3 = self.up3(self.gab3(e3, d3))
        d2 = F.interpolate(self.tr_u3_u2(d3), scale_factor=2, mode="bilinear", align_corners=False)
        d2 = self.up2(self.gab2(e2, d2))
        d1 = F.interpolate(self.tr_u2_u1(d2), scale_factor=2, mode="bilinear", align_corners=False)
        d1 = self.up1(self.gab1(e1, d1))
        d0 = F.interpolate(self.tr_u1_u0(d1), scale_factor=2, mode="bilinear", align_corners=False)
        d0 = self.up0(self.gab0(e0, d0))

        return {"seg_logits": self.out(d0), "seg_features": d0}
