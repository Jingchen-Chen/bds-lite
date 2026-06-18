from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from bds_lite.models.blocks import DoubleConv, UpBlock
from bds_lite.models.unet import UNetDecoder, UNetEncoder


class BoundaryDecoder(nn.Module):
    """Small auxiliary boundary decoder used during training."""

    def __init__(self, boundary_classes: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        boundary_channels = max(c // 2, 8)
        self.up1 = UpBlock(c * 8, c * 4, c * 2)
        self.up2 = UpBlock(c * 2, c * 2, c)
        self.up3 = UpBlock(c, c, boundary_channels)
        self.out = nn.Conv2d(boundary_channels, boundary_classes, kernel_size=1)
        self.out_channels = boundary_channels

    def forward(self, features: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        x1, x2, x3, x4 = features
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        boundary_feature = self.up3(x, x1)
        return self.out(boundary_feature), boundary_feature


class BoundaryFeatureGate(nn.Module):
    """Tiny retained gate driven by the boundary-proxy segmentation feature."""

    def __init__(
        self,
        seg_channels: int,
        boundary_channels: int,
        scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.scale = float(scale)
        self.to_gate = nn.Conv2d(boundary_channels, seg_channels, kernel_size=1)
        nn.init.zeros_(self.to_gate.weight)
        nn.init.zeros_(self.to_gate.bias)

    def forward(
        self,
        seg_features: torch.Tensor,
        boundary_proxy: torch.Tensor,
    ) -> torch.Tensor:
        gate = torch.sigmoid(self.to_gate(boundary_proxy))
        return seg_features * (1.0 + self.scale * (gate - 0.5))


class HighResolutionSpatialPath(nn.Module):
    """Shallow full-resolution path for preserving fine spatial detail."""

    def __init__(self, in_channels: int, spatial_channels: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, spatial_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(spatial_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                spatial_channels,
                spatial_channels,
                kernel_size=3,
                padding=1,
                groups=spatial_channels,
                bias=False,
            ),
            nn.BatchNorm2d(spatial_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(spatial_channels, spatial_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(spatial_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HighResolutionFusion(nn.Module):
    """Fuse decoder features with the high-resolution spatial stream."""

    def __init__(
        self,
        seg_channels: int,
        spatial_channels: int,
        fusion_type: str = "concat",
    ) -> None:
        super().__init__()
        if fusion_type not in {"concat", "add"}:
            raise ValueError(f"unsupported high-resolution fusion type: {fusion_type}")
        self.fusion_type = fusion_type
        if fusion_type == "concat":
            self.fuse = nn.Sequential(
                nn.Conv2d(seg_channels + spatial_channels, seg_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(seg_channels),
                nn.ReLU(inplace=True),
                DoubleConv(seg_channels, seg_channels),
            )
        else:
            self.spatial_projection = nn.Conv2d(spatial_channels, seg_channels, kernel_size=1)
            self.fuse = DoubleConv(seg_channels, seg_channels)

    def forward(
        self,
        seg_features: torch.Tensor,
        spatial_features: torch.Tensor,
    ) -> torch.Tensor:
        if seg_features.shape[2:] != spatial_features.shape[2:]:
            spatial_features = F.interpolate(
                spatial_features,
                size=seg_features.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        if self.fusion_type == "concat":
            return self.fuse(torch.cat([seg_features, spatial_features], dim=1))
        return self.fuse(seg_features + self.spatial_projection(spatial_features))


class BDSLiteUNet(nn.Module):
    """Training-time boundary distillation variant of a lightweight U-Net."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 32,
        boundary_classes: int = 1,
        boundary_gate: bool = False,
        boundary_gate_scale: float = 0.5,
        detach_boundary_features: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = UNetEncoder(in_channels, base_channels)
        self.seg_decoder = UNetDecoder(num_classes, base_channels)
        self.boundary_decoder = BoundaryDecoder(boundary_classes, base_channels)
        self.detach_boundary_features = detach_boundary_features
        self.seg_to_boundary_feature = nn.Conv2d(
            base_channels,
            self.boundary_decoder.out_channels,
            kernel_size=1,
        )
        self.boundary_gate = (
            BoundaryFeatureGate(
                seg_channels=base_channels,
                boundary_channels=self.boundary_decoder.out_channels,
                scale=boundary_gate_scale,
            )
            if boundary_gate
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        return_boundary: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        if return_boundary is None:
            return_boundary = self.training

        features = self.encoder(x)
        seg_logits, seg_features = self.seg_decoder(features)
        seg_boundary_features = self.seg_to_boundary_feature(seg_features)
        if self.boundary_gate is not None:
            seg_features = self.boundary_gate(seg_features, seg_boundary_features)
            seg_logits = self.seg_decoder.out(seg_features)

        output = {
            "seg_logits": seg_logits,
            "seg_features": seg_features,
        }

        if return_boundary:
            boundary_features_input = (
                [feature.detach() for feature in features]
                if self.detach_boundary_features
                else features
            )
            boundary_logits, boundary_features = self.boundary_decoder(boundary_features_input)
            output["boundary_logits"] = boundary_logits
            output["boundary_features"] = boundary_features
            output["seg_boundary_features"] = seg_boundary_features

        return output


class BDSLiteUNetHR(nn.Module):
    """BDS-Lite U-Net with a shallow high-resolution spatial fusion path."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 32,
        boundary_classes: int = 1,
        spatial_channels: int = 16,
        fusion_type: str = "concat",
        boundary_gate: bool = True,
        boundary_gate_scale: float = 0.25,
        detach_boundary_features: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = UNetEncoder(in_channels, base_channels)
        self.seg_decoder = UNetDecoder(num_classes, base_channels)
        self.spatial_path = HighResolutionSpatialPath(in_channels, spatial_channels)
        self.hr_fusion = HighResolutionFusion(base_channels, spatial_channels, fusion_type)
        self.boundary_decoder = BoundaryDecoder(boundary_classes, base_channels)
        self.detach_boundary_features = detach_boundary_features
        self.seg_to_boundary_feature = nn.Conv2d(
            base_channels,
            self.boundary_decoder.out_channels,
            kernel_size=1,
        )
        self.boundary_gate = (
            BoundaryFeatureGate(
                seg_channels=base_channels,
                boundary_channels=self.boundary_decoder.out_channels,
                scale=boundary_gate_scale,
            )
            if boundary_gate
            else None
        )
        self.out = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        return_boundary: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        if return_boundary is None:
            return_boundary = self.training

        features = self.encoder(x)
        _, seg_features = self.seg_decoder(features)
        spatial_features = self.spatial_path(x)
        fused_features = self.hr_fusion(seg_features, spatial_features)
        seg_boundary_features = self.seg_to_boundary_feature(fused_features)
        if self.boundary_gate is not None:
            fused_features = self.boundary_gate(fused_features, seg_boundary_features)
        seg_logits = self.out(fused_features)

        output = {
            "seg_logits": seg_logits,
            "seg_features": fused_features,
        }

        if return_boundary:
            boundary_features_input = (
                [feature.detach() for feature in features]
                if self.detach_boundary_features
                else features
            )
            boundary_logits, boundary_features = self.boundary_decoder(boundary_features_input)
            output["boundary_logits"] = boundary_logits
            output["boundary_features"] = boundary_features
            output["seg_boundary_features"] = seg_boundary_features
            output["spatial_features"] = spatial_features

        return output
