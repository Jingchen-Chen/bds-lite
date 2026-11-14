from __future__ import annotations

from typing import Any

from torch import nn

from bds_lite.models import AttentionUNet, BDSLiteUNet, BDSLiteUNetHR, EGEUNet, MobileViTUNet, UNet


def build_model(config: dict[str, Any]) -> nn.Module:
    model_cfg = config.get("model", {})
    dataset_cfg = config.get("dataset", {})
    name = model_cfg.get("name", "unet")
    common = {
        "in_channels": int(model_cfg.get("in_channels", dataset_cfg.get("in_channels", 1))),
        "num_classes": int(model_cfg.get("num_classes", dataset_cfg.get("num_classes", 2))),
        "base_channels": int(model_cfg.get("base_channels", 32)),
    }

    if name == "unet":
        return UNet(**common)
    if name == "ege_unet":
        c_list = model_cfg.get("c_list")
        if c_list is not None:
            return EGEUNet(**common, c_list=tuple(int(c) for c in c_list))
        return EGEUNet(**common)
    if name == "attention_unet":
        return AttentionUNet(**common)
    if name == "mobilevit_unet":
        return MobileViTUNet(
            **common,
            transformer_dim=(
                int(model_cfg["transformer_dim"])
                if model_cfg.get("transformer_dim") is not None
                else None
            ),
            transformer_depth=int(model_cfg.get("transformer_depth", 1)),
            transformer_heads=int(model_cfg.get("transformer_heads", 4)),
        )
    if name == "bdslite_unet":
        return BDSLiteUNet(
            **common,
            boundary_classes=int(model_cfg.get("boundary_classes", 1)),
            boundary_gate=bool(model_cfg.get("boundary_gate", False)),
            boundary_gate_scale=float(model_cfg.get("boundary_gate_scale", 0.5)),
            detach_boundary_features=bool(model_cfg.get("detach_boundary_features", False)),
        )
    if name in {"bdslite_unet_hr", "bdslite_dual_stream_unet"}:
        return BDSLiteUNetHR(
            **common,
            boundary_classes=int(model_cfg.get("boundary_classes", 1)),
            spatial_channels=int(model_cfg.get("spatial_channels", 16)),
            fusion_type=str(model_cfg.get("fusion_type", "concat")),
            boundary_gate=bool(model_cfg.get("boundary_gate", True)),
            boundary_gate_scale=float(model_cfg.get("boundary_gate_scale", 0.25)),
            detach_boundary_features=bool(model_cfg.get("detach_boundary_features", False)),
        )
    raise ValueError(f"unknown model name: {name}")
