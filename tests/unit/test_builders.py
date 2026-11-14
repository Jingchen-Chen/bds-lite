import torch

from bds_lite.models import AttentionUNet, BDSLiteUNet, BDSLiteUNetHR, EGEUNet, MobileViTUNet, UNet
from bds_lite.training import build_model


def test_build_model_uses_dataset_shape_defaults() -> None:
    model = build_model(
        {
            "dataset": {"in_channels": 3, "num_classes": 2},
            "model": {"name": "unet", "base_channels": 8},
        }
    )

    assert isinstance(model, UNet)
    assert model.encoder.inc.net[0].in_channels == 3
    assert model.decoder.out.out_channels == 2


def test_build_bdslite_model_uses_dataset_shape_defaults() -> None:
    model = build_model(
        {
            "dataset": {"in_channels": 1, "num_classes": 9},
            "model": {"name": "bdslite_unet", "base_channels": 8},
        }
    )

    assert isinstance(model, BDSLiteUNet)
    assert model.encoder.inc.net[0].in_channels == 1
    assert model.seg_decoder.out.out_channels == 9


def test_build_bdslite_model_can_enable_boundary_gate() -> None:
    model = build_model(
        {
            "dataset": {"in_channels": 1, "num_classes": 3},
            "model": {
                "name": "bdslite_unet",
                "base_channels": 8,
                "boundary_gate": True,
                "boundary_gate_scale": 0.25,
            },
        }
    )

    assert isinstance(model, BDSLiteUNet)
    assert model.boundary_gate is not None


def test_build_bdslite_model_can_detach_boundary_features() -> None:
    model = build_model(
        {
            "dataset": {"in_channels": 1, "num_classes": 3},
            "model": {
                "name": "bdslite_unet",
                "base_channels": 8,
                "detach_boundary_features": True,
            },
        }
    )

    assert isinstance(model, BDSLiteUNet)
    assert model.detach_boundary_features is True


def test_build_bdslite_hr_model_forward_shape_and_boundary_toggle() -> None:
    model = build_model(
        {
            "dataset": {"in_channels": 1, "num_classes": 9},
            "model": {
                "name": "bdslite_unet_hr",
                "base_channels": 8,
                "boundary_classes": 1,
                "spatial_channels": 4,
                "boundary_gate": True,
                "boundary_gate_scale": 0.25,
            },
        }
    )

    assert isinstance(model, BDSLiteUNetHR)
    x = torch.zeros(2, 1, 224, 224)

    train_output = model(x, return_boundary=True)
    assert train_output["seg_logits"].shape == (2, 9, 224, 224)
    assert train_output["boundary_logits"].shape == (2, 1, 224, 224)
    assert train_output["seg_boundary_features"].shape[2:] == (224, 224)

    infer_output = model(x, return_boundary=False)
    assert infer_output["seg_logits"].shape == (2, 9, 224, 224)
    assert "boundary_logits" not in infer_output


def test_bdslite_hr_alias_builds_dual_stream_model() -> None:
    model = build_model(
        {
            "dataset": {"in_channels": 1, "num_classes": 9},
            "model": {
                "name": "bdslite_dual_stream_unet",
                "base_channels": 8,
                "spatial_channels": 4,
                "fusion_type": "add",
            },
        }
    )

    assert isinstance(model, BDSLiteUNetHR)


def test_build_attention_unet_forward_shape() -> None:
    model = build_model(
        {
            "dataset": {"in_channels": 1, "num_classes": 9},
            "model": {"name": "attention_unet", "base_channels": 8},
        }
    )

    assert isinstance(model, AttentionUNet)
    output = model(torch.zeros(1, 1, 32, 32))
    assert output["seg_logits"].shape == (1, 9, 32, 32)
    assert output["seg_features"].shape[1] == 8


def test_build_ege_unet_forward_shape_isic2018() -> None:
    model = build_model(
        {
            "dataset": {"in_channels": 3, "num_classes": 2},
            "model": {"name": "ege_unet"},
        }
    )
    assert isinstance(model, EGEUNet)
    out = model(torch.zeros(1, 3, 224, 224))
    assert out["seg_logits"].shape == (1, 2, 224, 224)


def test_build_ege_unet_forward_shape_synapse() -> None:
    model = build_model(
        {
            "dataset": {"in_channels": 1, "num_classes": 9},
            "model": {"name": "ege_unet"},
        }
    )
    assert isinstance(model, EGEUNet)
    out = model(torch.zeros(2, 1, 224, 224))
    assert out["seg_logits"].shape == (2, 9, 224, 224)


def test_ege_unet_is_compact_around_50k_parameters() -> None:
    model = build_model(
        {
            "dataset": {"in_channels": 3, "num_classes": 2},
            "model": {"name": "ege_unet"},
        }
    )
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Reference EGE-UNet (Ruan et al., MICCAI 2023) reports ~52K params.
    # Our port should be in the same order of magnitude.
    assert 30_000 < n < 80_000, f"unexpected EGE-UNet param count: {n:,}"


def test_build_mobilevit_unet_forward_shape() -> None:
    model = build_model(
        {
            "dataset": {"in_channels": 3, "num_classes": 2},
            "model": {
                "name": "mobilevit_unet",
                "base_channels": 8,
                "transformer_depth": 1,
                "transformer_heads": 4,
            },
        }
    )

    assert isinstance(model, MobileViTUNet)
    output = model(torch.zeros(1, 3, 32, 32))
    assert output["seg_logits"].shape == (1, 2, 32, 32)
    assert output["seg_features"].shape[1] == 8
