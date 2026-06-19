import torch

from bds_lite.models import BDSLiteUNet, BDSLiteUNetHR, UNet
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


def test_config_of_record_name_alias_builds_bdslite() -> None:
    # The resolved config snapshots (configs/run_resolved/) use model.name "bds_lite";
    # the public dispatch key is "bdslite_unet". Both must build the same model.
    # See docs/config_of_record.md.
    record = build_model(
        {
            "dataset": {"in_channels": 3, "num_classes": 2},
            "model": {
                "name": "bds_lite",
                "base_channels": 32,
                "boundary_gate": True,
                "boundary_gate_scale": 0.25,
            },
        }
    )
    public = build_model(
        {
            "dataset": {"in_channels": 3, "num_classes": 2},
            "model": {
                "name": "bdslite_unet",
                "base_channels": 32,
                "boundary_gate": True,
                "boundary_gate_scale": 0.25,
            },
        }
    )
    assert isinstance(record, BDSLiteUNet) and isinstance(public, BDSLiteUNet)
    record_params = sum(p.numel() for p in record.parameters())
    public_params = sum(p.numel() for p in public.parameters())
    assert record_params == public_params == 2_198_003


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
