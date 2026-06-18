import pytest
import torch

from bds_lite.training.losses import BDSLiteLoss, LossWeights, weights_from_config


def test_surface_loss_contributes_when_sdf_is_present() -> None:
    outputs = {
        "seg_logits": torch.randn(2, 3, 8, 8),
    }
    batch = {
        "mask": torch.randint(0, 3, (2, 8, 8)),
        "sdf": torch.randn(2, 2, 8, 8),
    }
    loss_fn = BDSLiteLoss(LossWeights(dice=1.0, ce=1.0, surface=0.01))

    loss, scalars = loss_fn(outputs, batch)

    assert torch.isfinite(loss)
    assert "surface" in scalars


def test_surface_loss_ignores_absent_sdf_channels() -> None:
    outputs = {
        "seg_logits": torch.randn(1, 3, 8, 8),
    }
    batch = {
        "mask": torch.zeros((1, 8, 8), dtype=torch.long),
        "sdf": torch.zeros(1, 2, 8, 8),
    }
    loss_fn = BDSLiteLoss(LossWeights(dice=0.0, ce=0.0, surface=1.0))

    loss, scalars = loss_fn(outputs, batch)

    assert torch.isfinite(loss)
    assert scalars["surface"] == 0.0


def test_distance_error_loss_contributes_when_sdf_is_present() -> None:
    outputs = {
        "seg_logits": torch.randn(2, 3, 8, 8),
    }
    batch = {
        "mask": torch.randint(0, 3, (2, 8, 8)),
        "sdf": torch.randn(2, 2, 8, 8),
    }
    loss_fn = BDSLiteLoss(LossWeights(dice=1.0, ce=1.0, distance=0.2))

    loss, scalars = loss_fn(outputs, batch)

    assert torch.isfinite(loss)
    assert "distance" in scalars
    assert scalars["distance"] >= 0.0


def test_distance_error_loss_ignores_absent_sdf_channels() -> None:
    outputs = {
        "seg_logits": torch.randn(1, 3, 8, 8),
    }
    batch = {
        "mask": torch.zeros((1, 8, 8), dtype=torch.long),
        "sdf": torch.zeros(1, 2, 8, 8),
    }
    loss_fn = BDSLiteLoss(LossWeights(dice=0.0, ce=0.0, distance=1.0))

    loss, scalars = loss_fn(outputs, batch)

    assert torch.isfinite(loss)
    assert scalars["distance"] == 0.0


def test_far_false_positive_loss_contributes_when_sdf_is_present() -> None:
    outputs = {
        "seg_logits": torch.randn(2, 3, 8, 8),
    }
    batch = {
        "mask": torch.randint(0, 3, (2, 8, 8)),
        "sdf": torch.randn(2, 2, 8, 8),
    }
    loss_fn = BDSLiteLoss(LossWeights(dice=1.0, ce=1.0, far_fp=0.2))

    loss, scalars = loss_fn(outputs, batch)

    assert torch.isfinite(loss)
    assert "far_fp" in scalars
    assert scalars["far_fp"] >= 0.0


def test_far_false_positive_loss_ignores_absent_sdf_channels() -> None:
    outputs = {
        "seg_logits": torch.randn(1, 3, 8, 8),
    }
    batch = {
        "mask": torch.zeros((1, 8, 8), dtype=torch.long),
        "sdf": torch.zeros(1, 2, 8, 8),
    }
    loss_fn = BDSLiteLoss(LossWeights(dice=0.0, ce=0.0, far_fp=1.0))

    loss, scalars = loss_fn(outputs, batch)

    assert torch.isfinite(loss)
    assert scalars["far_fp"] == 0.0


def test_foreground_class_weights_are_parsed_from_config() -> None:
    weights = weights_from_config(
        {
            "loss": {
                "dice_weight": 1.0,
                "far_fp_weight": 0.15,
                "foreground_class_weights": [1.0, 2.5],
            }
        }
    )

    assert weights.foreground_class_weights == (1.0, 2.5)
    assert weights.far_fp == 0.15


def test_foreground_class_weights_apply_to_composite_loss() -> None:
    outputs = {
        "seg_logits": torch.randn(2, 3, 8, 8),
    }
    batch = {
        "mask": torch.randint(0, 3, (2, 8, 8)),
        "sdf": torch.randn(2, 2, 8, 8),
    }
    loss_fn = BDSLiteLoss(
        LossWeights(
            dice=1.0,
            ce=0.0,
            surface=1.0,
            distance=1.0,
            far_fp=1.0,
            foreground_class_weights=(1.0, 3.0),
        )
    )

    loss, scalars = loss_fn(outputs, batch)

    assert torch.isfinite(loss)
    assert {"dice", "surface", "distance", "far_fp"}.issubset(scalars)


def test_foreground_class_weights_validate_length() -> None:
    outputs = {
        "seg_logits": torch.randn(2, 3, 8, 8),
    }
    batch = {
        "mask": torch.randint(0, 3, (2, 8, 8)),
    }
    loss_fn = BDSLiteLoss(
        LossWeights(
            dice=1.0,
            ce=0.0,
            foreground_class_weights=(1.0,),
        )
    )

    with pytest.raises(ValueError, match="one value per foreground class"):
        loss_fn(outputs, batch)
