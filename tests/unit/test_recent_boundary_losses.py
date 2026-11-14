from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy import ndimage as ndi

from bds_lite.losses.boundary_dou import BoundaryDoULoss
from bds_lite.losses.gsl import (
    GeneralizedSurfaceLoss,
    alpha_value,
    distance_transform_maps_numpy,
    distance_transform_maps_torch,
)


def _logits_from_mask(
    mask: torch.Tensor, num_classes: int, confidence: float = 8.0
) -> torch.Tensor:
    one_hot = torch.nn.functional.one_hot(mask.long(), num_classes=num_classes)
    one_hot = one_hot.permute(0, 3, 1, 2).float()
    return confidence * (2.0 * one_hot - 1.0)


def test_boundary_dou_perfect_overlap_is_lower_than_shift_and_miss() -> None:
    target = torch.zeros((1, 16, 16), dtype=torch.long)
    target[:, 4:12, 4:12] = 1
    shifted = torch.zeros_like(target)
    shifted[:, 5:13, 5:13] = 1
    miss = torch.zeros_like(target)

    loss_fn = BoundaryDoULoss(num_classes=2)
    perfect_loss = loss_fn(_logits_from_mask(target, 2), target)
    shifted_loss = loss_fn(_logits_from_mask(shifted, 2), target)
    miss_loss = loss_fn(_logits_from_mask(miss, 2), target)

    assert perfect_loss < shifted_loss < miss_loss
    assert torch.isfinite(miss_loss)


def test_boundary_dou_empty_foreground_does_not_nan() -> None:
    target = torch.zeros((2, 8, 8), dtype=torch.long)
    logits = torch.randn(2, 3, 8, 8)

    loss = BoundaryDoULoss(num_classes=3)(logits, target)

    assert torch.isfinite(loss)


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (2, -1.0),
        (4, -0.5),
        (8, 0.125),
        (16, 0.53125),
    ],
)
def test_boundary_dou_adaptive_alpha_matches_reference_formula(size: int, expected: float) -> None:
    target = torch.zeros((1, 24, 24), dtype=torch.float32)
    target[:, 2 : 2 + size, 2 : 2 + size] = 1.0

    alpha = BoundaryDoULoss(num_classes=2).adaptive_alpha(target)

    assert float(alpha) == pytest.approx(expected, abs=1e-4)


def test_gsl_perfect_overlap_is_lower_than_shift_and_miss() -> None:
    target = torch.zeros((1, 16, 16), dtype=torch.long)
    target[:, 4:12, 4:12] = 1
    shifted = torch.zeros_like(target)
    shifted[:, 5:13, 5:13] = 1
    miss = torch.zeros_like(target)
    dtm = distance_transform_maps_torch(target, num_classes=2)

    loss_fn = GeneralizedSurfaceLoss(num_classes=2, total_epochs=10, alpha_schedule="step5")
    loss_fn.set_epoch(10)
    perfect_loss, _ = loss_fn(_logits_from_mask(target, 2), target, dtm)
    shifted_loss, _ = loss_fn(_logits_from_mask(shifted, 2), target, dtm)
    miss_loss, _ = loss_fn(_logits_from_mask(miss, 2), target, dtm)

    assert perfect_loss < shifted_loss < miss_loss


def test_gsl_empty_foreground_does_not_nan() -> None:
    target = torch.zeros((1, 8, 8), dtype=torch.long)
    logits = torch.randn(1, 3, 8, 8)
    loss, parts = GeneralizedSurfaceLoss(num_classes=3, total_epochs=5)(logits, target)

    assert torch.isfinite(loss)
    assert torch.isfinite(parts["gsl_boundary"])


def test_gsl_dtm_matches_scipy_reference() -> None:
    target = np.zeros((1, 7, 7), dtype=np.int64)
    target[:, 2:5, 3:6] = 1
    expected = np.stack(
        [
            ndi.distance_transform_edt(~(target[0] == class_id)).astype(np.float32)
            for class_id in range(2)
        ],
        axis=0,
    )[None]

    np.testing.assert_allclose(distance_transform_maps_numpy(target, 2), expected, atol=1e-4)
    actual = distance_transform_maps_torch(torch.from_numpy(target), 2).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-4)


def test_gsl_alpha_schedule_values() -> None:
    assert alpha_value(1, 20, "step5") == pytest.approx(1.0)
    assert alpha_value(20, 20, "step5") == pytest.approx(0.0)
    assert alpha_value(1, 20, "linear") == pytest.approx(1.0)
    assert alpha_value(20, 20, "linear") == pytest.approx(0.0)


def test_gsl_rejects_wrong_class_count() -> None:
    target = torch.zeros((1, 8, 8), dtype=torch.long)
    logits = torch.randn(1, 2, 8, 8)
    loss_fn = GeneralizedSurfaceLoss(num_classes=3, total_epochs=5)

    with pytest.raises(ValueError, match="expected 3 channels"):
        loss_fn(logits, target)
