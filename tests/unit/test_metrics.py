import math

import numpy as np

from bds_lite.evaluation.metrics import assd, boundary_f1, dice_score, hd95, iou_score


def test_overlap_metrics_are_one_for_perfect_prediction() -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 1

    assert dice_score(mask, mask, label=1) == 1.0
    assert iou_score(mask, mask, label=1) == 1.0


def test_boundary_f1_is_one_for_matching_boundaries() -> None:
    boundary = np.zeros((8, 8), dtype=np.uint8)
    boundary[2, 2:6] = 1
    boundary[5, 2:6] = 1
    boundary[2:6, 2] = 1
    boundary[2:6, 5] = 1

    assert boundary_f1(boundary, boundary) == 1.0


def test_surface_distance_metrics_are_nan_for_empty_prediction() -> None:
    pred = np.zeros((8, 8), dtype=np.uint8)
    target = np.zeros((8, 8), dtype=np.uint8)
    target[2:6, 2:6] = 1

    assert math.isnan(hd95(pred, target, label=1))
    assert math.isnan(assd(pred, target, label=1))


def test_dice_iou_match_analytic_on_shifted_square() -> None:
    target = np.zeros((100, 100), dtype=np.uint8)
    target[30:70, 30:70] = 1
    pred = np.zeros_like(target)
    pred[30:70, 31:71] = 1

    assert abs(dice_score(pred, target, 1) - 0.975) < 1e-5
    assert abs(iou_score(pred, target, 1) - 0.9512195) < 1e-5


def test_hd95_assd_are_one_pixel_for_unit_shift_square() -> None:
    target = np.zeros((100, 100), dtype=np.uint8)
    target[30:70, 30:70] = 1
    pred = np.zeros_like(target)
    pred[30:70, 31:71] = 1

    assert abs(hd95(pred, target, 1) - 1.0) < 1e-6
    assert assd(pred, target, 1) > 0.0
    assert assd(pred, target, 1) <= 1.0


def test_assd_is_sensitive_to_distant_false_positive_blob() -> None:
    """ASSD must move noticeably when a remote false positive is added.

    This guards against any future refactor that would silently switch to a
    one-sided or median-based distance and quietly destroy the manuscript's
    surface-distance comparisons.
    """
    target = np.zeros((100, 100), dtype=np.uint8)
    target[30:70, 30:70] = 1
    pred = np.zeros_like(target)
    pred[30:70, 31:71] = 1
    baseline = assd(pred, target, 1)

    pred_with_blob = pred.copy()
    pred_with_blob[5:8, 5:8] = 1
    perturbed = assd(pred_with_blob, target, 1)

    assert perturbed - baseline > 0.5


def test_boundary_f1_is_zero_when_boundaries_are_disjoint_within_tolerance() -> None:
    """A far prediction should not be credited as a near-boundary match."""
    target = np.zeros((100, 100), dtype=np.uint8)
    target[10:20, 10:20] = 1
    pred = np.zeros_like(target)
    pred[60:80, 60:80] = 1

    from bds_lite.data.boundary import mask_to_boundary

    pb = mask_to_boundary(pred, radius=1)
    tb = mask_to_boundary(target, radius=1)
    assert boundary_f1(pb, tb, tolerance=2) < 0.05


def test_metrics_treat_per_label_extraction_consistently() -> None:
    """Multi-class label extraction matches the contract used by evaluate.py."""
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:10, 2:10] = 1
    mask[10:18, 10:18] = 2
    pred = mask.copy()
    pred[5, 5] = 0  # one false negative on label 1

    assert dice_score(pred, mask, label=2) == 1.0  # untouched
    assert dice_score(pred, mask, label=1) < 1.0  # degraded
