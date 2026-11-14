from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def dice_score(pred: np.ndarray, target: np.ndarray, label: int, eps: float = 1e-6) -> float:
    pred_bin = np.asarray(pred) == label
    target_bin = np.asarray(target) == label
    intersection = np.logical_and(pred_bin, target_bin).sum()
    denominator = pred_bin.sum() + target_bin.sum()
    return float((2.0 * intersection + eps) / (denominator + eps))


def iou_score(pred: np.ndarray, target: np.ndarray, label: int, eps: float = 1e-6) -> float:
    pred_bin = np.asarray(pred) == label
    target_bin = np.asarray(target) == label
    intersection = np.logical_and(pred_bin, target_bin).sum()
    union = np.logical_or(pred_bin, target_bin).sum()
    return float((intersection + eps) / (union + eps))


def surface(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = ndi.binary_erosion(mask)
    return np.logical_xor(mask, eroded)


def surface_distances(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, float] | None = None,
) -> np.ndarray:
    pred_surface = surface(pred)
    target_surface = surface(target)
    if not pred_surface.any() or not target_surface.any():
        return np.asarray([], dtype=np.float32)

    sampling = spacing if spacing is not None else None
    target_distance = ndi.distance_transform_edt(~target_surface, sampling=sampling)
    pred_distance = ndi.distance_transform_edt(~pred_surface, sampling=sampling)
    distances = np.concatenate(
        [
            target_distance[pred_surface],
            pred_distance[target_surface],
        ]
    )
    return distances.astype(np.float32)


def hd95(
    pred: np.ndarray,
    target: np.ndarray,
    label: int = 1,
    spacing: tuple[float, float] | None = None,
) -> float:
    distances = surface_distances(np.asarray(pred) == label, np.asarray(target) == label, spacing)
    if distances.size == 0:
        return float("nan")
    return float(np.percentile(distances, 95))


def assd(
    pred: np.ndarray,
    target: np.ndarray,
    label: int = 1,
    spacing: tuple[float, float] | None = None,
) -> float:
    distances = surface_distances(np.asarray(pred) == label, np.asarray(target) == label, spacing)
    if distances.size == 0:
        return float("nan")
    return float(np.mean(distances))


def boundary_f1(
    pred_boundary: np.ndarray,
    target_boundary: np.ndarray,
    tolerance: int = 2,
    eps: float = 1e-6,
) -> float:
    pred_boundary = np.asarray(pred_boundary).astype(bool)
    target_boundary = np.asarray(target_boundary).astype(bool)
    if np.array_equal(pred_boundary, target_boundary):
        return 1.0
    if not pred_boundary.any() and not target_boundary.any():
        return 1.0
    if not pred_boundary.any() or not target_boundary.any():
        return 0.0

    pred_match_zone = ndi.binary_dilation(pred_boundary, iterations=tolerance)
    target_match_zone = ndi.binary_dilation(target_boundary, iterations=tolerance)
    precision = np.logical_and(pred_boundary, target_match_zone).sum() / (pred_boundary.sum() + eps)
    recall = np.logical_and(target_boundary, pred_match_zone).sum() / (target_boundary.sum() + eps)
    denom = precision + recall
    if denom <= 0.0:
        return 0.0
    return float(2.0 * precision * recall / denom)
