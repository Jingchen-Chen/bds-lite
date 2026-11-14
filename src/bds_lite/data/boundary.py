from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy import ndimage as ndi


def _valid_labels(mask: np.ndarray, ignore_index: int | None = None) -> list[int]:
    labels = [int(label) for label in np.unique(mask)]
    return [label for label in labels if label != 0 and label != ignore_index]


def mask_to_boundary(
    mask: np.ndarray,
    radius: int = 1,
    ignore_index: int | None = None,
) -> np.ndarray:
    """Generate a binary boundary map from an integer segmentation mask."""
    if radius < 1:
        raise ValueError("radius must be >= 1")

    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"expected a 2D mask, got shape {mask.shape}")

    boundary = np.zeros(mask.shape, dtype=bool)
    for label in _valid_labels(mask, ignore_index):
        region = mask == label
        if not region.any():
            continue
        dilated = ndi.binary_dilation(region, iterations=radius)
        eroded = ndi.binary_erosion(region, iterations=radius)
        boundary |= np.logical_xor(dilated, eroded)
    return boundary.astype(np.uint8)


def mask_to_one_hot_boundary(
    mask: np.ndarray,
    num_classes: int,
    radius: int = 1,
    include_background: bool = False,
    ignore_index: int | None = None,
) -> np.ndarray:
    """Generate class-wise boundary targets with shape [K, H, W]."""
    mask = np.asarray(mask)
    labels: Iterable[int] = range(num_classes) if include_background else range(1, num_classes)
    channels: list[np.ndarray] = []
    for label in labels:
        if label == ignore_index:
            continue
        channels.append(mask_to_boundary((mask == label).astype(np.uint8), radius=radius))

    if not channels:
        return np.zeros((0, *mask.shape), dtype=np.uint8)
    return np.stack(channels, axis=0).astype(np.uint8)


def binary_to_sdf(
    binary_mask: np.ndarray, spacing: tuple[float, float] | None = None
) -> np.ndarray:
    """Return signed distance where negative values are inside the foreground."""
    binary = np.asarray(binary_mask).astype(bool)
    if binary.ndim != 2:
        raise ValueError(f"expected a 2D binary mask, got shape {binary.shape}")

    if not binary.any():
        return np.zeros(binary.shape, dtype=np.float32)
    if binary.all():
        return -np.ones(binary.shape, dtype=np.float32)

    sampling = spacing if spacing is not None else None
    outside = ndi.distance_transform_edt(~binary, sampling=sampling)
    inside = ndi.distance_transform_edt(binary, sampling=sampling)
    return (outside - inside).astype(np.float32)


def mask_to_sdf(
    mask: np.ndarray,
    num_classes: int | None = None,
    spacing: tuple[float, float] | None = None,
    include_background: bool = False,
    ignore_index: int | None = None,
) -> np.ndarray:
    """Generate one signed-distance map per foreground class."""
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"expected a 2D mask, got shape {mask.shape}")

    if num_classes is None:
        labels = _valid_labels(mask, ignore_index)
    else:
        labels = list(range(num_classes)) if include_background else list(range(1, num_classes))
        labels = [label for label in labels if label != ignore_index]

    maps = [binary_to_sdf(mask == label, spacing=spacing) for label in labels]
    if not maps:
        return np.zeros((0, *mask.shape), dtype=np.float32)
    return np.stack(maps, axis=0).astype(np.float32)
