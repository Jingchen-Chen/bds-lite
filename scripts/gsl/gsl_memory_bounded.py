"""Isolated memory-bounded GSL distance-transform wrapper."""

from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage as ndi


def distance_transform_maps_scipy(
    target: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Compute the tracked GSL classwise Euclidean DTM on CPU in linear memory."""
    if target.ndim != 3:
        raise ValueError(f"expected target with shape [N,H,W], got {tuple(target.shape)}")
    target_array = target.detach().cpu().numpy()
    batch, height, width = target_array.shape
    output = np.empty(
        (batch, int(num_classes), height, width),
        dtype=np.float32,
    )
    diagonal = float(np.sqrt(height**2 + width**2))
    for sample_index, sample in enumerate(target_array):
        for class_id in range(int(num_classes)):
            mask = sample == class_id
            if mask.any():
                output[sample_index, class_id] = ndi.distance_transform_edt(~mask).astype(
                    np.float32
                )
            else:
                output[sample_index, class_id].fill(diagonal)
    return torch.from_numpy(output).to(device=target.device)


def install_memory_bounded_gsl() -> None:
    """Patch only the current Python process; tracked source remains unchanged."""
    import bds_lite.losses.gsl as gsl_module

    gsl_module.distance_transform_maps_torch = distance_transform_maps_scipy
