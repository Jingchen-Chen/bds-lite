import numpy as np

from bds_lite.data.boundary import mask_to_boundary, mask_to_sdf


def test_mask_to_boundary_returns_binary_map() -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 1

    boundary = mask_to_boundary(mask, radius=1)

    assert boundary.shape == mask.shape
    assert boundary.dtype == np.uint8
    assert boundary.max() == 1
    assert boundary.sum() > 0


def test_mask_to_sdf_returns_foreground_channel() -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 1

    sdf = mask_to_sdf(mask)

    assert sdf.shape == (1, 8, 8)
    assert sdf.dtype == np.float32
    assert sdf[0, 3, 3] < 0
    assert sdf[0, 0, 0] > 0
