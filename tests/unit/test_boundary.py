import numpy as np

from bds_lite.data.boundary import (
    binary_to_sdf,
    mask_to_boundary,
    mask_to_one_hot_boundary,
    mask_to_sdf,
)


def test_mask_to_boundary_returns_binary_map() -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 1

    boundary = mask_to_boundary(mask, radius=1)

    assert boundary.shape == mask.shape
    assert boundary.dtype == np.uint8
    assert boundary.max() == 1
    assert boundary.sum() > 0


def test_mask_to_boundary_matches_binary_fixture() -> None:
    mask = np.pad(np.ones((3, 3), dtype=np.uint8), 1)

    boundary = mask_to_boundary(mask, radius=1)

    expected = np.array(
        [
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(boundary, expected)


def test_mask_to_one_hot_boundary_matches_multiclass_fixture() -> None:
    mask = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 2, 2, 0],
            [0, 0, 2, 2, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    boundary = mask_to_one_hot_boundary(mask, num_classes=3, radius=1)

    expected = np.array(
        [
            [
                [0, 1, 1, 0, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0],
                [0, 1, 1, 1, 1],
                [0, 1, 1, 1, 1],
                [0, 0, 1, 1, 0],
            ],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(boundary, expected)


def test_mask_to_sdf_returns_foreground_channel() -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 1

    sdf = mask_to_sdf(mask)

    assert sdf.shape == (1, 8, 8)
    assert sdf.dtype == np.float32
    assert sdf[0, 3, 3] < 0
    assert sdf[0, 0, 0] > 0


def test_mask_to_sdf_sign_unclipped_distance_and_absent_class_behavior() -> None:
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[2, 2] = 1

    sdf = mask_to_sdf(mask, num_classes=3)

    assert sdf.shape == (2, 5, 5)
    assert sdf[0, 2, 2] == -1.0
    assert sdf[0, 0, 0] > 1.0
    np.testing.assert_array_equal(sdf[1], np.zeros((5, 5), dtype=np.float32))


def test_binary_to_sdf_empty_and_all_foreground_behavior() -> None:
    empty = np.zeros((4, 4), dtype=np.uint8)
    full = np.ones((4, 4), dtype=np.uint8)

    np.testing.assert_array_equal(binary_to_sdf(empty), np.zeros((4, 4), dtype=np.float32))
    np.testing.assert_array_equal(binary_to_sdf(full), -np.ones((4, 4), dtype=np.float32))
