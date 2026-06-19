from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from bds_lite.data.converters import (
    _resize_gray,
    _resize_mask,
    convert_acdc_transunet,
    convert_isic2018,
    convert_synapse,
)
from bds_lite.data.datasets import NpzSegmentationDataset


def _write_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _write_h5(path: Path, image: np.ndarray, label: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("image", data=image)
        h5.create_dataset("label", data=label)


def test_convert_isic2018_writes_scaled_chw_image_and_binary_mask(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    output_root = tmp_path / "processed"

    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[..., 0] = 255
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 255

    _write_image(image_dir / "ISIC_0000001.jpg", image)
    _write_image(mask_dir / "ISIC_0000001_segmentation.png", mask)

    converted = convert_isic2018(image_dir, mask_dir, output_root, image_size=(8, 8))

    assert len(converted) == 1
    with np.load(output_root / "train" / "ISIC_0000001.npz") as sample:
        assert sample["image"].shape == (3, 8, 8)
        assert sample["image"].dtype == np.float32
        assert 0.0 <= float(sample["image"].min()) <= float(sample["image"].max()) <= 1.0
        assert sample["mask"].shape == (8, 8)
        assert sample["mask"].dtype == np.uint8
        assert set(np.unique(sample["mask"]).tolist()) <= {0, 1}


def test_convert_isic2018_dry_run_does_not_write(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    output_root = tmp_path / "processed"

    _write_image(image_dir / "ISIC_0000001.jpg", np.zeros((4, 4, 3), dtype=np.uint8))
    _write_image(mask_dir / "ISIC_0000001_segmentation.png", np.zeros((4, 4), dtype=np.uint8))

    converted = convert_isic2018(image_dir, mask_dir, output_root, dry_run=True)

    assert len(converted) == 1
    assert not (output_root / "train" / "ISIC_0000001.npz").exists()


def test_resize_policy_is_bilinear_for_images_and_nearest_for_masks() -> None:
    image = np.array([[0.0, 4.0], [8.0, 12.0]], dtype=np.float32)
    mask = np.array([[0, 2], [3, 4]], dtype=np.uint8)

    resized_image = _resize_gray(image, (4, 4))
    resized_mask = _resize_mask(mask, (4, 4))

    assert resized_image.shape == (4, 4)
    assert resized_image.dtype == np.float32
    assert not set(np.unique(resized_image).tolist()) <= set(np.unique(image).tolist())
    assert resized_mask.shape == (4, 4)
    assert resized_mask.dtype == np.uint8
    assert set(np.unique(resized_mask).tolist()) <= {0, 2, 3, 4}


def test_convert_acdc_transunet_preserves_input_intensity_values(tmp_path: Path) -> None:
    slices_dir = tmp_path / "acdc_h5"
    output_root = tmp_path / "processed"
    label = np.array([[0, 1], [2, 3]], dtype=np.uint8)

    for patient in ("patient001", "patient002", "patient003"):
        image = np.array([[-2.5, 0.0], [1.0, 3.5]], dtype=np.float32)
        _write_h5(slices_dir / f"{patient}_frame01_slice_0.h5", image=image, label=label)

    counts = convert_acdc_transunet(
        slices_dir,
        output_root,
        val_fraction=1 / 3,
        test_fraction=1 / 3,
        seed=2026,
    )

    assert counts == {"train": 1, "val": 1, "test": 1}
    written = sorted(output_root.glob("*/*.npz"))
    assert len(written) == 3
    with np.load(written[0]) as sample:
        np.testing.assert_array_equal(sample["mask"], label)
        assert float(sample["image"].min()) == -2.5
        assert float(sample["image"].max()) == 3.5


def test_convert_synapse_preserves_values_and_splits_cases(tmp_path: Path) -> None:
    train_dir = tmp_path / "train_npz"
    test_dir = tmp_path / "test_vol_h5"
    output_root = tmp_path / "processed"
    image = np.array([[-1.0, 0.5], [2.0, 4.0]], dtype=np.float32)
    label = np.array([[0, 1], [2, 0]], dtype=np.uint8)

    for case_id in ("case0001", "case0002"):
        train_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(train_dir / f"{case_id}_slice000.npz", image=image, label=label)
    _write_h5(
        test_dir / "case0003.h5",
        image=image[None, ...],
        label=label[None, ...],
    )

    counts = convert_synapse(
        train_dir,
        test_dir,
        output_root,
        val_fraction=0.5,
        seed=2026,
    )

    assert counts == {"train": 1, "val": 1, "test": 1}
    written = sorted(output_root.glob("*/*.npz"))
    assert len(written) == 3
    with np.load(written[0]) as sample:
        assert sample["image"].shape == (1, 2, 2)
        assert float(sample["image"].min()) == -1.0
        assert float(sample["image"].max()) == 4.0


def test_npz_dataset_loads_optional_targets_and_enforces_requirements(tmp_path: Path) -> None:
    split_dir = tmp_path / "processed" / "train"
    split_dir.mkdir(parents=True)
    np.savez_compressed(
        split_dir / "sample.npz",
        image=np.zeros((1, 2, 2), dtype=np.float32),
        mask=np.zeros((2, 2), dtype=np.uint8),
        boundary=np.ones((2, 2), dtype=np.uint8),
        sdf=np.ones((1, 2, 2), dtype=np.float32),
    )

    dataset = NpzSegmentationDataset(tmp_path / "processed", split="train", require_sdf=True)
    sample = dataset[0]

    assert sample["image"].shape == (1, 2, 2)
    assert sample["mask"].shape == (2, 2)
    assert sample["boundary"].shape == (1, 2, 2)
    assert sample["sdf"].shape == (1, 2, 2)
