from pathlib import Path

import numpy as np
from PIL import Image

from bds_lite.data.converters import convert_isic2018


def write_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def test_convert_isic2018_writes_npz_samples(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    output_root = tmp_path / "processed"

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[..., 0] = 255
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255

    write_image(image_dir / "ISIC_0000001.jpg", image)
    write_image(mask_dir / "ISIC_0000001_segmentation.png", mask)

    converted = convert_isic2018(image_dir, mask_dir, output_root)

    assert len(converted) == 1
    output_path = output_root / "train" / "ISIC_0000001.npz"
    assert output_path.exists()
    with np.load(output_path) as sample:
        assert sample["image"].shape == (3, 8, 8)
        assert sample["image"].dtype == np.float32
        assert sample["mask"].shape == (8, 8)
        assert sample["mask"].dtype == np.uint8
        assert sample["mask"].max() == 1


def test_convert_isic2018_dry_run_does_not_write(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    output_root = tmp_path / "processed"

    write_image(image_dir / "ISIC_0000001.jpg", np.zeros((4, 4, 3), dtype=np.uint8))
    write_image(mask_dir / "ISIC_0000001_segmentation.png", np.zeros((4, 4), dtype=np.uint8))

    converted = convert_isic2018(image_dir, mask_dir, output_root, dry_run=True)

    assert len(converted) == 1
    assert not (output_root / "train" / "ISIC_0000001.npz").exists()
