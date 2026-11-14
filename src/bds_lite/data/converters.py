from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ConvertedSample:
    sample_id: str
    image_path: Path
    mask_path: Path
    output_path: Path


def load_rgb_image(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size is not None:
        image = image.resize(size, Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1)).astype(np.float32)


def load_binary_mask(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    mask = Image.open(path).convert("L")
    if size is not None:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    return (np.asarray(mask) > 0).astype(np.uint8)


def save_npz_sample(
    output_path: str | Path,
    image: np.ndarray,
    mask: np.ndarray,
    sample_id: str | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "image": image.astype(np.float32),
        "mask": mask.astype(np.uint8),
    }
    if sample_id is not None:
        payload["sample_id"] = np.asarray(sample_id)
    np.savez_compressed(output_path, **payload)


def find_isic_mask(image_path: Path, mask_dir: Path) -> Path:
    candidates = [
        mask_dir / f"{image_path.stem}_segmentation.png",
        mask_dir / f"{image_path.stem}.png",
        mask_dir / f"{image_path.stem}_Segmentation.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no mask found for {image_path.name} in {mask_dir}")


def collect_isic_pairs(image_dir: str | Path, mask_dir: str | Path) -> list[tuple[Path, Path]]:
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    image_paths = sorted(
        path
        for suffix in ("*.jpg", "*.jpeg", "*.png")
        for path in image_dir.glob(suffix)
        if path.is_file()
    )
    if not image_paths:
        raise FileNotFoundError(f"no images found under {image_dir}")
    return [(image_path, find_isic_mask(image_path, mask_dir)) for image_path in image_paths]


def split_pairs(
    pairs: list[tuple[Path, Path]],
    val_fraction: float,
    seed: int,
) -> dict[str, list[tuple[Path, Path]]]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must satisfy 0 <= val_fraction < 1")

    if val_fraction == 0.0:
        return {"train": pairs}

    rng = np.random.default_rng(seed)
    indices = np.arange(len(pairs))
    rng.shuffle(indices)
    val_count = max(1, int(round(len(pairs) * val_fraction)))
    val_indices = set(indices[:val_count].tolist())
    train = [pair for idx, pair in enumerate(pairs) if idx not in val_indices]
    val = [pair for idx, pair in enumerate(pairs) if idx in val_indices]
    return {"train": train, "val": val}


def convert_isic2018(
    image_dir: str | Path,
    mask_dir: str | Path,
    output_root: str | Path,
    split: str = "train",
    image_size: tuple[int, int] | None = None,
    val_fraction: float = 0.0,
    seed: int = 2026,
    limit: int | None = None,
    dry_run: bool = False,
) -> list[ConvertedSample]:
    pairs = collect_isic_pairs(image_dir, mask_dir)
    if limit is not None:
        pairs = pairs[:limit]

    split_map = split_pairs(pairs, val_fraction=val_fraction, seed=seed)
    if split != "train" or val_fraction == 0.0:
        split_map = {split: pairs}

    converted: list[ConvertedSample] = []
    output_root = Path(output_root)
    for split_name, split_pairs_ in split_map.items():
        for image_path, mask_path in split_pairs_:
            output_path = output_root / split_name / f"{image_path.stem}.npz"
            sample = ConvertedSample(
                sample_id=image_path.stem,
                image_path=image_path,
                mask_path=mask_path,
                output_path=output_path,
            )
            converted.append(sample)
            if dry_run:
                continue

            image = load_rgb_image(image_path, size=image_size)
            mask = load_binary_mask(mask_path, size=image_size)
            save_npz_sample(output_path, image=image, mask=mask, sample_id=image_path.stem)

    return converted


# ---------------------------------------------------------------------------
# Synapse multi-organ CT
# ---------------------------------------------------------------------------


def _resize_gray(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a 2D float32 array with bilinear interpolation; returns float32."""
    img = Image.fromarray(array.astype(np.float32))
    img = img.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def _resize_mask(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a 2D integer label map with nearest-neighbour interpolation."""
    img = Image.fromarray(array.astype(np.uint8))
    img = img.resize(size, Image.Resampling.NEAREST)
    return np.asarray(img, dtype=np.uint8)


def _synapse_case_id(filename: str) -> str:
    m = re.match(r"(case\d+)", filename)
    if m is None:
        raise ValueError(f"cannot parse case id from {filename!r}")
    return m.group(1)


def _split_cases(
    cases: list[str],
    val_fraction: float,
    seed: int,
) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    shuffled = list(cases)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_fraction)))
    return {
        "train": shuffled[val_count:],
        "val": shuffled[:val_count],
    }


def convert_synapse(
    train_npz_dir: str | Path,
    test_h5_dir: str | Path,
    output_root: str | Path,
    image_size: tuple[int, int] | None = None,
    val_fraction: float = 0.2,
    seed: int = 2026,
    skip_background_only: bool = False,
    dry_run: bool = False,
) -> dict[str, int]:
    """Convert Synapse data (HiFormer layout) to BDS-Lite NPZ format.

    Reads 2D training slices from ``train_npz_dir`` and 3D test volumes from
    ``test_h5_dir``.  Slices are split at the *case* level (not slice level) so
    no volume leaks into validation.  Returns a ``{split: count}`` dict.
    """
    import h5py

    train_npz_dir = Path(train_npz_dir)
    test_h5_dir = Path(test_h5_dir)
    output_root = Path(output_root)

    # Collect training cases
    train_files = sorted(train_npz_dir.glob("*.npz"))
    if not train_files:
        raise FileNotFoundError(f"no .npz files found under {train_npz_dir}")

    all_cases = sorted({_synapse_case_id(f.name) for f in train_files})
    case_split = _split_cases(all_cases, val_fraction=val_fraction, seed=seed)
    counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}

    # Train + val slices from pre-sliced NPZ files
    for split_name, case_ids in case_split.items():
        case_set = set(case_ids)
        split_files = [f for f in train_files if _synapse_case_id(f.name) in case_set]
        for f in split_files:
            with np.load(f) as data:
                image = data["image"].astype(np.float32)  # [H, W]
                mask = data["label"].astype(np.uint8)  # [H, W], float->int

            if skip_background_only and mask.max() == 0:
                continue

            if image_size is not None:
                image = _resize_gray(image, image_size)
                mask = _resize_mask(mask, image_size)

            image_chw = image[None, ...]  # [1, H, W]
            sample_id = f.stem

            output_path = output_root / split_name / f"{sample_id}.npz"
            counts[split_name] += 1
            if not dry_run:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    output_path,
                    image=image_chw,
                    mask=mask,
                    sample_id=np.asarray(sample_id),
                )

    # Test split from 3D H5 volumes (sliced along Z)
    test_h5_files = sorted(test_h5_dir.glob("*.h5"))
    if not test_h5_files:
        raise FileNotFoundError(f"no .h5 files found under {test_h5_dir}")

    for h5_path in test_h5_files:
        with h5py.File(h5_path, "r") as h:
            volume = h["image"][:]  # [Z, H, W] float32, normalised 0-1
            labels = h["label"][:]  # [Z, H, W] uint8

        case_id = h5_path.stem.replace(".npy", "")
        for z in range(volume.shape[0]):
            image_slice = volume[z].astype(np.float32)
            mask_slice = labels[z].astype(np.uint8)

            if skip_background_only and mask_slice.max() == 0:
                continue

            if image_size is not None:
                image_slice = _resize_gray(image_slice, image_size)
                mask_slice = _resize_mask(mask_slice, image_size)

            image_chw = image_slice[None, ...]
            sample_id = f"{case_id}_slice{z:03d}"

            output_path = output_root / "test" / f"{sample_id}.npz"
            counts["test"] += 1
            if not dry_run:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    output_path,
                    image=image_chw,
                    mask=mask_slice,
                    sample_id=np.asarray(sample_id),
                )

    return counts


# ---------------------------------------------------------------------------
# ACDC cardiac MRI
# ---------------------------------------------------------------------------


def _normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Per-volume min-max normalisation to [0, 1]."""
    vmin, vmax = float(volume.min()), float(volume.max())
    if vmax - vmin < 1e-8:
        return np.zeros_like(volume, dtype=np.float32)
    return ((volume - vmin) / (vmax - vmin)).astype(np.float32)


def _collect_acdc_frames(patient_dir: Path) -> list[tuple[Path, Path]]:
    """Return [(image_path, gt_path), ...] for all annotated frames in patient_dir."""
    pairs: list[tuple[Path, Path]] = []
    for gt_path in sorted(patient_dir.glob("*_gt.nii.gz")):
        stem = gt_path.name.replace("_gt.nii.gz", "")
        img_path = patient_dir / f"{stem}.nii.gz"
        if img_path.exists():
            pairs.append((img_path, gt_path))
    return pairs


def convert_acdc(
    src_root: str | Path,
    output_root: str | Path,
    image_size: tuple[int, int] | None = None,
    val_fraction: float = 0.1,
    test_fraction: float = 0.2,
    seed: int = 2026,
    skip_background_only: bool = False,
    dry_run: bool = False,
) -> dict[str, int]:
    """Convert ACDC cardiac MRI data to BDS-Lite NPZ format.

    Expects the standard ACDC directory layout::

        <src_root>/
          patient001/
            patient001_frame01.nii.gz
            patient001_frame01_gt.nii.gz
            ...
          patient002/
            ...

    Patients are split at the *patient* level (train/val/test) before slicing
    so no volume leaks across splits.  Labels: 0=BG, 1=RV, 2=MYO, 3=LV.
    """
    import nibabel as nib

    src_root = Path(src_root)
    output_root = Path(output_root)

    patient_dirs = sorted(d for d in src_root.iterdir() if d.is_dir() and "patient" in d.name)
    if not patient_dirs:
        raise FileNotFoundError(f"no patient directories found under {src_root}")

    rng = np.random.default_rng(seed)
    shuffled = list(patient_dirs)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_test = max(1, int(round(n * test_fraction)))
    n_val = max(1, int(round(n * val_fraction)))
    split_map: dict[str, list[Path]] = {
        "test": shuffled[:n_test],
        "val": shuffled[n_test : n_test + n_val],
        "train": shuffled[n_test + n_val :],
    }

    counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}

    for split_name, patients in split_map.items():
        for patient_dir in patients:
            pairs = _collect_acdc_frames(patient_dir)
            if not pairs:
                continue
            for img_path, gt_path in pairs:
                img_nib = nib.load(str(img_path))
                gt_nib = nib.load(str(gt_path))
                volume = _normalize_volume(img_nib.get_fdata(dtype=np.float32))
                labels = gt_nib.get_fdata().astype(np.uint8)

                # NIfTI is [H, W, Z] — iterate over last axis
                for z in range(volume.shape[2]):
                    image_slice = volume[:, :, z]
                    mask_slice = labels[:, :, z]

                    if skip_background_only and mask_slice.max() == 0:
                        continue

                    if image_size is not None:
                        image_slice = _resize_gray(image_slice, image_size)
                        mask_slice = _resize_mask(mask_slice, image_size)

                    image_chw = image_slice[None, ...]
                    frame_stem = img_path.name.replace(".nii.gz", "")
                    sample_id = f"{frame_stem}_slice{z:03d}"

                    output_path = output_root / split_name / f"{sample_id}.npz"
                    counts[split_name] += 1
                    if not dry_run:
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(
                            output_path,
                            image=image_chw,
                            mask=mask_slice,
                            sample_id=np.asarray(sample_id),
                        )

    return counts


# ---------------------------------------------------------------------------
# ACDC cardiac MRI — TransUNet pre-sliced H5 format
# ---------------------------------------------------------------------------


def _acdc_patient_id(filename: str) -> str:
    m = re.match(r"(patient\d+)", filename)
    if m is None:
        raise ValueError(f"cannot parse patient id from {filename!r}")
    return m.group(1)


def convert_acdc_transunet(
    slices_dir: str | Path,
    output_root: str | Path,
    image_size: tuple[int, int] | None = None,
    val_fraction: float = 0.1,
    test_fraction: float = 0.2,
    seed: int = 2026,
    skip_background_only: bool = False,
    dry_run: bool = False,
) -> dict[str, int]:
    """Convert TransUNet-style pre-sliced ACDC H5 files to BDS-Lite NPZ format.

    Expects a flat directory of ``patient{NNN}_frame{NN}_slice_{N}.h5`` files,
    each containing ``image`` [H, W] float32 (z-score normalised) and
    ``label`` [H, W] uint8.  Patients are split at the *patient* level.
    Labels: 0=BG, 1=RV, 2=MYO, 3=LV.
    """
    import h5py

    slices_dir = Path(slices_dir)
    output_root = Path(output_root)

    h5_files = sorted(slices_dir.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"no .h5 files found under {slices_dir}")

    all_patients = sorted({_acdc_patient_id(f.name) for f in h5_files})
    rng = np.random.default_rng(seed)
    shuffled = list(all_patients)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_test = max(1, int(round(n * test_fraction)))
    n_val = max(1, int(round(n * val_fraction)))
    split_map: dict[str, set[str]] = {
        "test": set(shuffled[:n_test]),
        "val": set(shuffled[n_test : n_test + n_val]),
        "train": set(shuffled[n_test + n_val :]),
    }
    patient_to_split = {p: s for s, ps in split_map.items() for p in ps}

    counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}

    for h5_path in h5_files:
        patient_id = _acdc_patient_id(h5_path.name)
        split_name = patient_to_split[patient_id]

        with h5py.File(h5_path, "r") as h:
            image = h["image"][:].astype(np.float32)  # [H, W], z-score normalised
            mask = h["label"][:].astype(np.uint8)  # [H, W]

        if skip_background_only and mask.max() == 0:
            continue

        if image_size is not None:
            image = _resize_gray(image, image_size)
            mask = _resize_mask(mask, image_size)

        image_chw = image[None, ...]
        sample_id = h5_path.stem

        output_path = output_root / split_name / f"{sample_id}.npz"
        counts[split_name] += 1
        if not dry_run:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output_path,
                image=image_chw,
                mask=mask,
                sample_id=np.asarray(sample_id),
            )

    return counts
