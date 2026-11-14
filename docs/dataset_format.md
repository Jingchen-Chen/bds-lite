# Dataset Format

The training code expects preprocessed 2D `.npz` samples. Dataset conversion scripts live under `scripts/`.

## Path Convention

```text
data/processed/<dataset>/<split>/<sample_id>.npz
```

Examples:

```text
data/processed/synapse/train/case0001_slice0032.npz
data/processed/acdc/val/patient001_frame01_slice06.npz
data/processed/isic2018/train/ISIC_0000001.npz
```

## Required Arrays

| Key | Shape | Dtype | Notes |
| --- | --- | --- | --- |
| `image` | `[C, H, W]` | `float32` | Normalize before saving or let config define normalization. |
| `mask` | `[H, W]` | integer | Class labels, background should be `0`. |

## Optional Arrays

| Key | Shape | Dtype | Notes |
| --- | --- | --- | --- |
| `boundary` | `[1, H, W]` or `[K, H, W]` | `uint8`/`float32` | Generated from `mask`. |
| `sdf` | `[1, H, W]` or `[K, H, W]` | `float32` | Signed-distance map. |
| `spacing` | `[2]` | `float32` | Pixel spacing used for distance metrics. |

## Split Files

The directory-per-split layout is the default. If a dataset needs explicit split files, place them in:

```text
data/splits/<dataset>/<split>.txt
```

Each line should contain a relative `.npz` path from `data/processed/<dataset>/`.

## ISIC2018 Conversion

The first supported converter is for ISIC2018 Task 1 lesion segmentation.

Expected raw layout:

```text
data/raw/isic2018/
  ISIC2018_Task1-2_Training_Input/
    ISIC_0000000.jpg
  ISIC2018_Task1_Training_GroundTruth/
    ISIC_0000000_segmentation.png
```

Convert all training images:

```bash
python3 scripts/convert_isic2018.py \
  --image-dir data/raw/isic2018/ISIC2018_Task1-2_Training_Input \
  --mask-dir data/raw/isic2018/ISIC2018_Task1_Training_GroundTruth \
  --output-root data/processed/isic2018
```

Create a deterministic validation split while converting:

```bash
python3 scripts/convert_isic2018.py \
  --image-dir data/raw/isic2018/ISIC2018_Task1-2_Training_Input \
  --mask-dir data/raw/isic2018/ISIC2018_Task1_Training_GroundTruth \
  --output-root data/processed/isic2018 \
  --val-fraction 0.2 \
  --seed 2026
```

For a quick smoke test:

```bash
python3 scripts/convert_isic2018.py \
  --image-dir data/raw/isic2018/ISIC2018_Task1-2_Training_Input \
  --mask-dir data/raw/isic2018/ISIC2018_Task1_Training_GroundTruth \
  --limit 8 \
  --dry-run
```
