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

The basic `scripts/train.py` and `scripts/evaluate.py` interfaces read every `.npz` file in
the selected split directory. They do not read an external split list.

Paper experiments use JSON manifests containing `sample_ids` and split metadata, checked by
their study-specific training adapters. The archived manifests can be extracted to:

```text
recovered/followup/splits/<dataset>/<split>.json
```

See the [reproduction guide](reproducing.md#archived-inputs) for extraction and the
[protocol index](../protocols/README.md) for each study's preprocessing and split requirements.
The converters' random split options below are for new experiments; reproducing the paper requires
its registered split membership.

## ACDC Conversion

Convert an official NIfTI training release containing `patient001/`, `patient002/`, etc.:

```bash
python scripts/convert_acdc.py \
  --format nifti \
  --src data/raw/acdc/training \
  --output-root data/processed/acdc \
  --image-size 224x224 \
  --val-fraction 0.1 --test-fraction 0.2 --seed 2026
```

This creates patient-level train/validation/test splits for the basic training example in the
README. It does not prepare the separate official-test evaluation of the paper.

For a configuration that requires boundary or signed-distance targets, generate them after conversion:

```bash
python scripts/prepare_boundary_targets.py --dataset acdc --split train --num-classes 4
```

This command adds `boundary` and `sdf` arrays to the training samples in place.
Use `python scripts/convert_synapse.py --help` for the Synapse converter's source-layout options.

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
