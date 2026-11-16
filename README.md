# BDS-Lite

Code for **When does boundary distillation help lightweight medical image segmentation?**

Jingchen Chen, Chenxiang Hong, Yao Wu, Tao Yang · Revised manuscript, *PeerJ Computer Science*.

[Paper](paper/main.pdf) · [Supplement](paper/supplement.pdf) · [Reproduction](docs/reproducing.md) · [Evidence](evidence/README.md)

We compare training-only boundary signals in compact 2D U-Nets on ISIC2018, ACDC and Synapse.
The study includes component ablations, repeated training runs, external-teacher distillation,
and augmentation controls. The auxiliary boundary decoder is the reference for distillation.

## Installation

Python 3.10 or newer, with a PyTorch build suitable for your GPU:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,data]"
```

Teacher experiments additionally use `segmentation_models_pytorch==0.6.0` and external weights.
See the [recorded environment and requirements](docs/reproducing.md#environment).

## Data

Obtain [ISIC2018 Task 1](https://challenge.isic-archive.com/data/),
[ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/), and
[Synapse / BTCV](https://www.synapse.org/Synapse:syn3193805) from their providers.
[Conversion examples and data format](docs/dataset_format.md) describe the basic interface.
Paper experiments require the registered split manifests and preprocessing in the
[reproduction guide](docs/reproducing.md#paper-experiments).
Datasets, trained checkpoints and teacher caches are not bundled.

## Training and evaluation

A basic U-Net example after preparing ACDC:

```bash
python scripts/train.py --config configs/experiments/acdc_unet_baseline.yaml --device cuda
python scripts/evaluate.py --checkpoint results/checkpoints/acdc_unet_baseline/best.pt --split test
```

For the paper's experiments, use the study trainers and locked configurations in
`configs/revision/`, following the [study entry points](docs/reproducing.md#paper-experiments).
They use manifest-based data loading and study-specific evaluation.

## Verify the results

Recompute the archived ACDC statistics on a CPU, with NumPy and SciPy:

```bash
python analysis/revision/verify_archived_statistics.py
python evidence/verify_snapshot.py
```

No extraction, datasets or GPU are needed for these checks. To restore supporting results,
protocols and provenance records before further analysis:

```bash
python evidence/restore.py
```

The [evidence index](evidence/README.md) maps the paper's studies to their saved inputs.
[Rebuilding the paper](docs/reproducing.md#building-the-paper) restores its inputs automatically.

## Layout

| Directory | Contents |
| --- | --- |
| `src/bds_lite/` | Models, losses, metrics and data utilities |
| `scripts/` | Data preparation, training and evaluation |
| `configs/`, `splits/` | Paper configurations, basic examples and split identifiers |
| `analysis/` | Statistical analyses and figure/table generators |
| `results/` | Published summary tables and the independent verifier's contrast table |
| `evidence/` | Frozen archives, supporting records, checksums and restoration tools |
| `paper/` | Manuscript, supplement, LaTeX sources and published figures |
| `docs/`, `tests/` | Reproduction guide, attribution and tests |

## Citation

> Jingchen Chen, Chenxiang Hong, Yao Wu, and Tao Yang. *When does boundary distillation help
> lightweight medical image segmentation?* Revised manuscript, 2026.

See [CITATION.cff](CITATION.cff) for machine-readable citation metadata.

## License

Original code and text: [MIT](LICENSE). Third-party code and dataset-derived figures retain
their respective terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
