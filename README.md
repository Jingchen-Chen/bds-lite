# BDS-Lite

Code and experiment records for **When does boundary distillation help lightweight medical image segmentation?**

Jingchen Chen, Chenxiang Hong, Yao Wu, Tao Yang. Revised manuscript submitted to *PeerJ Computer Science*.

[Paper](paper/main.pdf) · [Supplement](paper/supplement.pdf) · [Reproduction guide](docs/reproducing.md) · [Results](results/README.md)

We study training-time boundary signals for compact 2D U-Nets on ISIC2018, ACDC and Synapse.
The experiments compare an auxiliary boundary decoder, the components of BDS-Lite, and distillation
from frozen external teachers over repeated training runs. This repository includes models and losses,
run configurations, split manifests, analysis code, archived results and manuscript sources.

## Installation

Use Python 3.10 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,data]"
```

Install a PyTorch build appropriate for your GPU and CUDA environment. External-teacher experiments
also require `segmentation_models_pytorch==0.6.0` and teacher weights. The recorded study environment
and remaining reproduction dependencies are described in the [reproduction guide](docs/reproducing.md#environment).

## Data preparation

Obtain the datasets from their providers:

- [ISIC2018 Task 1](https://challenge.isic-archive.com/data/): skin lesion segmentation.
- [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/): cardiac MRI segmentation.
- [Synapse / BTCV](https://www.synapse.org/Synapse:syn3193805): abdominal organ segmentation.

Converters are provided in `scripts/convert_isic2018.py`, `scripts/convert_acdc.py` and
`scripts/convert_synapse.py`. The basic training interface reads samples from:

```text
data/processed/<dataset>/{train,val,test}/<sample_id>.npz
```

See the [dataset format and conversion examples](docs/dataset_format.md). To reproduce a paper
experiment, use its registered manifests and preprocessing procedure in [protocols/](protocols/README.md).
Datasets and trained checkpoints are not bundled; archived case-level results support statistical
reanalysis without either.

## Training and evaluation

After preparing ACDC data, train and evaluate a plain U-Net with the basic interface:

```bash
python scripts/train.py --config configs/experiments/acdc_unet_baseline.yaml --device cuda
python scripts/evaluate.py \
  --checkpoint results/checkpoints/acdc_unet_baseline/best.pt \
  --split test
```

The trainer saves logs under `results/runs/<experiment>/` and checkpoints under
`results/checkpoints/<experiment>/`. Configurations in `configs/experiments/` inherit defaults from
`configs/base.yaml` and a dataset configuration.

Paper experiments use study-specific trainers, locked configurations in `configs/revision/` and
registered run queues. Follow the [study entry points](docs/reproducing.md#paper-experiments) for those
experiments; the basic example above uses a different data-loading and evaluation interface.

## Reproducing the analysis

The main archived ACDC contrasts can be checked on a CPU with NumPy and SciPy, without extracting
archives or downloading data:

```bash
python analysis/revision/verify_archived_statistics.py
python evidence/verify_snapshot.py
```

For archive checks, per-case extraction, tests and rebuilding the paper, see the
[reproduction guide](docs/reproducing.md). Prebuilt [manuscript](paper/main.pdf) and
[supplement](paper/supplement.pdf) PDFs are included.

## Repository structure

| Path | Contents |
| --- | --- |
| [src/bds_lite/](src/bds_lite/) | Models, losses, data loading and training utilities |
| [configs/](configs/) / [splits/](splits/) | Experiment configurations and split manifests |
| [scripts/](scripts/) | Data conversion, training and evaluation entry points |
| [analysis/](analysis/) | Statistical analyses, tables and figures |
| [protocols/](protocols/README.md) | Study protocols, amendments and run queues |
| [results/](results/README.md) | Case-level results, summary tables and validation outputs |
| [evidence/](evidence/) | Archived results, checksums and verification scripts |
| [paper/](paper/README.md) | Manuscript sources, PDFs and build scripts |
| [tests/](tests/) / [docs/](docs/) | Tests and supporting documentation |

## Citation

Please cite the accompanying manuscript if you use this work:

> Jingchen Chen, Chenxiang Hong, Yao Wu, and Tao Yang. *When does boundary distillation help
> lightweight medical image segmentation?* Revised manuscript, 2026.

Machine-readable software and paper citation metadata are provided in [CITATION.cff](CITATION.cff).

## License

Original code and text are released under the [MIT License](LICENSE). Third-party code and dataset-derived
figures retain their respective licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
For contributions and issue reports, see [CONTRIBUTING.md](CONTRIBUTING.md).
