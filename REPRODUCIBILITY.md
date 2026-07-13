# Reproducibility

This repository is built so that **every reported number can be traced to a configuration, a script, and a machine-readable artifact**. This file is the operational entry point; additional details are provided in `reproduce.sh`, `DATASETS.md`, `docs/compute.md`, `docs/statistics.md`, and `docs/traceability.md`.

## Environment of record

- **Hardware:** one NVIDIA GeForce RTX 5060 Laptop GPU; Linux x86_64.
- **CUDA runtime:** 13.0.
- **Python:** 3.14.4.
- **PyTorch:** 2.11.0 (`torch==2.11.0+cu130`).
- **TorchVision:** 0.26.0.
- **Key libraries:** NumPy 2.4.4, SciPy 1.17.1, scikit-image 0.26.0, pandas 3.0.2, matplotlib 3.10.9, and h5py 3.16.0.
- **Optional dependency:** `nibabel`, required only when converting ACDC data from the official NIfTI format.

Exact package pins are provided in `requirements.txt`; the corresponding conda environment is provided in `environment.yml`. Additional environment details are documented in `docs/compute.md`.

```bash
pip install -r requirements.txt
pip install -e . --no-deps

# Alternatively:
conda env create -f environment.yml
```

The pinned PyTorch package uses the CUDA 13.0 build. Users with a different CUDA environment may need to install the corresponding official PyTorch build while retaining the remaining dependency versions.

## Random seeds and deterministic analysis

- **Training/model seeds:** 1, 2, and 3, representing three independent training runs per configuration.
- **Data-split seed:** 2026, with the locked split manifests stored under `splits/`.
- **Cluster-analysis seed:** `default_rng(20260606)`.

The bootstrap random-number generator is initialized at module level. Therefore, bootstrap confidence-interval endpoints are reproducible only when `analysis/generate_rescue_analysis.py` is run once from beginning to end. All non-bootstrap calculations in that script are deterministic for a fixed set of input prediction arrays.

Fresh model training can produce small numerical differences across GPU architectures, CUDA versions, PyTorch builds, and nondeterministic GPU kernels. Consequently, new training runs are not expected to reproduce the original prediction arrays bit-for-bit.

See `docs/statistics.md` for the complete statistical protocol.

## What is and is not distributed

| Distributed in this repository | Not distributed |
|---|---|
| Source code under `src/` | Original ISIC2018, ACDC, and Synapse datasets |
| Training, evaluation, analysis, and figure-generation scripts | Trained model checkpoints (`*.pt`) |
| Resolved configurations and locked split manifests | Per-case prediction arrays (`*.npy`) |
| Aggregate result CSV/JSON files | Training logs and temporary build files |
| Cluster-aware analysis outputs | Locally generated real-image case panels |
| Paper figures and machine-readable manuscript tables | Third-party data-derived images that cannot be redistributed |

The original datasets must be obtained from their official providers under the applicable access, licensing, and data-use conditions. Instructions are provided in `DATASETS.md` and `docs/data_access.md`.

Trained checkpoints and per-case prediction arrays are large intermediate artifacts and are not included in the repository. They can be regenerated locally by running the documented training and evaluation stages.

The committed aggregate results and cluster-analysis tables are the artifacts of record supporting the manuscript. They allow the reported values to be inspected without repeating the full GPU training process.

## Data preparation

Download the original datasets according to `DATASETS.md`. Replace the placeholder paths below with the corresponding local dataset locations.

### ISIC2018

```bash
python scripts/convert_isic2018.py \
  --image-dir /path/to/isic2018/images \
  --mask-dir /path/to/isic2018/masks \
  --output-root data/processed/isic2018 \
  --image-size 224x224
```

### ACDC

For the pre-sliced TransUNet-style HDF5 layout:

```bash
python scripts/convert_acdc.py \
  --src /path/to/acdc/ACDC_training_slices \
  --format transunet \
  --output-root data/processed/acdc \
  --image-size 224x224 \
  --val-fraction 0.1 \
  --test-fraction 0.2 \
  --seed 2026
```

For the official NIfTI layout, use `--format nifti` and install `nibabel`.

### Synapse

The source directory is expected to contain `train_npz/` and `test_vol_h5/`.

```bash
python scripts/convert_synapse.py \
  --src /path/to/Synapse \
  --output-root data/processed/synapse \
  --image-size 224x224 \
  --val-fraction 0.2 \
  --seed 2026
```

### Boundary and signed-distance targets

Generate the training-time boundary and signed-distance targets after conversion:

```bash
python scripts/prepare_boundary_targets.py \
  --dataset isic2018 \
  --split train \
  --num-classes 2

python scripts/prepare_boundary_targets.py \
  --dataset acdc \
  --split train \
  --num-classes 4

python scripts/prepare_boundary_targets.py \
  --dataset synapse \
  --split train \
  --num-classes 9
```

Dataset-specific source layouts, access conditions, and preprocessing details are documented in `DATASETS.md`.

## Main reproduction commands

`reproduce.sh` documents the complete data preparation → training → evaluation → analysis workflow. Computationally expensive stages are gated behind `RUN_HEAVY=1`, allowing the script to be inspected safely without starting model training.

### 1. Train U-Net and BDS-Lite

A single example run is:

```bash
python scripts/train.py \
  --config configs/run_resolved/isic2018_unet_seed1.yaml
```

The complete matched U-Net and BDS-Lite experiment consists of three datasets, two methods, and three training seeds:

```bash
for dataset in isic2018 acdc synapse; do
  for model in unet bds_lite_full; do
    for seed in 1 2 3; do
      python scripts/train.py \
        --config "configs/run_resolved/${dataset}_${model}_seed${seed}.yaml"
    done
  done
done
```

Run the matched U-Net+GSL comparator separately:

```bash
bash scripts/gsl/run_phase16_gsl.sh
```

All experiments use the resolved configurations under `configs/run_resolved/` and `configs/gsl/`.

### 2. Evaluate trained checkpoints

Re-evaluate the trained U-Net and BDS-Lite checkpoints and generate the per-case prediction arrays required by the cluster-aware analysis:

```bash
python scripts/reeval_main_seeds.py
```

The prediction arrays are written under:

```text
outputs/evaluations/predictions/
```

The per-run evaluation summaries are written under:

```text
outputs/evaluations/
```

Profile model size, computational cost, latency, and memory usage with:

```bash
python scripts/profile_model.py
```

### 3. Run the cluster-aware analysis

After the evaluation stage has generated all required U-Net and BDS-Lite prediction arrays, run:

```bash
python analysis/generate_rescue_analysis.py
```

The script expects predictions for:

- U-Net and BDS-Lite;
- seeds 1, 2, and 3;
- ISIC2018 validation data;
- ACDC test data;
- Synapse test data.

The processed ground-truth files must also be present under:

```text
data/processed/<dataset>/<split>/
```

Run the analysis script once from beginning to end to preserve the documented bootstrap RNG sequence.

### 4. Regenerate the paper figures

```bash
python scripts/figures/generate_architecture_figure.py
python scripts/figures/generate_effect_size_heatmap.py
```

Figure 1 is an architecture schematic and does not use patient images. Figure 2 is generated from the committed cluster-level statistical artifact.

## Expected outputs and manuscript mapping

| Command or artifact | Output | Manuscript item |
|---|---|---|
| `scripts/reeval_main_seeds.py` | Per-run evaluation JSON files and per-case prediction arrays under `outputs/evaluations/` | Evaluation evidence underlying Tables 2 and 3 |
| `results/main_test_means.csv` | Committed U-Net and BDS-Lite aggregate results | Table 2 |
| `scripts/gsl/run_phase16_gsl.sh` and the corresponding GSL aggregation scripts | `results/matched_gsl_summary.csv` and related GSL artifacts | Table 2 |
| `analysis/generate_rescue_analysis.py` | `analysis/outputs/cluster_level_statistics.csv` | Table 3 |
| `analysis/generate_rescue_analysis.py` | `analysis/outputs/failure_case_manifest.csv`, `per_case_metrics.csv`, and supporting analyses | Per-case and failure-case analyses |
| `scripts/profile_model.py` | `results/profiling/*` and `results/resource_profile_isic2018_comparison.csv` | Table 4 |
| `scripts/figures/generate_architecture_figure.py` | Architecture schematic | Figure 1 |
| `scripts/figures/generate_effect_size_heatmap.py` | Cluster-level effect-size heatmap | Figure 2 |

See `docs/paper_mapping.md` for the complete manuscript item → artifact crosswalk.

## Verified reproductions

During release preparation, the following checks were performed:

- Given the fixed per-case prediction arrays used for the manuscript, a complete run of `analysis/generate_rescue_analysis.py` regenerated `cluster_level_statistics.csv`, `failure_case_manifest.csv`, and the supporting per-case tables bit-for-bit, including identical SHA-256 hashes.
- The bootstrap confidence intervals reproduced exactly when the full analysis script was run once from beginning to end.
- The U-Net+GSL surface-loss implementation was verified against its Apache-2.0 upstream implementation from MIST; see `THIRD_PARTY_NOTICES.md`.
- The seed-2026 split manifests are subject-disjoint and SHA-256 stamped. Their integrity is checked by `tests/unit/test_split_integrity.py`.

These bit-for-bit checks apply to fixed input artifacts. They do not imply that newly trained checkpoints or newly generated prediction arrays will be bit-identical across different hardware or software environments.

## Known limitations

- The original datasets are third-party resources and are not redistributed. Users must obtain them independently from the official providers.
- Trained checkpoints and per-case prediction arrays are not distributed. Full regeneration therefore requires access to the original datasets, a compatible GPU environment, and the computational time needed to repeat all training and evaluation runs.
- Table 3 can be recalculated after regenerating the complete set of U-Net and BDS-Lite prediction arrays and rebuilding the processed ground truth.
- Three training seeds are treated as descriptive replication and are not used as the independent sample size for superiority claims.
- ACDC and Synapse contain relatively small numbers of independent clusters: 20 patients and 12 cases, respectively.
- GPU kernels, interpolation operations, and distance-transform implementations can introduce small numerical differences across systems.
- The final manuscript contains two figures. Earlier real-image representative and failure-case panels are not redistributed because they contain images derived from third-party medical datasets; their case identifiers and metric values are retained in `analysis/outputs/failure_case_manifest.csv`.
- The typeset figures were finalized in the LaTeX project. The provided scripts reproduce their scientific content and source values but do not guarantee pixel-identical publication exports.
