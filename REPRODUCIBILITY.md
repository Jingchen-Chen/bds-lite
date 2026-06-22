# Reproducibility

This repository is built so that **every reported number traces to a config, a script,
and a machine-readable artifact**. This file is the operational entry point; deeper
detail lives in `reproduce.sh`, `docs/compute.md`, `docs/statistics.md`, and
`docs/traceability.md`.

## Environment of record

- **Hardware:** single NVIDIA GeForce RTX 5060 Laptop GPU; Linux (x86_64).
- **CUDA runtime:** 13.0.
- **Python:** 3.14.4.
- **PyTorch:** 2.11.0 (`torch==2.11.0+cu130`), TorchVision 0.26.0.
- Key libraries: NumPy 2.4.4, SciPy 1.17.1, scikit-image 0.26.0, pandas 3.0.2,
  matplotlib 3.10.9, h5py 3.16.0. Exact pins: `requirements.txt`; conda:
  `environment.yml`. Full detail: `docs/compute.md`.

```bash
pip install -r requirements.txt    # exact pins; PyTorch is a CUDA-13.0 build
pip install -e . --no-deps         # install the local bds_lite package
# or: conda env create -f environment.yml
```

## Seeds

- **Training/model seeds: 1, 2, 3** (three independent runs per configuration).
- **Data split seed: 2026** (locked manifests under `splits/`).
- The cluster-aware analysis uses a module-global RNG seeded `default_rng(20260606)`;
  bootstrap CI endpoints reproduce **exactly only when `analysis/generate_rescue_analysis.py`
  is run end-to-end** (all non-bootstrap quantities are fully deterministic). See
  `docs/statistics.md`.

## What is and isn't shipped

| Shipped in git | Not in git |
|---|---|
| code (`src/`), configs, split manifests, analysis script + outputs, result CSV/JSON, figures, manuscript tables | raw datasets (third-party — see `DATASETS.md`) |
| | trained checkpoints `*.pt` and per-case prediction `*.npy` (**Zenodo**, see `ZENODO_MANIFEST.csv`) |
| | logs, build caches (git-ignored) |

## Main commands

`reproduce.sh` documents the full convert → train → evaluate → analysis flow. Heavy
compute stages are gated behind `RUN_HEAVY=1` so the script is safe to read top-to-bottom;
the committed artifacts of record already reflect those stages.

```bash
# 0. data: see DATASETS.md (download raw, then scripts/convert_*.py + prepare_boundary_targets.py)

# 1. train (GPU; 150 epochs each), config of record under configs/run_resolved/ and configs/gsl/
python scripts/train.py --config configs/run_resolved/isic2018_unet_seed1.yaml
bash   scripts/gsl/run_phase16_gsl.sh          # U-Net+GSL comparator, matched protocol

# 2. evaluate -> per-case predictions + aggregate evals; resource profile
python scripts/reeval_main_seeds.py
python scripts/profile_model.py

# 3. cluster-aware analysis (SINGLE full-script run; needs Zenodo prediction arrays)
python analysis/generate_rescue_analysis.py

# 4. regenerate paper figures from artifacts of record
python scripts/figures/generate_architecture_figure.py
python scripts/figures/generate_effect_size_heatmap.py
```

## Expected outputs → manuscript items

| Command | Produces | Backs |
|---|---|---|
| `scripts/reeval_main_seeds.py` | `results/main_test_means.csv` | Table 2 (U-Net, BDS-Lite) |
| `scripts/gsl/run_phase16_gsl.sh` + `aggregate_phase16_gsl.py` | `results/matched_gsl_summary.csv` | Table 2 (U-Net+GSL) |
| `analysis/generate_rescue_analysis.py` | `analysis/outputs/cluster_level_statistics.csv`, `failure_case_manifest.csv`, `figures/failure_panels/*` | Table 3, Figure 3 |
| `scripts/profile_model.py` | `results/profiling/*`, `results/resource_profile_isic2018_comparison.csv` | Table 4 |
| `scripts/figures/generate_effect_size_heatmap.py` | `figures/figure_cluster_effect_size_heatmap.*` | Figure 2 |

See `docs/paper_mapping.md` for the complete table/figure → artifact crosswalk.

## Verified reproductions

- The cluster-aware analysis regenerates `cluster_level_statistics.csv` and
  `failure_case_manifest.csv` **bit-for-bit (identical sha256)** when run end-to-end
  (see `analysis/README.md`).
- The GSL comparator's surface term is **bit-identical** to its Apache-2.0 upstream
  (MIST); see `THIRD_PARTY_NOTICES.md`.
- Split manifests are seed-2026, subject-disjoint, sha256-stamped; guarded by
  `tests/unit/test_split_integrity.py`.

## Known limitations

- **No raw data** (third-party licenses) and **no checkpoints/prediction arrays** in git
  — the latter are on Zenodo. Reproducing Table 3 / Figure 3 needs the Zenodo prediction
  bundle plus locally rebuilt processed ground truth.
- **Three seeds** are descriptive replication, not a basis for seed-level superiority
  claims; ACDC/Synapse cluster sizes are small (20 patients / 12 cases).
- GPU kernels (distance transforms, interpolation) may introduce last-bit
  nondeterminism across hardware; conclusions are reported at the cluster-aware
  statistical level, not bit-exact prediction reproduction.
- The published (typeset) figures were finalized in the LaTeX project; the scripts here
  reproduce their **content** from artifacts of record, not pixel-identical exports
  (see `docs/paper_mapping.md`).
