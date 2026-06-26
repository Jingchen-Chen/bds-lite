# BDS-Lite — companion code and reproducibility repository

**When Does Boundary Distillation Help Lightweight Medical Image Segmentation?**

Repository: <https://github.com/Jingchen-Chen/bds-lite>

## Description

This is the code, configuration, and provenance companion to the paper of the same title (submitted to *PeerJ Computer Science* as an **AI Application**). Its purpose is that a reader can trace **every reported number to a config, a script, and a machine-readable artifact**, and re-run the analysis end-to-end. It is a study reproducibility package, not a general-purpose framework.

**What BDS-Lite is.** A compact 2D U-Net with a **training-time** boundary-distillation path: an auxiliary boundary decoder plus a feature-distillation loss and a bounded gate `F_s · (1 + α(g − 0.5))` with α = 0.25. The auxiliary branch is **removed at inference**, so the deployed graph is within ~0.06% of the U-Net parameter count (1,928,114 vs 1,927,042). It is compared against **U-Net** and **U-Net + GSL** (Generalized Surface Loss) on ISIC2018, ACDC, and Synapse over three training seeds.

**Headline finding — conditional and mixed (no uniform improvement).** Established with cluster-aware statistics (cluster bootstrap CIs, Holm-corrected signed-rank tests, sign-effect sizes, leave-one-cluster-out):

- **ISIC2018:** small boundary/distance gains reach Holm significance; DSC essentially unchanged.
- **ACDC:** no stable advantage (Holm-adjusted *p* ≈ 1.0).
- **Synapse:** overlap/boundary means up, but **surface-distance means worse** — a genuine trade-off.

There is **no** state-of-the-art claim, **no** uniform-improvement claim, **no** clinical-deployment or faster-training claim, and **no** Holm-corrected seed-level superiority at *n* = 3.

## Dataset Information

The three benchmarks are **third-party** datasets and are **not redistributed** here (no raw images, and no derived patient-image panels). Obtain each from its official provider under that provider's data-use terms:

| Dataset | Task / classes | Source | Access |
|---|---|---|---|
| **ISIC 2018** (Task 1, lesion segmentation) | binary skin lesion (2D), 1 fg class | ISIC Challenge archive — <https://challenge.isic-archive.com/data/> | free account |
| **ACDC** (Automated Cardiac Diagnosis Challenge) | cardiac MRI (2D slices), 3 fg classes | Creatis / MICCAI 2017 — <https://www.creatis.insa-lyon.fr/Challenge/acdc/> | registration + agreement; citation required |
| **Synapse** (Multi-Atlas Labeling Beyond the Cranial Vault, abdomen) | abdominal CT (2D slices), 8 fg organs | Synapse `syn3193805`, DOI `10.7303/syn3193805` — <https://www.synapse.org/Synapse:syn3193805> | Synapse account + data-use agreement |

Citations: ISIC2018 (Codella et al. 2019; Tschandl et al. 2018), ACDC (Bernard et al. 2018), Synapse (Landman et al. 2015). Deeper access/preprocessing notes: [`DATASETS.md`](DATASETS.md) and [`docs/data_access.md`](docs/data_access.md). Only the locked split manifests (`splits/`) and dataset configs (`configs/datasets/`) are shipped.

**Licensing note.** ISIC 2018 images carry per-image Creative Commons licenses (CC-0, CC-BY, or CC-BY-NC); ACDC and Synapse are access-controlled and governed by their data-use agreements. Because these terms do not give a blanket grant to redistribute patient images, **no original dataset images (or image-derived case panels) are stored or published in this repository or in the paper figures** (see *Figures*, below).

## Code Information

```
src/bds_lite/    U-Net, BDS-Lite (aux boundary decoder + bounded gate), GSL, losses, metrics, data
configs/         base + dataset configs; run_resolved/ (config of record); gsl/
splits/          locked seed-2026 split manifests (sha256, subject-disjoint): {acdc,isic2018,synapse}/*.json
scripts/         convert_*.py, prepare_boundary_targets.py, train.py, evaluate.py,
                 reeval_main_seeds.py, profile_model.py, gsl/, figures/
analysis/        generate_rescue_analysis.py (cluster-aware analysis) + outputs/ (artifacts of record)
results/         main_test_means.csv, matched_gsl_summary.csv, profiling/, gsl/, seed_level_wilcoxon_summary.csv
figures/         Figure_1.png, Figure_2.png + captions.md
docs/            data access, mechanism, naming, compute, statistics, config, paper_mapping, traceability
tests/           unit tests (split integrity, metrics, losses, boundary, profiling)
manuscript/      typeset PDF, paper figures, figure captions, tables.docx
```

Top-level guides: [`RELEASE_INVENTORY.md`](RELEASE_INVENTORY.md) (what is shipped / missing), [`DATASETS.md`](DATASETS.md) (data access), [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) (environment + commands).

**Not in git:** raw datasets (third-party — see `DATASETS.md`) and large binaries (trained checkpoints `*.pt`, per-case prediction arrays `*.npy/.npz`), which are on **Zenodo** (see [`ZENODO_MANIFEST.csv`](ZENODO_MANIFEST.csv); DOI to be minted at deposit — `TODO`). `.npy/.npz/.pt` and `*.log` are git-ignored by design.

## Requirements / dependencies

- **Hardware of record:** single NVIDIA GeForce RTX 5060 Laptop GPU; Linux x86_64.
- **Python 3.14.4**, **PyTorch 2.11.0** (`torch==2.11.0+cu130`, CUDA 13.0 build), TorchVision 0.26.0; NumPy 2.4.4, SciPy 1.17.1, scikit-image 0.26.0, pandas 3.0.2, matplotlib 3.10.9, h5py 3.16.0 (`nibabel` is an optional extra needed only for the ACDC NIfTI converter).
- Exact pins: [`requirements.txt`](requirements.txt); conda: [`environment.yml`](environment.yml); full detail: [`docs/compute.md`](docs/compute.md).

```bash
pip install -r requirements.txt    # exact pins; PyTorch is a CUDA-13.0 build
pip install -e . --no-deps         # install the local bds_lite package
# or: conda env create -f environment.yml
```

## Usage Instructions

```bash
# 0. Data: download raw datasets (see DATASETS.md), then build processed tensors
python scripts/convert_isic2018.py            # -> data/processed/isic2018/{train,val}
python scripts/convert_acdc.py                # -> data/processed/acdc/{train,val,test}   (needs nibabel)
python scripts/convert_synapse.py             # -> data/processed/synapse/{train,val,test}
python scripts/prepare_boundary_targets.py    # adds boundary + SDF training targets

# 1. Train (GPU; 150 epochs each); configs of record under configs/run_resolved/ and configs/gsl/
python scripts/train.py --config configs/run_resolved/isic2018_unet_seed1.yaml
bash   scripts/gsl/run_phase16_gsl.sh          # U-Net+GSL comparator, matched protocol

# 2. Evaluate -> per-case predictions + aggregate evals; resource profile
python scripts/reeval_main_seeds.py
python scripts/profile_model.py

# 3. Cluster-aware analysis (SINGLE end-to-end run; needs the Zenodo prediction arrays)
python analysis/generate_rescue_analysis.py

# 4. Figures from artifacts of record (the paper has two figures)
python scripts/figures/generate_architecture_figure.py   # Figure 1 schematic (no real images)
python scripts/figures/generate_effect_size_heatmap.py   # Figure 2 content
```

`reproduce.sh` documents the full convert → train → evaluate → analysis flow, with heavy compute gated behind `RUN_HEAVY=1` so it is safe to read top-to-bottom. Point the dataset configs (`configs/datasets/*.yaml`) at your local `data/processed/<dataset>` directory using relative paths.

## Methodology

A two-layer comparison. **Matched layer:** U-Net, BDS-Lite, and U-Net+GSL are trained under identical locked split manifests, seeds (1, 2, 3), processed 224×224 inputs, AdamW + cosine schedule (150 epochs), AMP policy, and max-validation-DSC checkpoint rule; all statistical claims are restricted to this family. **Contextual layer:** recent literature is used for positioning only, with no cross-protocol ranking. Five metrics are reported separately as three families — overlap (DSC, IoU), boundary (Boundary F1), and surface distance (HD95, ASSD) — because they can disagree. Inference uses the cluster as the statistical unit (images for ISIC2018; patients for ACDC; cases for Synapse); three seeds are treated as descriptive replication, not a basis for seed-level superiority. See the manuscript Methods and [`docs/mechanism.md`](docs/mechanism.md), [`docs/statistics.md`](docs/statistics.md).

## Reproducibility instructions and artifacts

Map of each paper item to its artifact of record (all paths below exist in this repo unless marked *Zenodo*):

| Paper item | Artifact of record | Producing script |
|---|---|---|
| Table 1 (dataset protocol) | `splits/`, `configs/datasets/` (+ `DATASETS.md`) | — |
| Table 2 (matched main results) | `results/main_test_means.csv` (U-Net, BDS-Lite) + `results/matched_gsl_summary.csv` (U-Net+GSL) | `scripts/reeval_main_seeds.py`, `scripts/gsl/` |
| Table 3 (cluster statistics) | `analysis/outputs/cluster_level_statistics.csv` | `analysis/generate_rescue_analysis.py` |
| Table 4 (training/inference cost) | `results/profiling/training_cost_combined_summary.csv`, `results/resource_profile_isic2018_comparison.csv` | `scripts/profile_model.py` |
| Per-case / failure analysis | `analysis/outputs/failure_case_manifest.csv`, `analysis/outputs/per_case_metrics.csv` | `analysis/generate_rescue_analysis.py` |
| Figure 1 (architecture) | schematic (synthetic input; no dataset image) | `scripts/figures/generate_architecture_figure.py` |
| Figure 2 (effect-size heatmap) | `figures/Figure_2.png` (content) | `scripts/figures/generate_effect_size_heatmap.py` |

The cluster analysis is a **single end-to-end run** of `analysis/generate_rescue_analysis.py`; `cluster_level_statistics.csv` and `failure_case_manifest.csv` reproduce **bit-for-bit (identical sha256)** when run end-to-end (all non-bootstrap quantities are fully deterministic; the bootstrap RNG is seeded `default_rng(20260606)`). See [`analysis/README.md`](analysis/README.md), [`docs/statistics.md`](docs/statistics.md), and the traceability hub [`docs/traceability.md`](docs/traceability.md). Split manifests are seed-2026, subject-disjoint, and sha256-stamped, guarded by `tests/unit/test_split_integrity.py`.

## Figures (image-rights note)

To comply with the datasets' redistribution terms, the paper's figures contain **no original patient images**. The final paper has **two figures**:

- **Figure 1** (architecture) is a schematic; its input panel is a **synthetic** image and the mask/boundary panels are schematic label maps.
- **Figure 2** is the cluster-level effect-size heatmap (numeric; no images).

There is **no Figure 3**. The earlier representative/failure-case panels were rendered on top of real ACDC/Synapse/ISIC scans and are **not redistributed**; they are regenerable locally from your own downloaded data via `analysis/generate_rescue_analysis.py`, and the selected case IDs and exact metric deltas are recorded in `analysis/outputs/failure_case_manifest.csv`.

## Important caveats

- **ISIC2018 `_test` = validation.** ISIC2018 has no public test split; everything labeled ISIC "test" is the 519-image validation set. Read [`docs/naming.md`](docs/naming.md) before tracing ISIC numbers.
- **Config of record.** Some fields in the resolved snapshots need interpretation; see [`docs/config_of_record.md`](docs/config_of_record.md).
- **Three seeds** are descriptive replication; ACDC/Synapse cluster sizes are small (20 patients / 12 cases). GPU kernels may introduce last-bit nondeterminism across hardware, so conclusions are reported at the cluster-aware statistical level.

## Citation

If you use this repository or its artifacts, please cite the paper and this software (see [`CITATION.cff`](CITATION.cff)), and the GSL/MIST references in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Please also cite the dataset references above as required by their providers.

## License

Original code in this repository is released under the **MIT License** ([`LICENSE`](LICENSE)). The GSL component (`src/bds_lite/losses/gsl.py`) is adapted from the **Apache-2.0** MIST framework; the upstream license is vendored verbatim at [`THIRD_PARTY_LICENSES/Apache-2.0-MIST.txt`](THIRD_PARTY_LICENSES/Apache-2.0-MIST.txt) and the modifications are documented in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Datasets are **not** covered by this license and remain under their providers' terms.
