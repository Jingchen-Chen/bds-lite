# BDS-Lite — companion repository

**When Does Boundary Distillation Help Lightweight Medical Image Segmentation?**

This repository is the reproducibility/provenance companion to the paper. Its purpose
is that a reviewer can trace **every reported number to a config, a script, and a
machine-readable artifact**. It is not a general-purpose framework.

## What BDS-Lite is

A compact U-Net with a **training-time** boundary-distillation path: an auxiliary
boundary decoder `h_φ` and a bounded gate `q̃ = q · (1 + α(g − 0.5))` with α = 0.25.
The auxiliary branch is **removed at inference**, so the deployed graph is within
~0.06% of the U-Net parameter count (1,928,114 vs 1,927,042). It is evaluated against
**U-Net** and **U-Net + GSL** (Generalized Surface Loss) on ISIC2018, ACDC, and
Synapse, over three seeds.

## Headline finding: conditional and mixed (no uniform improvement)

The effect of the boundary path is **task-dependent**, established with cluster-aware
statistics (cluster bootstrap CIs, Holm-corrected signed-rank tests, sign effect
sizes, leave-one-cluster-out):

- **ISIC2018:** cluster-level boundary/distance metrics improve and reach Holm
  significance; DSC essentially unchanged.
- **ACDC:** no stable advantage (Holm-adjusted p ≈ 1.0).
- **Synapse:** overlap/boundary means up, but **distance means worse** — a genuine
  trade-off.

There is **no** state-of-the-art claim, **no** uniform-improvement claim, **no**
clinical-deployment or faster-training claim, and **no** Holm-corrected seed-level
superiority at n = 3. See `docs/mechanism.md` and
`manuscript/tables/table6_claim_evidence_final.csv`.

## Repository layout

```
manuscript/      the paper (.md) + Tables 1–6 (.csv/.md) + integration audit
src/bds_lite/    U-Net, BDS-Lite (h_φ + bounded gate), GSL, losses, metrics, data
configs/         base + dataset configs, run_resolved/ (config of record), gsl/
splits/          locked seed-2026 split manifests (sha256, subject-disjoint)
scripts/         train / evaluate / convert / profile (+ scripts/gsl/)
analysis/        cluster-aware analysis script + outputs/ (artifacts of record)
results/         main means, matched-GSL, seed-level stats, profiling, gsl/
figures/         Figures 1–6 + the 24 failure panels + captions
docs/            data access, mechanism, naming caveat, compute, statistics, config
tests/           unit tests (split integrity, metrics, losses, boundary, profiling)
```

**Not in git:** raw datasets (third-party; see `docs/data_access.md`), and large
binaries — trained checkpoints and per-case prediction arrays — which are on **Zenodo**
(see `ZENODO_MANIFEST.csv`). `.npy/.npz/.pt` and `*.log` are git-ignored by design.

## Install

```bash
pip install -r requirements.txt   # exact pins; PyTorch built for CUDA 13.0
# or: conda env create -f environment.yml
```

Environment of record: Python 3.14.4, PyTorch 2.11.0+cu130, RTX 5060 Laptop GPU,
Linux. See `docs/compute.md`.

## Reproduce map (paper → artifact)

| Paper item | Artifact of record |
|---|---|
| Table 1 (dataset protocol) | `manuscript/tables/table1_dataset_protocol.csv`, `splits/` |
| Table 2 (matched main results) | `results/main_test_means.csv` (U-Net, BDS-Lite) + `results/matched_gsl_summary.csv` (U-Net+GSL) |
| Table 3 (cluster statistics) | `analysis/outputs/cluster_level_statistics.csv` |
| Table 4 (training/inference cost) | `results/profiling/training_cost_combined_summary.csv` |
| Table 5 (failure/limitation) | `analysis/outputs/failure_case_manifest.csv` (+ `manuscript/tables/table5_*`) |
| Table 6 (claim evidence) | `manuscript/tables/table6_claim_evidence_final.csv` |
| Figures 1–6 | `figures/figure{1..6}_*.{pdf,png}` |
| 24 failure panels | `figures/failure_panels/*.png` |

`reproduce.sh` documents the full convert → train → evaluate → analysis flow (compute
stages gated behind `RUN_HEAVY=1`). The cluster analysis is a **single end-to-end run**
of `analysis/generate_rescue_analysis.py`; its outputs were verified to reproduce
bit-for-bit (see `analysis/README.md`, `docs/statistics.md`).

## Important caveats

- **ISIC2018 `_test` = validation.** ISIC2018 has no test split; everything labeled
  ISIC "test" is the 519-image validation set, and the prediction/aggregate artifacts
  use opposite `_test`/`_val` suffixes. Read `docs/naming.md` before tracing ISIC
  numbers.
- **Config of record.** The `model.name: bds_lite` and `dataset.split_file` fields in
  the resolved snapshots need interpretation; see `docs/config_of_record.md`.

## Licensing and third-party code

Original code is MIT (`LICENSE`). The GSL component (`src/bds_lite/losses/gsl.py`) is
adapted from the Apache-2.0 MIST framework; see `THIRD_PARTY_NOTICES.md` and
`THIRD_PARTY_LICENSES/Apache-2.0-MIST.txt`. Datasets are not redistributed.

## Citation

See `CITATION.cff`. Please cite the paper and the GSL/MIST references listed in
`THIRD_PARTY_NOTICES.md`.
