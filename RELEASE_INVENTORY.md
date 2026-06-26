# Release inventory

Audit of what the manuscript claims and where each item lives in this release. Source paths refer to the author's working tree (`bds-lite/`); target paths are in this public repository. The companion `docs/provenance_crosswalk.md` resolves the verbatim `source_artifact` paths embedded in the shipped CSVs.

Status legend: ✅ present · ➕ added/regenerated in this pass · 🌐 distributed via Zenodo · ⚠️ caveat (see notes).

## Datasets and protocol

| Paper item | Release artifact | Source | Status |
|---|---|---|---|
| ISIC2018 / ACDC / Synapse access + protocol | `DATASETS.md`, `docs/data_access.md` | dataset docs | ✅ |
| Locked split manifests (seed 2026) | `splits/{isic2018,acdc,synapse}/*.json` | `data/splits/<ds>/*.json` | ✅ |
| Table 1 (datasets/protocol) | `manuscript/tables/table1_dataset_protocol.{csv,md}` | submission_candidate_manuscript_v1 | ✅ |
| Raw images/labels | **not shipped** (third-party licenses) | — | ⚠️ rebuild locally |

## Methods / code

| Paper item | Release artifact | Source | Status |
|---|---|---|---|
| Compact U-Net backbone | `src/bds_lite/models/unet.py`, `models/blocks.py` | `src/bds_lite/models/` | ✅ |
| BDS-Lite (aux boundary decoder, projection, bounded gate, distillation) | `src/bds_lite/models/bds_lite.py` | same | ✅ |
| Losses (CE, Dice, boundary BCE+Dice, SDF surface, distillation) | `src/bds_lite/training/losses.py` | `src/bds_lite/training/` | ✅ |
| U-Net+GSL (GeneralizedSurfaceLoss, MIST-derived) | `src/bds_lite/losses/gsl.py` (+`THIRD_PARTY_LICENSES/`) | `src/bds_lite/losses/` | ✅ ⚠️ Apache-2.0 |
| Boundary / SDF target generation | `src/bds_lite/data/boundary.py` (+`data/README.md`) | `src/bds_lite/data/` | ✅ |
| Metrics (DSC, IoU, Boundary F1, HD95, ASSD) | `src/bds_lite/evaluation/metrics.py` | `src/bds_lite/evaluation/` | ✅ |
| Resource profiling (params/FLOPs/latency/memory) | `src/bds_lite/evaluation/profiling.py`, `scripts/profile_model.py` | same | ✅ |
| Dataset loading / converters | `src/bds_lite/data/{datasets,converters,schema}.py`, `scripts/convert_*.py` | same | ✅ |
| Training/eval entrypoints | `scripts/train.py`, `scripts/evaluate.py`, `scripts/reeval_main_seeds.py` | same | ✅ |
| Seed control / config utils | `src/bds_lite/utils/{seed,config}.py`, `src/bds_lite/training/builders.py` | same | ✅ |

## Configs

| Paper item | Release artifact | Status |
|---|---|---|
| U-Net configs, 3 datasets × 3 seeds | `configs/run_resolved/{ds}_unet_seed{1,2,3}.yaml` | ✅ |
| BDS-Lite configs, 3 datasets × 3 seeds | `configs/run_resolved/{ds}_bds_lite_full_seed{1,2,3}.yaml` | ✅ |
| U-Net+GSL matched configs, 3 datasets × 3 seeds | `configs/gsl/phase16_matched_gsl_{ds}_seed{1,2,3}.yaml` | ✅ |
| Dataset / base configs (size, normalization, classes) | `configs/datasets/*.yaml`, `configs/base.yaml` | ✅ |

## Results / analysis artifacts

| Paper item | Release artifact | Source | Status |
|---|---|---|---|
| Table 2 (matched main results) | `results/main_test_means.csv`, `results/matched_gsl_summary.csv`, `results/matched_gsl_seed_results.csv` | final_manuscript_audit_v2, submission_blocker_resolution_v1 | ✅ |
| Table 3 (cluster stats: Wilcoxon W, raw p, Holm p, sign effect, CI, LOO) | `analysis/outputs/cluster_level_statistics.csv` | midterm_q2_rescue | ✅ |
| Table 4 (resource profile, train/inf cost) | `results/profiling/training_cost_combined_summary.csv`, `environment.json`, `results/resource_profile_isic2018_comparison.csv` | submission_critical_evidence_v1 | ✅ |
| Seed-level Wilcoxon (n=3, all Holm≈1.0) | `results/seed_level_wilcoxon_summary.csv` | final_manuscript_audit_v2 | ✅ |
| Favorable-fraction / per-case / subgroup / organ / outlier / tradeoff summaries | `analysis/outputs/*.csv` | midterm_q2_rescue | ✅ |
| Figure-source per-case metrics | `analysis/outputs/per_case_metrics.csv`, `per_class_case_metrics.csv` | midterm_q2_rescue | ✅ |
| Gate-collapse robustness (Resource section) | `analysis/inputs/p1b_gate_comparison_summary.csv`, `analysis/outputs/gate_removal_summary.csv` | post_recovery_eval | ✅ |
| Trained checkpoints (`*.pt`) | Zenodo `checkpoints_unet_bdslite`, `checkpoints_gsl` | `outputs/runs/*/best.pt` | 🌐 |
| Per-case prediction arrays (`*.npy`) | Zenodo `predictions` | `outputs/evaluations/predictions/` | 🌐 |

## Figures

| Paper item | Release artifact | Generator | Status |
|---|---|---|---|
| Figure 1 (architecture) | `figures/figure_architecture_schematic.*` (+ legacy `figure1_*`) | `scripts/figures/generate_architecture_figure.py` | ➕ ⚠️ schematic, see below |
| Figure 2 (cluster-level effect-size heatmap) | `figures/figure_cluster_effect_size_heatmap.*` | `scripts/figures/generate_effect_size_heatmap.py` | ➕ regenerated from Table 3 CSV |
| ~~Figure 3~~ | **removed from the final paper**; representative/failure cases are reported in text + `analysis/outputs/failure_case_manifest.csv` | `analysis/generate_rescue_analysis.py` | 🚫 removed (final paper has 2 figures) |
| Real-image per-case panels | generated locally only — **not shipped** (datasets' redistribution terms; regenerable) | `analysis/generate_rescue_analysis.py` | 🚫 not redistributed |
| Captions | `figures/captions.md` | — | ✅ ⚠️ six-figure legacy text |

## Release scaffolding (this repository)

| Item | File | Status |
|---|---|---|
| README, license, citation | `README.md`, `LICENSE`, `CITATION.cff` | ✅ |
| Env pins | `requirements.txt`, `environment.yml`, `pyproject.toml` | ✅ |
| `.gitignore` | `.gitignore` | ✅ |
| Dataset guide | `DATASETS.md`, `docs/data_access.md` | ➕ DATASETS.md added |
| Reproduction guide | `REPRODUCIBILITY.md`, `reproduce.sh` | ➕ REPRODUCIBILITY.md added |
| Paper ↔ repo mapping | `docs/paper_mapping.md`, `docs/traceability.md`, `docs/provenance_crosswalk.md` | ➕ paper_mapping.md added |
| Third-party notices | `THIRD_PARTY_NOTICES.md`, `THIRD_PARTY_LICENSES/` | ✅ |
| Zenodo manifest | `ZENODO_MANIFEST.csv` (sha256 of every large binary) | ✅ |
| Unit tests | `tests/unit/*.py` | ✅ |

## Missing / caveats / to confirm

1. **No committed script bit-reproduces the typeset Figure 1 or Figure 2 PNG.** The published figures were finalized in the LaTeX project. `generate_architecture_figure.py` regenerates the *schematic* (the published Fig 1 is a polished multi-panel redrawing); `generate_effect_size_heatmap.py` regenerates Fig 2 faithfully from the artifact of record. Content is reproducible; pixels are not guaranteed identical.
2. **Figure numbering drift.** The shipped `figures/figure1..figure6_*` set is the earlier six-figure layout; the final paper has two figures. Reconciled in `docs/paper_mapping.md`; the legacy files are retained for provenance.
3. **Table 3 `Decision` column semantics differ** between the shipped CSV/MD (signed-rank significance) and the typeset Table 3 (stricter ✓ favorable rule). Same underlying data; both renderings derivable from `cluster_level_statistics.csv`. Flagged in `docs/paper_mapping.md` — not silently changed.
4. **Large binaries on Zenodo, not git** (checkpoints, prediction arrays). Reproducing Table 3 requires downloading the Zenodo prediction bundle + rebuilding processed ground truth from raw data. (The final paper has no Figure 3.)
5. **DOI placeholder.** `CITATION.cff` / README Zenodo DOI must be filled after the first Zenodo deposition.
