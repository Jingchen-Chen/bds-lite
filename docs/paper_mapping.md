# Paper → repository mapping

This file maps every table and figure in the **final manuscript** (`When does boundary distillation help lightweight medical image segmentation?`) to the configs, scripts, and machine-readable artifacts that produce it. It is the single place that reconciles the paper's numbering with the repository contents.

The final paper has **4 tables** and **2 figures**. (An earlier draft had 6 figures; those source files are retained under `figures/figure1..figure6_*` for provenance — see the figure section below.)

For the underlying claim → evidence map see `manuscript/tables/table6_claim_evidence_final.csv`; for original-path → public-path resolution see `docs/provenance_crosswalk.md`; for statistical methods see `docs/statistics.md`.

## Methods → configs

| Manuscript element | Config(s) of record |
|---|---|
| U-Net baseline, 3 seeds, 3 datasets | `configs/run_resolved/{isic2018,acdc,synapse}_unet_seed{1,2,3}.yaml` |
| BDS-Lite (full recipe), 3 seeds, 3 datasets | `configs/run_resolved/{isic2018,acdc,synapse}_bds_lite_full_seed{1,2,3}.yaml` |
| U-Net+GSL comparator (matched, Phase-16), 3 seeds | `configs/gsl/phase16_matched_gsl_{isic2018,acdc,synapse}_seed{1,2,3}.yaml` |
| Dataset-level settings (size, normalization, classes) | `configs/datasets/{isic2018,acdc,synapse}.yaml`, `configs/base.yaml` |
| Loss weights (λ_ce=1, λ_dice=1, λ_sdf=1, λ_b=0.05, λ_distill=0.05, λ_gsl=1) | encoded in the run-resolved + gsl configs above |
| Locked split manifests (split seed 2026) | `splits/{isic2018,acdc,synapse}/*.json` |

See `docs/config_of_record.md` for the `model.name: bds_lite` alias and the vestigial `dataset.split_file` field.

## Tables

| Table | Title | Artifact of record | Producing script |
|---|---|---|---|
| **Table 1** | Datasets and locked protocol | `manuscript/tables/table1_dataset_protocol.csv`, `splits/` | split manifests (seed 2026); guarded by `tests/unit/test_split_integrity.py` |
| **Table 2** | Matched three-seed main results | `results/main_test_means.csv` (U-Net, BDS-Lite) + `results/matched_gsl_summary.csv`, `results/matched_gsl_seed_results.csv` (U-Net+GSL) | `scripts/reeval_main_seeds.py`; GSL via `scripts/gsl/run_phase16_gsl.sh` + `scripts/gsl/aggregate_phase16_gsl.py` |
| **Table 3** | Cluster-aware statistical summary (BDS-Lite vs U-Net) | `analysis/outputs/cluster_level_statistics.csv` | `analysis/generate_rescue_analysis.py` (single end-to-end run; bit-for-bit verified) |
| **Table 4** | Training and inference resource profile | `results/profiling/training_cost_combined_summary.csv`, `results/profiling/environment.json`, `results/resource_profile_isic2018_comparison.csv` | `scripts/profile_model.py`; GSL step times via `scripts/gsl/profile_memory_bounded_gsl.py` |

Note: Table 3's `Decision` column in the shipped CSV/MD records signed-rank significance (`significant` / `not significant`), whereas the typeset Table 3 in the paper applies the stricter favorable-decision rule (✓ requires Holm significance **and** a bootstrap CI excluding zero **and** a coherent direction — so only ISIC2018 HD95 and Boundary F1 are ✓). Both views derive from the same `cluster_level_statistics.csv`; the columns `mean_oriented_delta`, `bootstrap_ci_low/high`, and `p_value_holm` contain everything needed to reproduce either rendering.

## Figures

| Figure | Title | Content source | Generator |
|---|---|---|---|
| **Figure 1** | BDS-Lite architecture and training pipeline | schematic (code) | `scripts/figures/generate_architecture_figure.py` → `figures/figure_architecture_schematic.*` |
| **Figure 2** | Cluster-level BDS-Lite-vs-U-Net effect-size heatmap | `analysis/outputs/cluster_level_statistics.csv` (= Table 3) | `scripts/figures/generate_effect_size_heatmap.py` → `figures/figure_cluster_effect_size_heatmap.*` |
| **(removed) Figure 3** | The final paper has **no Figure 3**. Representative/failure cases are reported in the text and in `analysis/outputs/failure_case_manifest.csv` (real-image panels generated locally, **not redistributed**) | `analysis/generate_rescue_analysis.py` |

**Caveat (publication vs reproduced figures).** The figures *typeset in the manuscript* were finalized in the LaTeX project (a polished multi-panel Figure 1 whose input panel is a synthetic image). The scripts above regenerate the **content** — the architecture structure and the heatmap values — from the artifacts of record, but are not guaranteed pixel-identical to the typeset exports. There is **no committed script that bit-reproduces the typeset Figure 2 PNG**; `generate_effect_size_heatmap.py` reproduces it faithfully from the artifact of record.

**Legacy six-figure set.** `figures/figure1_architecture_training_inference_decoupling`, `figure2_matched_main_results`, `figure3_per_case_delta_distributions`, `figure4_boundary_distance_relationships`, `figure5_subgroup_organ_heatmap`, and `figure6_failure_cases` belong to a previous six-figure manuscript layout. They are kept for provenance and are described in `figures/captions.md`. The final paper uses the two figures mapped above; the legacy `figure6_failure_cases` is **not** used in the final paper (which has no Figure 3).

## Key numbers (must stay consistent across paper and docs)

| Quantity | Value | Where in repo |
|---|---|---|
| U-Net parameters | 1,927,042 | `results/resource_profile_isic2018_comparison.csv`, Table 4 |
| BDS-Lite total (training graph) params | 2,198,003 | Table 4 |
| BDS-Lite deployed (aux-removed) params | 1,928,114 | Table 4, README |
| Deploy params above U-Net | 1,072 (+0.056%) | Figure 1 caption, Table 4, README |
| Inference FLOPs increase | ≈0.68% (15.970 → 16.079 GFLOPs) | Table 4 |
| Deploy param difference (rounded) | ≈0.06% | README, `docs/compute.md` |
| Seeds / split seed | 1,2,3 / 2026 | `docs/compute.md`, `splits/` |
