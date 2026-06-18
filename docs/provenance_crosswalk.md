# Provenance crosswalk (original working tree → public repo)

The shipped tables and result CSVs keep their **`source_artifact` columns verbatim**:
they record the exact path, in the author's original working tree, where each number
was computed. That is the faithful provenance trail. Because this public repository
reorganizes artifacts into clean paths, use the table below to locate the shipped copy
of any referenced file. The numbers are byte-identical; only the location changed.

| Original path (as cited in `source_artifact`) | Public-repo location |
|---|---|
| `results/midterm_q2_rescue/analysis/*.csv` | `analysis/outputs/*.csv` |
| `results/midterm_q2_rescue/analysis/analysis_run_summary.json` | `analysis/outputs/analysis_run_summary.json` |
| `results/midterm_q2_rescue/scripts/generate_rescue_analysis.py` | `analysis/generate_rescue_analysis.py` |
| `results/midterm_q2_rescue/figures/failure_cases/*.png` | `figures/failure_panels/*.png` |
| `results/final_manuscript_audit_v2/recomputed_main_test_means.csv` | `results/main_test_means.csv` (filtered to U-Net + BDS-Lite; EGE rows dropped) |
| `results/final_manuscript_audit_v2/seed_level_wilcoxon_summary.csv` | `results/seed_level_wilcoxon_summary.csv` |
| `results/submission_blocker_resolution_v1/gsl_memory_bounded/matched_gsl_summary.csv` | `results/matched_gsl_summary.csv` |
| `results/submission_blocker_resolution_v1/gsl_memory_bounded/matched_gsl_seed_results.csv` | `results/matched_gsl_seed_results.csv` |
| `results/submission_blocker_resolution_v1/gsl_memory_bounded/evaluations/*.csv` | `results/gsl/evaluations/*.csv` |
| `results/submission_blocker_resolution_v1/gsl_memory_bounded/configs/*.yaml` | `configs/gsl/*.yaml` (paths remapped) |
| `results/submission_blocker_resolution_v1/gsl_memory_bounded/runs/*/{config.json,metrics.jsonl}` | `results/gsl/runs/*/` (paths remapped) |
| `results/submission_blocker_resolution_v1/gsl_memory_bounded/checkpoints/*/best.pt` | Zenodo bundle `checkpoints_gsl` → `results/gsl/checkpoints/*/best.pt` |
| `results/submission_blocker_resolution_v1/profiling/*` | `results/profiling/*` |
| `results/submission_critical_evidence_v1/profiling/training_cost_summary.csv` | combined into `results/profiling/training_cost_combined_summary.csv` |
| `results/submission_critical_evidence_v1/figures/figure{1,3,4,5,6}_*.{pdf,png}` | `figures/figure{1,3,4,5,6}_*.{pdf,png}` |
| `results/submission_candidate_manuscript_v1/figures/figure2_matched_main_results.*` | `figures/figure2_matched_main_results.*` |
| `results/submission_candidate_manuscript_v1/manuscript/*` | `manuscript/*` |
| `results/submission_candidate_manuscript_v1/tables/*` | `manuscript/tables/*` |
| `results/tables/resource_profile_isic2018_comparison.csv` | `results/resource_profile_isic2018_comparison.csv` |
| `results/post_recovery_eval/p1b_gate_comparison/summary.csv` | `analysis/inputs/p1b_gate_comparison_summary.csv` |
| `outputs/runs/<run>/config_resolved.yaml` | `configs/run_resolved/<run>.yaml` |
| `outputs/runs/<run>/best.pt` | Zenodo bundle `checkpoints_unet_bdslite` → same path |
| `outputs/evaluations/predictions/<dir>/*.npy` | Zenodo bundle `predictions` → same path |
| `data/splits/<dataset>/*.json` | `splits/<dataset>/*.json` |

## Original artifacts deliberately NOT shipped

| Original path | Why omitted |
|---|---|
| `results/post_recovery_eval/gsl_baselines/`, `results/next_stage/p2_recent_baselines/` | Superseded GSL/baseline evaluations; not used by the manuscript (its GSL comparator is the matched Phase-16 rerun). See `analysis/README.md`. |
| `manuscript_v2/` | A different, non-target manuscript; excluded entirely. |
| `splits/isic2018_split.json` (seed-1, 1815/389/390) | Stale split; superseded by the seed-2026 manifests in `splits/`. See `docs/naming.md`. |
| `*.log`, `*.egg-info`, build caches | Logs (avoid path leaks), build cruft. |
