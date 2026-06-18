# Results artifacts

Machine-readable results backing the manuscript tables. Each CSV's **`source_artifact`
column is kept verbatim** — it records the exact path, in the author's original working
tree, where the value was computed (preserving the provenance trail and the verified
file hashes). To locate the shipped copy of any referenced path, use
[`../docs/provenance_crosswalk.md`](../docs/provenance_crosswalk.md). See
[`../docs/traceability.md`](../docs/traceability.md) for the full claim → evidence map.

| File | Backs | Notes |
|---|---|---|
| `main_test_means.csv` | Table 2 (U-Net, BDS-Lite) | filtered to in-scope methods; EGE rows dropped |
| `matched_gsl_summary.csv`, `matched_gsl_seed_results.csv` | Table 2 (U-Net+GSL) | matched-protocol Phase-16 rerun |
| `seed_level_wilcoxon_summary.csv` | seed-level stats | all Holm-adjusted p ≈ 1.0 at n = 3 |
| `resource_profile_isic2018_comparison.csv` | deploy params | +1,072 params vs U-Net |
| `profiling/` | Table 4 | training/inference cost + environment |
| `gsl/evaluations/`, `gsl/runs/`, `gsl/splits/`, `gsl/checkpoints/` (Zenodo) | GSL provenance | per-seed evals, run config/metrics, locked seed-2026 split |

The cluster-aware statistics (Table 3) and the 24-panel failure selection (Table 5)
live under [`../analysis/outputs/`](../analysis/outputs/).
