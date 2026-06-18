# Cluster-aware analysis

`generate_rescue_analysis.py` produces the manuscript's primary cluster-aware
evidence and the deterministic 24-panel failure-case selection. The statistical
methods and the bootstrap-RNG caveat are documented in [`../docs/statistics.md`](../docs/statistics.md).

## Artifacts of record (do not regenerate to "make them match")

The committed files in [`outputs/`](outputs/) are the **artifacts of record**:

- `cluster_level_statistics.csv` — manuscript Table 3 (per-dataset, per-metric
  cluster bootstrap CIs, Holm-adjusted signed-rank p-values, sign effect sizes,
  leave-one-cluster-out ranges).
- `failure_case_manifest.csv` — the 24 deterministically selected failure cases
  (rendered in `../figures/failure_panels/`).
- `per_case_metrics.csv`, `per_class_case_metrics.csv`, and the per-case /
  subgroup / outlier / tradeoff / correlation summaries.
- `analysis_run_summary.json` — run provenance (sample and cluster counts).

## Reproduction status (verified)

Running the full script end-to-end regenerates every committed output. This was
verified against the source working tree:

- `cluster_level_statistics.csv` and `failure_case_manifest.csv` reproduced
  **bit-for-bit (identical sha256)**, including the bootstrap CI endpoints — the
  RNG caveat in `../docs/statistics.md` holds because the script is run end-to-end.
- All 12 supporting per-case tables reproduced **bit-for-bit** as well.
- Sample/cluster counts: ISIC 519 (val), ACDC 366 (test), Synapse 1568 (test);
  clusters 519 / 20 / 12; 2453 case rows; 24 failure panels.

## Inputs

- **Per-case predictions** (`outputs/evaluations/predictions/<dataset>_<method>_seed<n>_<split>/*.npy`)
  for `unet` and `bds_lite_full`, three seeds — distributed via Zenodo (see
  `../ZENODO_MANIFEST.csv`). Place them under `outputs/evaluations/predictions/`.
- **Processed ground truth** (`data/processed/<dataset>/<split>/*.npz`) — rebuilt
  locally from the raw datasets (see `../docs/data_access.md`).
- **Supporting-table inputs** — `inputs/p1b_gate_comparison_summary.csv` (gate-removal
  comparison) and `../results/resource_profile_isic2018_comparison.csv` (deployment
  resource profile), both vendored/shipped.

## Difference from the original internal script

The original script additionally emitted three GSL supporting tables
(`gsl_validation_summary.csv`, `gsl_test_summary.csv`, `gsl_holm_comparisons.csv`)
built from a **superseded** set of GSL baseline evaluations and a `manuscript_v2`
Holm family. Those inputs are **not** part of this release and did **not** feed the
manuscript's reported GSL numbers — the manuscript's GSL comparator is the
matched-protocol Phase-16 rerun in [`../results/`](../results/) and
[`../results/gsl/`](../results/gsl/). That branch has been removed from
`supporting_tables()`, and the corresponding three counts were dropped from
`analysis_run_summary.json`. Nothing else changed; all artifacts of record are
unaffected.
