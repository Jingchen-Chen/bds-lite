# Results

Frozen outputs. The per-case results of every training run are in the primary evidence archive
(`../evidence/`); extract them into `revision_runs/` (git-ignored) as described in the
[reproduction guide](../docs/reproducing.md#archived-inputs).

| Folder | Contents |
|---|---|
| `tables/` | Statistics and summary tables used by the paper (`tables/families/`: ablation, comparator, sensitivity, expansion and confirmation families) |
| `acdc_confirm/`, `acdc_confirm_eval/`, `acdc_confirm_eval_posthoc/` | Preregistered ACDC confirmatory study: results, integrity checks, official-test evaluation, study report |
| `followup/`, `followup_eval/` | Follow-up controls E1–E3: results, post hoc analyses, summary |
| `kd_isic2018/`, `kd_acdc_synapse/`, `kd_unified/`, `kd_posthoc_balanced/`, `teacher_eval/` | External-teacher distillation: benchmarks, phase-1 validation, pooled results, post hoc balanced family, teacher evaluation |
| `official_isic/`, `historical_reevaluation/`, `prediction_recovery/`, `extra_cohorts/` | Per-case results of fixed, already-trained checkpoints, with provenance and parity checks |
| `volume/`, `surface_cdf/`, `error_analysis/`, `structural_descriptors/` | Volume-level, surface-distance, error-location and structural-descriptor analyses |
| `posthoc_restructure/`, `posthoc_review_20261112/` | Post hoc analyses (described in the evidence addenda in `../evidence/`) |
| `gates/` | Reproduction gates run before new training |
| `validation/` | Machine-readable receipts of validation checks |
| `audit/` | Checkpoint catalogue, redacted follow-up git log, Experiment C blinding check |
