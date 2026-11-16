# Independent experimental inventory audit

Written UTC: 2026-11-10T20:27:34.972705+00:00. No model execution, training or original-artifact changes.

Inventory rows: 659. Physical-directory training records, historical withdrawn design slots, caches, diagnostics and analysis records are distinct units. These are not automatically independent models or training realizations. The historical duplicate_of field is retained but has not been independently re-hashed.

## Current observed statuses

| Status | Count |
|---|---:|
| cache_complete | 4 |
| completed_150_epoch_record | 517 |
| completed_blinded_excluded | 14 |
| completed_output_present | 12 |
| equivalence_diagnostic | 6 |
| ledger_complete_epoch150_checkpoint_present | 9 |
| not_started_or_unavailable | 55 |
| partial_record | 2 |
| short_diagnostic | 14 |
| started_incomplete_blinded_excluded | 1 |
| failed_saved_error_record | 1 |
| withdrawn_never_run | 24 |

## Per-family coverage

| Family | Observed status counts |
|---|---|
| ACDC confirmation | completed_150_epoch_record: 34 |
| ACDC index-grid region vs paper-GSL | completed_150_epoch_record: 6 |
| ACDC phase-reduction control | completed_150_epoch_record: 6 |
| ACDC three-index context control | completed_150_epoch_record: 3 |
| E1 long-schedule GSL | partial_record: 1 |
| EGE-UNet compact baseline | ledger_complete_epoch150_checkpoint_present: 9 |
| ISIC D4 augmentation baseline | completed_150_epoch_record: 3 |
| ISIC U/S/B/F factorial | completed_150_epoch_record: 12 |
| ISIC boundary-specificity control | completed_150_epoch_record: 3 |
| ISIC pathway controls | completed_150_epoch_record: 6 |
| Phase 16 matched GSL | completed_150_epoch_record: 9 |
| Stage 1 expansion | completed_150_epoch_record: 40 |
| Stage 2 Experiment C | completed_blinded_excluded: 14; not_started_or_unavailable: 55; started_incomplete_blinded_excluded: 1 |
| WP2/WP3 Tier B | completed_150_epoch_record: 23 |
| WP2/WP4 queue A | completed_150_epoch_record: 46 |
| WP4 queue B | completed_150_epoch_record: 27 |
| boundary readout probes | completed_output_present: 1 |
| cohort shift | completed_output_present: 1 |
| distance-missingness sensitivity | completed_output_present: 1 |
| error location | completed_output_present: 1 |
| exploratory external KD students | completed_150_epoch_record: 9 |
| exploratory external teacher | completed_150_epoch_record: 3 |
| exploratory teacher cache | cache_complete: 3 |
| failed launch | failed_saved_error_record: 1 |
| family O reproducibility grid | completed_150_epoch_record: 117 |
| follow-up E1/E2 students | completed_150_epoch_record: 34 |
| follow-up E3 augmentation | completed_150_epoch_record: 20 |
| follow-up compact teacher/cache | cache_complete: 1; completed_150_epoch_record: 1 |
| gate identity intervention | completed_output_present: 1 |
| gate structure probes | completed_output_present: 1 |
| gradient probes | completed_output_present: 1 |
| near-duplicate sensitivity | completed_output_present: 1 |
| p1 gate pilot (single seed) | completed_150_epoch_record: 15 |
| p1b gate-free multi-seed | completed_150_epoch_record: 36 |
| p2 recent-loss baselines (GSL, Boundary DoU) | completed_150_epoch_record: 18; partial_record: 1 |
| p2b BDS-Lite(no gate)+GSL | completed_150_epoch_record: 9 |
| reference estimators | completed_output_present: 1 |
| reproduction gate | short_diagnostic: 12 |
| resource profiles | completed_output_present: 1 |
| shared-variance sensitivity | completed_output_present: 1 |
| single-seed exploration | completed_150_epoch_record: 37 |
| smoke | short_diagnostic: 2 |
| stacked-volume evaluation | completed_output_present: 1 |
| trainer equivalence diagnostic | equivalence_diagnostic: 6 |
| withdrawn registration | withdrawn_never_run: 24 |

## Scope limits and defensible counts

- The old 536-row ledger has 512 nonempty registered paths and 24 withdrawn designs without run paths. All 425 ledger-status-complete run directories exist; 55 registered Experiment C paths are absent. The old ledger status counts are stale and must not be presented as current completion counts.
- Primary histories/logs are read for maximum epoch for non-blinded records. A 150-epoch record establishes recorded completion, not data provenance or an independently unique stochastic realization. Checkpoint-only completion is explicitly weaker.
- Experiment C: 14 completed final-marker filenames, one partial folder, 55 absent run folders. No C final_metrics/history/test metric content is read. All C outcomes remain excluded.
- The nonoverlapping core DESIGN count is 237 intended training slots (earlier217 plus20 E3), including four teachers; reused references count once. This is separate from all-history record totals. E3 test evaluation is audited separately. It is not valid to add analysis-record rows, reused reference slots, caches or gate interventions to training totals.
- Historical aliases, re-evaluated checkpoints and exact reproductions exist. The old ledger reports379 tensor-distinct and approximately358 independent realizations among425 historical completed records, but those historical reductions were NOT independently rehashed here and remain REPORTED only.
- Secondary diagnostic/sensitivity/resource outputs are inventoried by existence and row count, not certified numerical correctness. Their original source provenance remains necessary.
- Lost original trainer, missing historical checkpoints, source-data/teacher-cache portability, label semantics, and cohort/reconstruction effects remain unresolved.

## Inventory tooling failures and continuation

- A read-only grouping query failed with IndexError on an empty-path withdrawn design slot; no file was written. Empty-path design records are explicitly handled.
- Initial CSV packaging failed with ValueError: dict contains fields not in fieldnames: trained. It wrote only the header of this new audit CSV. The authorized independent packaging continuation includes the actual trained field and preserves all records; no scientific assertion or result was changed.

## Parse uncertainties

None in the primary epoch trajectory parses.

## Count reconciliation

There are 526 non-blinded completed training RECORDS: 517 with independently
read 150-epoch trajectories and nine EGE records with a historical completion
claim plus an epoch-150 checkpoint filename. Fourteen additional completed
Experiment C marker records are blinded/excluded, giving 540 completed-record
claims if excluded C is counted. These totals include aliases/re-evaluations and
are explicitly not counts of independent trained models. The additional
non-ledger completed training records number101: 34 confirmation students,
34 E1/E2 students, one compact teacher,20 E3 students, nine exploratory KD
students and three external teachers. Four cache steps and six equivalence
diagnostics are separate non-training records.

The known AMP-failed launch is backed by its saved status/error record and is
classified as failure, rather than an unresolved completed experiment. The two
partial non-blinded records are the aborted historical long-schedule GSL attempt
(epoch6) and a ten-epoch P2 verification, not incompletely finished current
confirmatory runs.
