# Revision workspace map (read this first)

Files are not moved, because training configs, queues and provenance records reference exact paths
and hashes. This index states which documents are authoritative.

Current editorial status: `EDITORIAL_REVIEW_20261108.md`; current reviewer mapping: `REVIEWER_COVERAGE.md`. These supersede dated status descriptions elsewhere for the submission text.

## Authoritative now
| Topic | File |
|---|---|
| Governing plan | `plan.md` (copy of the PeerJ major-revision plan) |
| What the submitted results are built from; confirmed problems; retractions | `GITHUB_AUDIT.md` |
| Current strategy, experiment necessity, priorities | `REVISION_STRATEGY.md` |
| Decisions needed from the authors | `HUMAN_DECISIONS.md` |
| Task status (generated) | `STATUS.md`, `tasks.csv` (`analysis/revision/refresh_status.py`) |
| Registered analysis plan | `../analysis/revision/prereg_revision.md` |
| Training protocols (hashed into configs) | `TRAINING_ADAPTER_PROTOCOL.md`, `ABLATION_PROTOCOL.md` + `ABLATION_PROTOCOL_AMENDMENT_1.md`/`_2.md`, `COMPARATOR_PROTOCOL.md`, `SENSITIVITY_PROTOCOL.md` |
| Revised manuscript / supplement | `manuscript/main.tex`, `manuscript/supplement.tex` (numbers in `manuscript/generated/`, never typed by hand) |
| Response letter | `response_to_reviewers.tex` / `.pdf` |
| Verified literature numbers | `lit_numbers.csv` |
| Rewrite assessment, validation, evidence map | `REWRITE_REPORT.md` |
| Reviewer-coverage matrix | `REVIEWER_COVERAGE.md` |
| Build (fails on crash or stale PDF) | `build_check.sh` |

## Superseded or historical (kept for provenance)
`FAMILY_PROVENANCE.md` (decision section superseded), `response_letter.tex` (old scaffold), `response_letter_v2.tex` / `.pdf` (superseded response),
`REMAINING_WORK.md`, `COMPLETED_WORK.md`, `OFFICIAL_ISIC_FINDINGS.md`, `VOLUME_FINDINGS.md`,
`GATE_IDENTITY_FINDINGS.md`, `SURFACE_ERROR_FINDINGS.md` (these four analyse the p1b family, now the
replication family), `manuscript/verified_results_draft.tex`, `manuscript/*_draft.tex`.

## Method keys used in all generated tables
`S0/S6` submitted U-Net/BDS-Lite (primary) · `S6g` submitted BDS-Lite with gate→identity at inference ·
`M0/M6` pipeline-matched references · `M1-M5` ablations · `GSL, C1, C3, C4` comparators ·
`R0/R6` replication family. Registry: `../analysis/revision/family_statistics.py`.

## Where results live
| Kind | Location |
|---|---|
| New training runs | `../results/revision_runs/<run_id>/` (final_metrics.json, preds, per-case CSV) |
| Submitted-model evaluations | `extra_cohorts/famO_*` |
| Official ISIC (p1b, followup arms) | `official_isic/` |
| Family statistics, collapse, error bands, native Jaccard, volume families | `tables/families/` |
| Figures | `figures/` |

## Live processes (do not kill)
`analysis/revision/supervise_gradient_queue.py` / `supervise_queue_v2.py` (GPU queue; stop only via
`touch revision/STOP_TRAINING`), `analysis/revision/auto_refresh.py` (stopped before the 2026-11-08 manuscript rewrite; keep it stopped during editorial work).
Never edit files under `src/bds_lite`, `scripts/revision`, `scripts/comparators` or registered configs
while runs are queued: they are hash-checked and resume would refuse.

Current comparison provenance: `validation/comparison_absolute_scores.csv` and
`validation/comparison_table_verification.json`; generator `analysis/revision/make_comparison_table.py`.
Author-requested post hoc balanced KD display: `kd_posthoc_balanced/{contrasts.csv,analysis.json}`;
original frozen analysis remains `kd_unified/`; confirmation remains `acdc_confirm/results/`.
