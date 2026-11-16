# KD multiclass study (ACDC, Synapse): dated decisions and deviations

`revision/KD_MULTICLASS_PROTOCOL.md` is frozen and hash-checked at every launch, so it is not edited after the freeze. This file records, with dates, the freeze commit and any deviation, and states which results had been seen at the time.

## Freeze

| Date (UTC) | Record |
|---|---|
| 2026-11-07T16:31:51Z | Freeze commit `ed087825ec4b89085030dee68a34a0c522ed3fa0`; protocol sha256 `699d59875fbe7aa0e9b663938bb64ef88a963e262e651499c962b4dd30d7ea53`. No ACDC/Synapse teacher or student training step and no new-arm test metric of any dataset existed. ISIC2018 Phase 1 validation results (commit `efee090`) were known. |

## Deviations

| Date | Section | Change | Reason | Before/after which results were seen |
|---|---|---|---|---|
| — | — | None. | — | — |

## Execution record (no protocol change)

| Date (UTC) | Record |
|---|---|
| 2026-11-07T16:37:19Z – 20:40:58Z | Queue `revision/queue_kd_mc.yaml` ran to `queue_finished` under `scripts/kd_mc/supervise_mc.py`: 115 queue attempts and 114 thermal stop/cool/resume cycles, with no halt. Code, configs, queue and protocol were byte-identical to `ed08782` (`git diff ed08782` empty) when the queue finished. |
| 2026-11-07 | Teacher gates PASSED. ACDC teacher val Boundary F1 0.9014 / DSC 0.8420 against thresholds 0.8356 / 0.7885. Synapse 0.9057 / 0.9008 against 0.8447 / 0.8399. Both cache checks passed (max abs diff 2.5e-4, tolerance 1e-3). |
| 2026-11-07T20:41Z | `analysis/kd3/integrity.py` PASSED for ACDC and Synapse, with no problems and feasibility F1–F3 true. Before this point no ACDC/Synapse new-arm test output existed. |
| 2026-11-07T20:41:32Z (ACDC), 20:42:16Z (Synapse) | One-time test evaluation with `analysis/kd3/evaluate_test.py`: three students and the teacher per dataset, each evaluated once. Post-test integrity re-check passed (`revision/kd_unified/integrity_*_post_test.json`). |
| 2026-11-07T20:43Z | `analysis/kd3/unified_analysis.py --datasets isic2018 acdc synapse`: outputs in `revision/kd_unified/`. No hyperparameter, run or analysis change was made after any test result. |
| Interpretive note (not a change) | The frozen pooled estimate is fixed-effect inverse-variance with a normal (±1.96 SE) interval. Its weights come from per-dataset SEs estimated on about 2–3 Welch df, so the pooled interval is optimistic and is read as descriptive only. |
