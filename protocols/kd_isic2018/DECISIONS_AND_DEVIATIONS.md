# KD comparator: dated decisions and deviations

`revision/KD_COMPARATOR_PROTOCOL.md` is frozen and hash-checked at every launch, so it is not edited after the freeze. This file records, with dates, the freeze commit, the Phase 2 decision (§6, Option A), and any deviation. For each entry it states which results had been seen at the time.

## Freeze

| Date (UTC) | Record |
|---|---|
| 2026-11-07 | Protocol, implementation, configs, queues and tests committed together before any teacher or student training step. |
| 2026-11-07T13:26:21Z | Freeze commit `f7f771f489b5e9238cf58c88de6692ef84a932da`; protocol sha256 `4320280d58c2b89caf17c970944c156d4cef9f0ce8fb1488ffd0ebd7aa30b3b9`. No teacher or student training step and no new-arm validation or test metric existed. |

## Phase 2 decision (Option A)

Superseded by Amendment 1 (2026-11-07): Phase 2 is cancelled. The Phase 1 feasibility checks F1–F3 decide only whether the three runs enter the test evaluation.

## Deviations

| Date | Section | Change | Reason | Before/after which results were seen |
|---|---|---|---|---|
| 2026-11-07 ~16:10 UTC | §§6, 7, 9, 13 | Amendment 1 (`revision/KD_COMPARATOR_AMENDMENT_1.md`): three D+KD-band seeds only; Phase 2a/2b cancelled; experiment reclassified as exploratory; test evaluation of seeds 1–3 after integrity checks; analysis in `analysis/kd3/`. | Author instruction (three-dataset study with three seeds per dataset; compute budget). | Before: no Phase 1 validation analysis run, no new-arm validation metric inspected, no new-arm test metric in existence. Known: teacher and seeds 1–2 complete, seed 3 at epoch 86, gate PASSED (students started). |

## Execution record after Amendment 1 (no further change)

| Date (UTC) | Record |
|---|---|
| 2026-11-07 | ISIC2018 integrity checks PASSED (`revision/kd_unified/integrity_isic2018.json`). The test set was evaluated once at 16:33:18Z, after the ACDC/Synapse protocol freeze (`ed08782`, 16:31:51Z). Unified analysis was rerun with all three datasets at 20:43Z; the ISIC2018 values are unchanged, and the primary Holm-adjusted p is now over three datasets (0.496). |
