# Amendment 1 to the ISIC2018 KD comparator protocol — three student seeds, exploratory

**Date:** 2026-11-07, written about 16:10 UTC and committed before the Phase 1 validation analysis was run and before any test metric of a new arm existed.
**Applies to:** `revision/KD_COMPARATOR_PROTOCOL.md` (frozen at `f7f771f`, sha256 `4320280d…`). The frozen file is left byte-identical, because its hash is checked at every launch and resume. This amendment supersedes it where they differ; it is not presented as the original plan.
**Requested by:** the author, in the instruction of 2026-11-07 to run a three-dataset study with three student seeds per dataset.

## What was known when this amendment was written

- Operational only. The teacher (seed 100) completed 150 epochs. The cache step completed, and the gate status was PASSED; this is implied by the students having started, and the gate values themselves were not read.
- D+KD-band seeds 1 and 2 had completed 150 epochs; seed 3 was at epoch 86. There had been 123 queue attempts, the restarts being verified thermal stops.
- No Phase 1 validation contrast had been computed. `analysis/kd/phase1_validation.py` had not been run, and no per-run validation metric of the new arms had been inspected.
- No test-set prediction or metric of any new arm existed. The training entry point builds no test dataset.

## Changes

1. **Seed count.** D+KD-band uses seeds 1, 2 and 3 only (the Phase 1 runs, unchanged and finished as configured). Phase 2a (seeds 4–10) and Phase 2b (D+KD-full, seeds 1–10) are **cancelled** and will not be run. The queue files `revision/queue_kd_phase2a.yaml` and `revision/queue_kd_phase2b.yaml` stay unapproved.
2. **Status.** The ISIC2018 experiment is reclassified as **exploratory**.
   - The confirmatory C1 test (fresh seeds 4–10) and the fixed-sequence C3 test are not executed.
   - No result of this experiment can carry the status "confirmed".
   - Without the D+KD-full control, no result can be attributed to the band restriction (boundary specificity is not assessable).
3. **Phase 1 → Phase 2 decision.** Option A is no longer needed for scheduling.
   - The Phase 1 validation analysis is still run unchanged (`analysis/kd/phase1_validation.py`) and reported.
   - Its feasibility checks F1–F3 now decide only whether the three runs enter the test evaluation. Failing runs are reported, not replaced.
4. **Test evaluation.** After the three D+KD-band runs and the teacher pass the integrity checks in `analysis/kd3/integrity.py`, the official test set (1,000 images) is evaluated **once** with `analysis/kd3/evaluate_test.py --dataset isic2018`. This is the same evaluator as all existing arms (`scripts/revision/train_run.py::evaluate`). The integrity checks cover:
   - 150 contiguous epochs; artifact hashes; config and source hashes equal to the registration;
   - the selected epoch equal to the first maximum of validation DSC;
   - finite losses; the teacher-cache hash; distinct seeds with differing first-epoch losses;
   - no test output present.

   The original step-8 script `analysis/kd/evaluate_test.py`, which requires ten seeds, is not used.
5. **Analysis** (`analysis/kd3/unified_analysis.py`, committed before any new-arm test metric exists; the frozen estimator `expansion_statistics.py::contrast` is unchanged).
   - **Primary (exploratory):** D+KD-band (seeds 1–3) − decoder-only (seeds 1–10), test Boundary F1, with a 95% interval and the effect label. The p value is reported unadjusted, and Holm-adjusted over the three datasets of the unified study.
   - **Secondary (descriptive):**
     - DSC, IoU, HD95 and ASSD;
     - D+KD-band against U-Net and BDS-Lite (seeds 1–10);
     - the same contrasts against seeds 1–3 of each baseline only (seed-matched sensitivity);
     - validation-set contrasts;
     - missed-structure and false-alarm rates;
     - per-seed values and seed consistency;
     - teacher test metrics;
     - training cost.
   - **Reading of an interval:**
     - wholly beyond ±0.01: "exploratory evidence of a relevant gain/harm";
     - excluding zero but overlapping ±0.01: "exploratory evidence of a gain/harm of unclear size";
     - otherwise "not established".

     No effect is called confirmed. Absence is claimed only for "no relevant effect" (interval within ±0.01), which is not expected to be reachable here.
6. **Mechanism probe.** Not run. It was planned for Phase 2 only and requires the cancelled seeds.

## Consequences stated in advance

- **Power.** With 3 runs against 10 on the test set, the expected 95% half-width is about ±0.038 to ±0.055 Boundary F1, and the 80%-power detectable difference is 0.050–0.073 (original protocol §7, row "Phase 1 design on test"). Only a large effect could be distinguished from zero. A null result here says little.
- Seeds 1–3 are the only D+KD-band runs. They are therefore not "fresh" relative to any decision, but no decision was taken on their results.
- The manuscript's existing claims are unaffected by this amendment.

## What did not change

- The three student runs, their configs and hashes, and the teacher and its cache.
- λ_kd, the band definition, θ, the evaluator, and the checkpoint rule.
- The decoder-only, U-Net and BDS-Lite reference runs, which are not retrained.
