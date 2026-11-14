# Frozen-teacher evaluation on official ACDC patients 101-150: pre-specified protocol

Written 2026-11-11, **before** the sanity checks, the new teacher inference and the analysis
of this task were run. Author request (task brief "Evaluate Frozen Teacher Models on the ACDC
Official Test Set", 2026-11-11). Inference only: no model is trained or retrained.

Status: post hoc, descriptive. Patients 101-150 have already been examined for every student
group. This is not part of a registered protocol and changes no registered estimator, decision
or claim (`revision/CLAIM_BOUNDARY_v4.md` C04-C06, C09, C13 remain governing).

## Disclosure of prior exposure

Both teachers were evaluated once before on these patients with the same evaluator:
external teacher on 2026-11-09 (`results/acdc_confirm_eval_posthoc/acdc_kd_teacher_seed100/`,
post hoc, author decision 7) and compact teacher on 2026-11-10 (in the registered E1/E2
evaluation job, `results/followup_eval/acdc_followup_e2_teacher_seed100/`). While locating
the checkpoints for this task, the existing external-teacher record was opened; its
`evaluation_record.json` shows a slice-class **row-level** Boundary F1 mean of 0.9000 (which
includes both-empty rows scored 1 and is not the patient-level aggregate used below). The
interpretation rules below are copied verbatim from the brief and are not modified.

## Models

| Teacher | Run | best.pt sha256 | Used by |
|---|---|---|---|
| external | `results/revision_runs/acdc_kd_teacher_seed100` | `735747303c444e13dd57457e3540ee410b5de43421b68fd92e2eda69c88dafa0` | D+KD-band, D+KD-full, D+KD-band_aug (cache `acdc_kd_teacher_cache`) |
| compact | `results/revision_runs/acdc_followup_e2_teacher_seed100` | `0bf28bcc7bce3ca75cc8fb3fbba277120c92702b78edb7979c147893df1f7c1e` | D+KD-band_c (cache `acdc_followup_e2_teacher_cache`) |

## Evaluation (identical to the student evaluation)

- Data: `scripts/acdc_confirm/evaluate_test.py::official_loader` (manifest and every NPZ
  sha256-checked against `revision/acdc_confirm/official_test_preprocessing.json`; batch 16,
  4 workers, no shuffle).
- Model loading: external teacher `scripts/kd_mc/teacher.py::build_teacher(config,
  pretrained=False)` (single channel replicated to three channels, `input_norm: none` for ACDC,
  as in its training); compact teacher `scripts/acdc_confirm/evaluate_test.py::load`
  (`bds_lite.training.build_model`, as for students). Strict state-dict load.
- Evaluator: unchanged `scripts/revision/train_run.py::evaluate` (argmax only, no TTA, no
  thresholding, no postprocessing; per-slice, per-foreground-class `case_metrics`).
- Aggregation: unchanged `analysis/revision/expansion_statistics.py::unit_values` (exclude
  both-empty class pairs, exclude undefined HD95/ASSD, mean over classes per slice, mean over
  slices per patient, mean over 50 patients); per-structure values with
  `analysis/kd3/unified_analysis.py::label_values`.

## Mandatory sanity checks (before teacher inference)

Recompute every student group's run-level Boundary F1 from saved predictions (recomputing
`case_metrics` against the official masks and requiring equality with the saved per-case
files) and aggregate with `unit_values`. Required, absolute tolerance 1e-4:
D (17) 0.7797; D+KD-band (17) 0.7975; D_aug (10) 0.8469; D+KD-band_aug (10) 0.8432 (paper,
`R@E3.E3.boundary_f1.mean.a`). If any fails, stop; do not run or report the teachers.

## Comparisons (descriptive only; no significance tests)

Groups: D, D+KD-band, D+KD-full, D+KD-band_c (17 runs each); D_aug, D+KD-band_aug (10 each).
Per group: mean/min/max of run-mean BF1; number of runs whose run-mean BF1 exceeds the
teacher's; per patient, teacher BF1 minus the group's across-run mean BF1 for that patient;
mean over the 50 patients and a 95% percentile bootstrap CI from 10,000 patient resamples with
replacement (`numpy.random.default_rng(20261111)`).

## Pre-specified interpretation (verbatim from the brief; applied to the external teacher)

**Outcome A — Consistent with the explanation.** If both conditions hold:
- The external teacher's Boundary F1 is lower than the mean Boundary F1 of D_aug.
- The 95% confidence interval for the paired difference (teacher minus D_aug) lies entirely
  below zero.

Conclude: "Augmented students outperform the teacher, consistent with the disappearance of the
distillation benefit."

**Outcome B — Explanation not supported.** If the external teacher's Boundary F1 is greater than
or equal to the mean Boundary F1 of D_aug, conclude: "This explanation is not supported."

**Outcome C — Inconclusive.** In all other cases, conclude: "Inconclusive."

"Boundary F1" for the teacher and "mean Boundary F1 of D_aug" are the patient-level aggregates
defined above (mean of the 10 D_aug run means). Report the outcome exactly; no alternative
explanations are searched for.
