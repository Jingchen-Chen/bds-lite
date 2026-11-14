# Registration: exploratory follow-up controls for D+KD-band on ACDC (E1, E2, optional E3)

**Status: FINAL, 2026-11-09.** Written and committed before any follow-up training run. After
the commit that first contains this file (the *registration commit*), this file is not edited.
Its sha256 is embedded in every follow-up config and in both queues, so any edit blocks launch
and resume. Deviations and operational events are appended to `revision/followup/PROTOCOL_LOG.md`
with UTC timestamps.

**Authorization.** The author requested these experiments on 2026-11-09 (follow-up brief:
"When does boundary distillation help?"). The commands are listed in `PLAN.md`, Phase 20.

**Manuscript.** PeerJ Computer Science CS-2026:06:142937 (revision), *When does boundary
distillation help lightweight medical image segmentation?*

## 1. Status of these experiments: secondary and exploratory

- **Test patients already examined.** All follow-up arms are evaluated on the same 50 official
  ACDC test patients (101–150) that the pre-registered Stage 3 replication used. The Stage 3
  results (D and D+KD-band on these patients) were analysed and read on 2026-11-08, before this
  registration was written.
- **Consequence.** Every follow-up comparison is **secondary and exploratory**. None is a new
  confirmatory test, none can change the Stage 3 decision, and none is described as
  confirmatory in the manuscript. The manuscript states this distinction explicitly.
- **What this registration does fix in advance.** The arms, seeds, endpoints, estimator,
  multiplicity family, the failed-run policy, and the mapping from outcomes to manuscript
  interpretations (§9), all before any follow-up run is trained or evaluated.
- **Unchanged.** `preregistration/acdc_confirmation.md`, its decision, its wording, and every
  Stage 3 artifact stay as they are. The 34 Stage 3 runs are not retrained, re-evaluated or
  modified.

## 2. Question

Stage 3 found that D+KD-band raised test Boundary F1 relative to the decoder-only arm (D) by
+0.0178 [95% CI +0.0098, +0.0258] on these patients, with a DSC difference of similar size
(+0.0136). Three properties changed together between D and D+KD-band: (i) the loss is restricted
to a band around the teacher's contours; (ii) the teacher is a different, larger
(24,436,804 parameters), ImageNet-pretrained network; (iii) the student is trained on soft
targets at all. The follow-up asks:

- **E1 (band restriction).** Does restricting the KD loss to the boundary band matter, compared
  with the same KD loss over the whole image from the same teacher?
- **E2 (teacher advantage).** Does the gain require a teacher with an informational advantage
  (pretraining and capacity, which remain confounded with each other), or does a teacher of the
  student's own architecture, trained without pretraining by the D recipe, give a gain too?

## 3. Arms

All arms: ACDC, the Stage 3 training data, split manifests, model, segmentation and boundary
losses, optimiser, schedule, batch size, AMP, gradient clipping, 150 epochs and checkpoint rule
(first maximum of validation DSC), through the frozen `scripts/kd_mc/train_mc.py`. Each config is
the Stage 3 generator output (`scripts/acdc_confirm/common.make_config`) with only `project.seed`,
`revision.*`, `kd.mode` and `kd.cache_run` changed (`scripts/kd_followup/common.make_config`;
tested in `tests/revision/test_kd_followup.py`). Run names: `acdc_followup_<arm>_seed<N>`.

### Reused Stage 3 arms (not retrained)

| Arm | Runs | Test outputs |
|---|---|---|
| D (decoder-only) | `results/revision_runs/acdc_confirm_decoder_seed{201..217}` | `results/acdc_confirm_eval/acdc_confirm_decoder_seed*/per_case_official_test.csv` |
| D+KD-band | `results/revision_runs/acdc_confirm_kd_band_seed{201..217}` | `results/acdc_confirm_eval/acdc_confirm_kd_band_seed*/per_case_official_test.csv` |

The hashes of these test outputs are checked against `results/acdc_confirm_eval/evaluation_record.json`
before analysis.

### E1: D+KD-full (`kd_full`; seeds 201–217; 17 runs)

- Same teacher and the same frozen teacher cache as D+KD-band
  (`results/revision_runs/acdc_kd_teacher_cache`, sha256 `42266ea0…077e`), λ_kd = 1, no
  temperature, both distributions clamped to [1e-6, 1 − 1e-6], fp32 with autocast disabled.
- The only change: the region B is every pixel of every image in the batch instead of the band.
  L_kd = Σ_{n,u} CE_{n,u} / (N·H·W + 1e-6), the per-pixel soft cross-entropy averaged over all
  pixels, including background and slices without foreground. Implementation: the launcher
  rebinds `scripts.kd_mc.loss.class_band` to an all-ones map for that process only
  (`scripts/kd_followup/launch.py`).
- **Note on loss scale.** The per-pixel average is the natural "same loss over the whole image".
  Because most pixels are easy background on which teacher and student agree, its value and
  gradient per image differ from the band loss at λ_kd = 1. This is a property of the control as
  specified, not tuned, and is reported (KD gradient probes at epochs 0, 1, 25, …, 150, as for
  Stage 3).

### E2: D+KD-band with a compact teacher without an informational advantage

- **Teacher** (`e2_teacher`, run `acdc_followup_e2_teacher_seed100`; trained once): the student
  architecture (`bdslite_unet`, base width 32, no pretraining) trained with the exact D recipe
  (the Stage 3 decoder-only config with seed 100, which is not a student seed), then frozen.
- **Cache** (`acdc_followup_e2_teacher_cache`): `scripts/kd_followup/cache_peer.py`, the same
  steps as the frozen `scripts/kd_mc/cache_mc.py` (its `infer()` is reused unchanged): fp32
  inference with TF32 off, the selected checkpoint in eval mode, 4-class softmax stored as
  float16 in manifest order for the 1,350 training slices; cached-vs-online check on 20 training
  slices (< 1e-3); validation consistency with the teacher run's own record (< 0.005).
- **No quality gate.** The external teacher had to reach the decoder-only validation mean. The E2
  teacher is by design a D model, so a gate is not meaningful; the cache record stores
  `gate.status = NOT_APPLICABLE` with the teacher's validation Boundary F1 and DSC.
- **Students** (`kd_band_peer`; seeds 201–217; 17 runs): identical to D+KD-band (band definition,
  KD loss, λ_kd = 1, clamp) except that they read the E2 cache.
- **Teacher on the test patients.** The E2 teacher is evaluated once on patients 101–150 for
  descriptive context only (as the external teacher was), in the same evaluation job.

### E3 (optional, exploratory): standard augmentation

- **Arms:** `decoder_aug` and `kd_band_aug`, seeds 201–210 (10 runs per arm), D and D+KD-band with
  uniform D4 training augmentation: one of the 8 rotations-by-90°/flip combinations drawn
  uniformly per sample from the DataLoader worker's torch RNG, applied identically to the image,
  mask, stored boundary target and cached teacher probabilities. Validation and test are not
  augmented. Exact transforms only (no interpolation).
- **Start condition:** only after every E1 and E2 run is complete or declared failed and E1/E2
  test evaluation is done, and only if, at that time, the expected E3 training time (about 10 h)
  still leaves time to analyse and integrate it in the same work session. Otherwise E3 is not
  run, and the manuscript says it was registered and not run.
- **E3 is its own family.** Its primary comparison is `kd_band_aug − decoder_aug` (Boundary F1),
  with no multiplicity adjustment; `decoder_aug − D` and `kd_band_aug − D+KD-band` (10 vs 17 runs)
  are descriptive.

## 4. Queues, order and supervision

- `revision/queue_followup_e1e2.yaml`: the 17 E1 runs (seeds 201–217), then the E2 teacher, its
  cache step, and the 17 E2 students. `revision/queue_followup_e3.yaml`: the 20 E3 runs (seed by
  seed, `decoder_aug` then `kd_band_aug`). Both are written by `scripts/kd_followup/register.py
  --write` after this file is final.
- Each queue registers the sha256 of this file, every config, the split manifests and every
  source file (the parent study's `kd_sources()`, `scripts/acdc_confirm/*.py`,
  `scripts/kd_followup/*.py`). The frozen runner (`scripts/kd_mc/run_queue_mc.py`, via
  `scripts/kd_followup/run_queue.py`) refuses to launch or resume on any change.
- Supervision: `scripts/kd_followup/supervise.py` (a copy of `scripts/acdc_confirm/supervise.py`
  with its own log folder), same thermal and disk guards; stop marker `revision/STOP_KD_TRAINING`.
- **GPU sharing.** The supervisor refuses an occupied GPU. Training is launched only when no other
  compute process is on the GPU.

## 5. Evaluation (once per run)

- After a queue finishes and its runs pass the integrity check
  (`analysis/followup/integrity.py`: completion marker, 150 epochs, artifact hashes, config hash,
  provenance source hashes equal to the registered ones, no test output in the run folder), the
  test set is evaluated once per run with the frozen `scripts/revision/train_run.py::evaluate` and
  the frozen official-test loader of `scripts/acdc_confirm/evaluate_test.py` (NPZ hashes checked
  against the preprocessing record): `analysis/followup/evaluate_test.py`, writing only under
  `results/followup_eval/<run>/` with an evaluation record. It refuses to evaluate a run twice.
- Only the official patients 101–150 are evaluated (no internal test split).

## 6. Failed runs

- A run fails if it exits non-zero for a reason other than a guarded thermal/stop resume, or
  produces a non-finite validation selection metric, or non-finite training losses.
- The cause is investigated and logged. If the cause is operational (environment, disk, process)
  the run is retried **once** after the fix, resuming from its last checkpoint when the frozen
  resume checks allow it and otherwise from scratch in the same folder after the partial folder
  is preserved under `results/revision_runs/_failed_followup/`. A fix never changes the
  registered settings.
- If the retry fails, the run is declared failed in `revision/followup/failed_runs.json` and
  excluded; the queue continues; the analysis uses all completed runs and reports the exclusion.
  Runs are not rerun to obtain complete results.

## 7. Endpoints

- **Primary:** test Boundary F1 (2 px tolerance), per-slice class macro, averaged per patient,
  then per run (the Stage 3 definition: `analysis/revision/expansion_statistics.py::unit_values`).
- **Secondary:** DSC, IoU, HD95 and ASSD (pixels), same aggregation.
- **Per structure:** RV, myocardium, LV for DSC, Boundary F1, HD95
  (`analysis/kd3/unified_analysis.py::label_values`), descriptive.

## 8. Estimator and multiplicity

- **Estimator:** the Stage 3 run-level contrast `analysis/revision/expansion_statistics.py::contrast`
  (Δ = first − second arm mean of run means; SE² = between-run term + test-patient term; 95%
  interval Δ ± t_{0.975,ν}·SE with ν the Welch–Satterthwaite degrees of freedom from the
  between-run terms; two-sided p from the same t). HD95/ASSD signs reversed (positive = smaller
  distance).
- **Readings:** the SESOI labels of manuscript Table 3 (`expansion_statistics.classify`): SESOI
  0.01 (DSC, IoU, Boundary F1), 2 px (HD95), 0.5 px (ASSD); "relevant gain/harm", "gain/harm,
  size unclear", "no relevant effect", "undetermined".
- **Planned comparisons (one Holm family, primary endpoint):**

  | ID | Comparison | Question |
  |---|---|---|
  | C1 | D+KD-band − D+KD-full | Is the band restriction needed? |
  | C2 | D+KD-full − D | How large is the general KD effect from this teacher? |
  | C3 | D+KD-band(compact teacher) − D | Does a same-capacity, non-pretrained teacher give a gain? |
  | C4 | D+KD-band − D+KD-band(compact teacher) | What does the teacher's advantage contribute? |

  Holm adjustment over C1–C4 for test Boundary F1 (`expansion_statistics.holm`). Secondary
  metrics: the same four comparisons with Holm adjustment within each metric, reported as
  descriptive.
- **Context row (not in the family):** D+KD-band − D is the Stage 3 result and is reported
  unchanged from `revision/acdc_confirm/results/`.
- **Expected sensitivity (approximate, stated in advance).** With the Stage 3 SE (0.0039;
  between-run 0.0026, patient 0.0029; ν ≈ 31), a difference must be roughly 0.011 (unadjusted) to
  0.014 (smallest Holm threshold, α/4) to reach 80% power. A "no relevant effect" reading needs
  |Δ| below about 0.002. Differences between two KD variants smaller than about 0.01 are therefore
  expected to remain "undetermined". The mapping in §9 is written with this in mind.

## 9. Pre-specified interpretation mapping

### 9.1 Outcome category of each comparison (primary endpoint)

| Category | Rule |
|---|---|
| **P** (positive) | Holm-adjusted p < 0.05 and Δ > 0 |
| **N** (negative) | Holm-adjusted p < 0.05 and Δ < 0 |
| **E** (equivalent) | not P or N, and the 95% interval lies inside ±0.01 ("no relevant effect") |
| **U** (unresolved) | otherwise |

The Table 3 reading (size) is reported alongside, e.g. "P, size unclear".

### 9.2 Band restriction (E1: C1, C2)

| Case | Condition | Interpretation written in the manuscript |
|---|---|---|
| B1 | C1 = P | In this exploratory control, restricting KD to the boundary band gave a larger Boundary F1 gain than the same loss over the whole image; the band restriction is associated with part of the Stage 3 gain. |
| B2 | C1 = E | Restricting KD to the band was not needed: full-image KD from the same teacher gave Boundary F1 within ±0.01 of D+KD-band. |
| B3 | C1 = U and C2 = P | Full-image KD from the same teacher also raised Boundary F1 over D; whether the band restriction adds to that is unresolved, so the Stage 3 gain cannot be attributed to boundary-specific distillation. |
| B4 | C1 = N | Full-image KD gave a larger Boundary F1 gain than band KD; the band restriction is not the source of the gain and may limit it. |
| B5 | C1 = U and C2 ∈ {U, E, N} | The band question is unresolved. If C2 = E or N, it is added, descriptively, that full-image KD did not reproduce the Stage 3 gain, which is compatible with a role of the band but does not establish it. |

### 9.3 Teacher advantage (E2: C3, C4)

| Case | Condition | Interpretation written in the manuscript |
|---|---|---|
| T1 | C4 = P and C3 ∈ {U, E, N} | The gain is associated with the external teacher's advantage (pretraining and capacity together; not separable here): a teacher of the student's own architecture without pretraining did not reproduce it (C3 = E: gave no relevant gain). |
| T2 | C3 = P and C4 ∈ {U, E} | A same-capacity, non-pretrained teacher also raised Boundary F1 over D; an informational advantage of the teacher was not required for a gain. The external teacher's additional contribution is within ±0.01 (C4 = E) or unresolved (C4 = U). |
| T3 | C3 = P and C4 = P | Both: a same-capacity teacher gave a gain, and the external teacher gave a larger one. |
| T4 | C3 = N | Band KD from a same-capacity teacher lowered Boundary F1 relative to D; the C4 reading is reported with it. |
| T5 | C3 ∈ {U, E} and C4 ∈ {U, E, N}, or C4 = N with C3 ≠ P | The teacher question is unresolved (or, if C4 = N, the compact teacher gave more than the external one, which is reported as such). |

### 9.4 Manuscript narrative (combinations)

| Combination | Narrative |
|---|---|
| T1 or T3, with B1 | The manuscript is organised around: whether boundary information helps depends on whether the teacher has an informational advantage, and, given such a teacher, restricting distillation to the boundary added to full-image distillation in this setting. The error-location analysis (§10.1) is presented as a post hoc hypothesis about where errors occur. |
| T1 or T3, with B2, B3 or B4 | The benefit is associated with KD from a stronger teacher, not specifically with boundary-focused distillation (the author's example). The title question is answered as: in the studied setting, distillation helped when the teacher carried information the student lacked, and the boundary restriction was not shown to matter. |
| T1 or T3, with B5 | Teacher advantage associated with the gain; band specificity unresolved. |
| T2, with B1 | Boundary-band KD helped even without a teacher advantage, and the band restriction added to full-image KD: a boundary-specific effect not tied to teacher strength. |
| T2, with B2, B3 or B4 | Soft-label distillation helped irrespective of teacher advantage and of the band: a general KD (regularisation) effect, not a boundary or teacher effect. |
| T2, with B5 | KD from a peer teacher helped; band specificity unresolved. |
| T4 or T5 (any B) | The teacher question is unresolved; the manuscript reports the estimates and keeps the Stage 3 conclusion with the attribution left open, plus whatever the B case allows. |

Regardless of case: every follow-up statement is labelled exploratory and secondary; Stage 3
remains the only confirmatory result; teacher pretraining and teacher capacity are never
separated in the wording.

### 9.5 E3 (if run)

| Case | Interpretation |
|---|---|
| `kd_band_aug − decoder_aug` interval excludes zero, positive | The D+KD-band difference was also present under standard augmentation (exploratory). |
| interval includes zero | Whether the D+KD-band difference survives standard augmentation is not established. |
| interval excludes zero, negative | Under augmentation, D+KD-band scored below D (exploratory). |

## 10. Post hoc analyses (labelled post hoc wherever reported)

These are not part of the planned family and are written after the Stage 3 results were known.

1. **Error location versus KD gain.** For each D run on patients 101–150, the fraction of wrong
   pixels (prediction ≠ reference, foreground classes) that lie within 2 px of the reference
   boundary, by structure and by patient; related to the per-structure and per-patient
   differences of each KD arm against D (Spearman correlation across patients; per-structure
   table). Hypothesis-generating only.
2. **Reference analyses of the estimator.** A hierarchical (run-within-arm × patient) bootstrap
   and a crossed random-effects (method-of-moments / ANOVA) model for the Stage 3 primary contrast
   and C1–C4, reported in the supplement as references for the pre-specified estimator.
3. **Stage 1 versus Stage 3 decoder-only mean.** Decomposition of the difference between the D
   Boundary F1 mean in Stage 1 (seeds 1–3, internal test patients) and Stage 3 (seeds 201–217,
   official patients) using the Stage 3 runs' existing evaluations on the internal patients.

## 11. Outputs

- Analysis: `analysis/followup/analyze.py` → `revision/followup/results/{results.json,
  contrasts.csv, per_class.csv, per_seed.csv}`; figure `revision/figures/kd_followup.pdf`.
- Progress: `revision/followup_progress.md`; summary: `revision/followup_summary.md`.
