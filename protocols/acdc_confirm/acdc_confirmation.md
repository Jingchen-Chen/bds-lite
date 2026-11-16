# Pre-registration: held-out ACDC replication of D+KD-band vs decoder-only

**Status: FINAL DRAFT v3, 2026-11-07. Not yet committed; awaiting the author's approval.**
- The freeze is the commit that first contains this file, together with the implementation (§9), the held-out manifest and records, and the 34 registered configs. Its hash is then recorded in `revision/acdc_confirm/PROTOCOL_LOG.md`.
- After the freeze this file is never edited. It is hash-registered in every config and in the queue, so any edit blocks launch and resume.
- **Every deviation from this protocol** is logged in `revision/acdc_confirm/PROTOCOL_LOG.md`, with a UTC timestamp and the reason, together with the operational events listed in §5.
- The held-out integrity and preprocessing results (§2, §8) were filled in on 2026-11-07, before any model touched the held-out data.
- Repository housekeeping that precedes this study: commit `4f1dae8` (AGENTS.md anchor path).

Manuscript: PeerJ Computer Science CS-2026:06:142937, *When does boundary distillation help lightweight medical image segmentation?* (revision).

Parent study: `revision/KD_MULTICLASS_PROTOCOL.md`, frozen at `ed08782` (sha256 `699d5987…d7ea53`). Its results are in `8ec14fe`, `revision/kd_unified/`.

## 1. Selection rationale (stated honestly)

- **Exploratory results.** Three seeds per arm; test Boundary F1, D+KD-band − decoder-only:
  - ISIC2018: +0.038 (Holm p = 0.496).
  - ACDC: +0.024 [+0.004, +0.044] (unadjusted p = 0.029, Holm p = 0.088).
  - Synapse: +0.015 (Holm p = 0.500).
- **ACDC was chosen for this study after those results were seen.** Its evidence looked most consistent:
  - all three metrics favoured D+KD-band;
  - every D+KD-band run scored above every decoder-only run (lowest D+KD-band 0.8186, highest decoder-only 0.8095);
  - validation moved the same way (+0.027).
- **The choice depends on the results.** The +0.024 point estimate is therefore expected to overstate the true effect, so the study is powered for +0.015 (§6).
- **The validation contrast is not independent evidence**, because checkpoints were selected on validation DSC.
- **Purpose.** To test whether the existing ACDC finding holds on patients that played no part in it, with fresh training seeds. It is not a search for a configuration that works. Nothing in the method is changed.
- **The pooled cross-dataset estimate of the exploratory study is not used as evidence** anywhere in this study. Three datasets cannot support a between-dataset variance.

## 2. Confirmation set: the 50 official ACDC test patients (101–150)

**Why a new set is needed.** Every ACDC split used so far comes from the 100 official training patients (`recovered/followup/splits/acdc/{train,val,test}.json`, split seed 2026):
- 70 / 10 / 20 patients; 1,350 / 186 / 366 slices; no overlap between splits.
- Internal test patients: 011 013 017 018 020 026 032 036 045 047 052 056 065 071 076 077 085 089 092 098.
- Before this study, no file for patients 101–150 existed on any mounted filesystem, and no repository file referred to them (audit, 2026-11-07).

**Source.** The original NIfTI files from the official CREATIS distribution, which the ACDC organizers state remain public with ground truth. The author downloads them and places them at `/home/jc/bds-followup/data/raw/acdc_official_test/`. No third-party preprocessed copy is used.

**Integrity check** (`scripts/acdc_confirm/official_test.py verify`, writes `revision/acdc_confirm/official_test_integrity.json` and `official_test_volumes.csv`):
- exactly patients 101–150;
- `Info.cfg` with ED and ES frames;
- the ED and ES image and `_gt` files present, and no other `_gt` file;
- image and label shapes and affines equal; finite images;
- label values ⊆ {0, 1, 2, 3}, in the repository convention (0 background, 1 RV, 2 myocardium, 3 LV);
- a label-semantics plausibility check: the fraction of LV-border pixels that touch myocardium, compared with development patients;
- per-volume spacing, slice count and pathology group recorded;
- sha256 of every file.

**Integrity result: PASSED** (2026-11-07; `official_test_integrity.json`, sha256 `822fda91…3112`; no problems):
- **Patients and frames.** 50 patients (101–150), 100 ED/ES volumes (50 ED, 50 ES), 1,076 slices. Each patient has `Info.cfg`, the ED and ES images and their `_gt` files, and no other `_gt` file (a `_4d.nii.gz` cine file is present and unused).
- **Pathology groups.** 10 each of DCM, HCM, MINF, NOR and RV.
- **Labels.** The values are {0, 1, 2, 3} in all 100 volumes. Image and label shapes and affines are equal, and all images are finite.
- **Geometry.**
  - Slices per volume: 6–21 (median 10).
  - In-plane spacing 1.367–1.953, slice spacing 5–10 (header units).
  - 21 distinct in-plane shapes (most common 216 × 256).
  - 98 volumes are stored as float32 with integer values in 0–255, and 2 (patient 134) as int16. Stage 1 (clip and z-score) does not depend on intensity scale.
- **Label semantics, against the development patients.** All values below are medians, official / development.
  - Fraction of LV-border pixels touching myocardium: 1.000 (minimum 1.000) / 1.000.
  - Fraction of RV-border pixels touching LV: 0.000 / 0.000.
  - Fraction of RV-border pixels touching myocardium, ED: 0.314 / 0.314.
  - Fraction of RV-border pixels touching myocardium, ES: 0.336 / 0.338.
  - LV/myocardium volume ratio, ED: 1.46 / 1.35.
  - LV/myocardium volume ratio, ES: 0.72 / 0.60.

  The adjacency pattern agrees with the repository convention; swapped labels would break it.

**Preprocessing.** `official_test.py preprocess` refuses unless the integrity check passed and the raw files are unchanged since it ran. It reproduces the pipeline that produced `data/processed/acdc`, in two stages.

*Stage 1: NIfTI to pre-sliced arrays.* This step was done by a third party before this repository existed, and its code is not in the repository. It was reconstructed and checked against the 80 local development patients (the training and validation patients of the internal split). The steps, per volume:
1. clip above the 99th percentile (numpy, linear interpolation);
2. z-score with the clipped volume's mean and SD (ddof 0);
3. take slices in SimpleITK axis order, i.e. the transpose of nibabel's [x, y].

*Stage 2: frozen.* `src/bds_lite/data/converters.py::convert_acdc_transunet`: `_resize_gray` (PIL bilinear) and `_resize_mask` (PIL nearest) to 224 × 224. Output NPZ keys: `image`, `mask`, `sample_id` (`patientNNN_frameFF_slice_Z`). Every slice is kept.

*Reproduction gate.* Run inside `preprocess`, and recorded in `revision/acdc_confirm/preprocessing_dev_gate.json`. Against the 1,536 stored development slices:
- masks are **bit-exact** (0 mismatched pixels);
- slice counts match for all 160 volumes;
- images agree within **0.0148** z-units (tolerance 0.015);
- 589 of the 1,536 slices agree within 1e-5.

The residual is a per-volume difference in the effective clip threshold of the unknown third-party code (at most +0.4 raw intensity units). It is small, but it is not zero, and it is disclosed as a limitation (§13).

**Preprocessing result** (2026-11-07; `official_test_preprocessing.json`, sha256 `baf27f3b…c43e9`):
- The reproduction gate passed again (1,536 slices; masks exact; image difference 0.0148).
- Output: **1,076 slices** from 100 volumes of 50 patients, in `data/processed/acdc_official_test/test/`.
- Manifest `revision/acdc_confirm/official_test_manifest.json`:
  - file sha256 `add0459b8efc038da22c4acf5eabd64eee6ba203b0cd18847bc7ee92a003d609`;
  - sorted-ID hash `dc2c3f80…2e3a`;
  - count 1,076.
- The per-NPZ sha256 values are in the preprocessing record and are checked again by `evaluate_test.py`.
- The frozen `ManifestDataset` loads all 1,076 slices: labels {0, 1, 2, 3}, mean image value 0.000. No model was involved.

**Impact of the reconstruction, validation patients only** (`scripts/acdc_confirm/preprocessing_impact.py`; `revision/acdc_confirm/preprocessing_impact.json`):
- **Setup.** The 6 exploratory models were evaluated on the 10 internal validation patients (186 slices) twice: (a) on the stored data; (b) on the same slices rebuilt from native NIfTI with the reconstructed stage 1. No test data was used.
- **Variant (a) reproduces each model's recorded `per_case_val.csv` exactly.**
- **Rebuilt data.** Masks are identical; images differ by at most 0.0089.
- **Per-model run-level differences, (b) − (a):**

  | Model | Boundary F1 | DSC |
  |---|---|---|
  | D+KD-band seed 1 | +0.00008 | −0.00005 |
  | D+KD-band seed 2 | −0.00004 | +0.00004 |
  | D+KD-band seed 3 | −0.00011 | +0.00002 |
  | Decoder-only seed 1 | +0.00008 | +0.00004 |
  | Decoder-only seed 2 | +0.00007 | +0.00004 |
  | Decoder-only seed 3 | +0.00011 | +0.00003 |

- **Maximum absolute run-level difference:** Boundary F1 0.00011, DSC 0.00005.
- **Largest single slice-label difference:** Boundary F1 0.041, DSC 0.019.
- **Changed predicted pixels:** 51–91 of 9.33 million per model.
- **The D+KD-band − decoder-only validation contrast** shifts from +0.026871 to +0.026759 (Boundary F1, −0.00011) and from +0.020545 to +0.020511 (DSC, −0.00003).

**No model has been, or will be, evaluated on patients 101–150 before the pre-registration commit.** This includes the six exploratory models and the teacher.
- Before the freeze, these patients are only read by the integrity check, the preprocessing script and a data-loading check. None of them loads a model. The teacher is never evaluated on them.
- The only model runs before the freeze were:
  - the 50-step equivalence check (§9), on training slices;
  - the preprocessing impact check above, on the 10 internal validation patients.

  Neither touched a test patient of either set.

**Fallback.** This applies if the official test set cannot be obtained, or fails the integrity check or the reproduction gate, before the freeze. `PRIMARY_SET` in `scripts/acdc_confirm/common.py` is then set to `"internal_test"` before the freeze commit, and the study is a **seed replication only**:
- the primary endpoint is evaluated on the 20 internal test patients, which were already seen in the exploratory study;
- §11 then uses the fallback wording, which states that the test set has been seen.

## 3. Hypothesis and primary endpoint (exactly one)

- **H1.** On ACDC, D+KD-band yields higher test Boundary F1 than decoder-only.
- **Primary endpoint.** Test Boundary F1 on the **50 official test patients**, replication seeds **201–217** in each arm:
  1. take the label macro per slice from the unchanged evaluator (`recover_predictions.py::case_metrics` via `scripts/revision/train_run.py::evaluate`);
  2. average over the slices of each patient (the patient is the unit; 50 units);
  3. average over patients for each run.
- **Estimator.** The frozen, **unpaired** `analysis/revision/expansion_statistics.py::contrast` with `unit_values` (sha256 `17099233…6b120`), unchanged. It is the estimator of the exploratory analysis:
  - Δ = difference of arm means of run means;
  - SE = √(SE_run² + SE_unit²), where SE_run is the Welch SE of run means and SE_unit is the SE over patients of the per-patient arm difference;
  - Welch df; 95% t interval.
- **Test.** Two-sided, α = 0.05. Success also requires Δ > 0, i.e. the lower bound of the 95% interval must be above 0 (equivalent to one-sided α = 0.025).
  - This is the estimator's own interval, so the finding is judged exactly as it was reported.
  - There is one primary test, so no multiplicity adjustment applies.

## 4. Secondary analyses (pre-specified, descriptive only)

Each analysis below reports Δ, the 95% interval and the SESOI reading, using the same estimator unless stated otherwise:

- **(a)** The **6 exploratory models** (seeds 1–3 per arm, never retrained) on the 50 official test patients. Their checkpoints are hash-checked against §8.
- **(b)** The **replication seeds on the internal 20-patient split**. This is a seed replication only, and it is labelled as already-seen data.
- **(c)** A **seed-paired sensitivity** analysis on the primary set and on (b):
  - d_s = run-mean difference at seed s;
  - SE = √(SD(d)²/N + SE_unit²), t with N − 1 df;
  - the observed cross-arm seed correlation is reported.
- **(d)** **DSC, IoU, HD95 and ASSD** on the primary set (distance signs reversed, so positive favours D+KD-band).
- **(d)** **Per-structure** DSC, Boundary F1 and HD95 for RV, myocardium and LV on the primary set.
- **Also reported:**
  - the exploratory and replication seeds combined on the official set;
  - per-seed run means;
  - seed consistency (D+KD-band runs above the decoder-only mean; minimum and maximum separation);
  - validation contrasts;
  - training epoch-hours.

No secondary result can change the primary decision or be promoted to primary.

## 5. Seeds and N (fixed in advance)

- **N = 17 per arm: seeds 201, 202, …, 217.** 34 runs.
- **Excluded seeds:**
  - 1–3, the exploratory runs;
  - 4–10, reserved by the stopped Experiment C (its decoder-only seeds 4 and 5 exist on disk; their metrics are unread and unused);
  - 100, the teacher.
- **Pairing.** For each seed s, both arms start from the same initial weights and the same batch order (§9). Trajectories diverge after the first update, because the losses differ.
- **Run order.** Seed by seed, decoder-only then D+KD-band (201-decoder, 201-KD, 202-decoder, …), so that thermal and time drift are balanced across arms.
- **Failed-run policy** (fixed in advance; added at the author's request before the freeze).
  1. **Interrupted runs resume from their last checkpoint.** This covers a thermal stop, a crash, an engineering failure such as out-of-memory or a driver error, and power loss.
     - Resume uses the frozen path: `last.pt` is saved at epoch 1, every 10 epochs, at epoch 150 and on any stop.
     - Every resume refuses a change to config, code or data.
     - Thermal stops resume automatically under the supervisor.
     - After any other halt, the supervisor is relaunched with the same command (`PLAN.md` Phase 18), and the relaunch is logged.
  2. **No run is excluded and no seed is replaced.** A run is declared failed only in two cases:
     - (i) it **diverges**: the frozen trainer raises `FloatingPointError` (a non-finite loss, gradient or validation metric);
     - (ii) it **cannot complete**: three consecutive resumes from its last checkpoint end in a non-thermal failure.
  3. **Declaring a run failed.** The declaration is made only with `scripts/acdc_confirm/declare_failed.py`.
     - The script checks the evidence: the `FloatingPointError` text, or at least three failure tracebacks, in the run's `queue_stdout.log`.
     - It records the run, the kind of failure, the reason and a UTC timestamp in `revision/acdc_confirm/failed_runs.json`, and in `PROTOCOL_LOG.md`.
     - The queue then skips that run, keeps its folder unchanged, and continues with the next one.
  4. **Reporting.** A failed run is reported as failed, with its last epoch and its reason. The analysis proceeds with the runs that completed and passed integrity:
     - the unpaired estimator allows unequal N;
     - the seed-paired sensitivity analysis uses the seeds that completed in both arms;
     - the **achieved power** at the realised (n_D+KD-band, n_decoder) is reported (§7).
     - At least 2 completed runs per arm are required. Otherwise the primary analysis cannot be run, and this is reported.
  5. **Integrity failures.** A completed run that fails an integrity check (§7) is a reported blocker, not a silent exclusion. The test evaluation does not run until the author decides.

## 6. Sample-size justification

**Inputs.** From ACDC test Boundary F1 of the 3 + 3 exploratory runs (20 internal patients). The frozen contrast reproduces the recorded value exactly (Δ 0.023873, p 0.0293).

| Quantity | Value |
|---|---|
| Between-seed SD of run means, D+KD-band / decoder-only | 0.00648 / 0.00754 |
| Pooled SD (4 df) | 0.00703 |
| χ² upper confidence bound on the SD, 80% / 90% | 0.01095 / 0.01363 |
| Seed × patient residual variance σ_e² (76 df) | 6.78e-4 (80% bound 7.87e-4) |
| Per-patient difference variance | 3.51e-4 |

**Method.** Monte Carlo simulation of the frozen estimator with N runs per arm and **50 patient units**: 100,000–200,000 replicates per cell, generator seed 20261107. Success = two-sided p < 0.05 and Δ > 0. Assumptions on the run-mean SD:
- **A:** the bound as observed on 20 patients, not reduced for 50 patients (conservative);
- **B:** the seed component bounded, and the patient-averaging component (σ_e²/20) rescaled to 50 patients (SD 0.01008 / 0.01295 at the 80% / 90% bound).

Assumptions on SE_unit:
- **fixed:** the per-patient difference variance divided by 50, i.e. 0.00265, not shrinking with N;
- **shrink:** generated from σ_e² at its 80% bound, with no patient × arm interaction.

**Power at Δ = +0.015, 90% SD bound, 50 patients:**

| Assumption | N = 16 | N = 17 | N = 18 | Smallest N for 80% |
|---|---|---|---|---|
| A, fixed | 0.780 | **0.803** | 0.824 | 17 |
| A, shrink | 0.832 | 0.857 | 0.877 | 15 |
| B, fixed | 0.816 | 0.839 | 0.857 | 16 |
| B, shrink | 0.868 | 0.889 | 0.907 | 14 |

**Other cells, smallest N for 80% power (50 patients):**
- Δ = +0.015 at the 80% bound: 10–12. At N = 17, power ≥ 0.92.
- Δ = +0.024 at the 90% bound: 7. At N = 17, power ≥ 0.995.

**For comparison, 20 internal patients** (the draft v1 basis):
- Δ = +0.015 at the 90% bound needed > 20 (fixed) or 17 (shrink).
- At N = 17, the fallback's power at +0.015 is 0.82 (80% bound) or 0.68 (90% bound), and 0.99 at +0.024 (90% bound).

**Choice: N = 17.** It is the smallest N that gives ≥ 80% power at +0.015 under the 90% upper bound on between-seed variance in every assumption above. N = 16 falls short (0.780) under the most conservative one. The number of seeds needed does not fall below 17 when moving from 20 to 50 patients, because the run-level variance dominates.

**Cost.** Mean observed time of the six exploratory runs on the RTX 5060, including thermal pauses (telemetry spans 26.3–28.2 min): 27.4 min per run. 34 runs ≈ **15.5 GPU-hours**. The one-time evaluation adds a few minutes. The teacher is not retrained; its cache is reused.

## 7. Analysis (run exactly once, after all 34 runs finish)

1. **Integrity** (`scripts/acdc_confirm/integrity.py`), per run (`analysis/kd3/integrity.py::check_run`, unchanged):
   - 150 contiguous epochs; validation-only final marker; artifact hashes;
   - config and source hashes equal to the registration;
   - selected epoch = first maximum of validation DSC;
   - finite losses; no test output.

   In addition:
   - every config passes `common.validate`;
   - teacher-cache hash in every provenance;
   - seeds 201–217 complete in both arms;
   - L_kd logged in every D+KD-band run and in no decoder-only run;
   - the equivalence check (§9) passed.

   Feasibility: L_kd decreased in every D+KD-band run. It is reported and never used to exclude a run.
2. **One-time test evaluation** (`scripts/acdc_confirm/evaluate_test.py`):
   - refuses if integrity fails or if any evaluation output exists;
   - unchanged `train_run.py::evaluate`, frozen `ManifestDataset` loader, batch 16, 4 workers, no shuffling;
   - outputs only under `results/acdc_confirm_eval/`; the parent-study run folders are not touched;
   - official set: the 34 replication models and the 6 exploratory models, with every NPZ hash checked against the preprocessing record;
   - internal split: the 34 replication models.
3. **Post-test integrity** (`integrity.py --post-test`).
4. **Analysis** (`scripts/acdc_confirm/analyze.py`): §3 and §4, plus the **achieved power**. Achieved power is the §6 simulation at the realised numbers of completed runs per arm, for Δ = +0.015 and +0.024, at the 80% and 90% SD bounds, under the most conservative assumption (A, fixed). It equals the planned power if no run failed. Outputs in `revision/acdc_confirm/results/`.
5. **Readings** per metric:
   - interval wholly above +SESOI (0.01 for DSC/IoU/Boundary F1, 2 px for HD95, 0.5 px for ASSD): relevant gain;
   - interval excluding 0 only: gain or harm of unclear size;
   - interval inside ±SESOI: no relevant effect;
   - otherwise: not established.
6. **Deviation checklist** against this file, item by item, in `revision/acdc_confirm/DEVIATION_CHECKLIST.md`.

## 8. Frozen items (none may change)

| Item | Frozen value |
|---|---|
| Training split | `recovered/followup/splits/acdc/{train,val,test}.json`, file sha256 `591a6cd0…afe9d` / `4bf60eb7…c6dc90` / `546493cb…6b4638`. Processed data `data/processed/acdc` (224 × 224, 1 channel, `normalize: none_stored`). |
| Held-out set | Patients 101–150; 1,076 slices in `data/processed/acdc_official_test/test/`; manifest `revision/acdc_confirm/official_test_manifest.json` (file sha256 `add0459b…d609`); per-NPZ sha256 values in `official_test_preprocessing.json` (`baf27f3b…c43e9`); raw-file sha256 values in `official_test_integrity.json` (`822fda91…3112`) |
| Data augmentation | None (no online augmentation; enforced by `require_protocol`) |
| Student model | `bdslite_unet`, base 32 channels, 1-class boundary decoder `D_b`, gate present with scale 0 (inert), 2,197,493 parameters |
| Segmentation and boundary loss | CE 1.0 + Dice 1.0, no class weights; `λ_b` = 0.05 (BCE + Dice, union two-pixel boundary target); distill, surface and distance weights 0 |
| Optimiser and schedule | AdamW, lr 1e-4, weight decay 1e-5, batch 16, **150 epochs**, cosine (T_max 150), AMP with a fresh GradScaler per epoch, gradient clip 1.0, 4 workers, deterministic cuDNN |
| Checkpoint selection | First maximum of validation DSC (strict `>` against a running best that starts at −1.0, so ties keep the earlier epoch); the final model is that epoch's state |
| Teacher | smp 0.6.0 `Unet("resnet34", imagenet)`, seed 100, `results/revision_runs/acdc_kd_teacher_seed100/best.pt`, sha256 `735747303c444e13dd57457e3540ee410b5de43421b68fd92e2eda69c88dafa0`, selected epoch 13. **Not retrained, not modified, not evaluated.** |
| Teacher cache | `results/revision_runs/acdc_kd_teacher_cache/train_teacher_prob_fp16.npy`, sha256 `42266ea089e495d651cf654ae7c5c287a2b40d77cc1131ea78f3a0d059a6077e` (1350 × 4 × 224 × 224, float16) |
| KD weight | `λ_kd` = 1.0 (D+KD-band); 0.0 (decoder-only) |
| KD target and loss | Teacher softmax, soft-target cross-entropy, both distributions clamped to [1e-6, 1 − 1e-6], fp32 |
| Temperature | **None.** The cached softmax is used as is (equivalent to T = 1); the implementation has no temperature setting, and none is added. |
| Band width | Union over foreground classes of the dilation-minus-erosion of the teacher argmax masks; 3 × 3 square element, one iteration, zero border (a band about 2 px wide) |
| Band normalisation | `Σ B·CE / (Σ B + 1e-6)` over the batch; an empty band contributes 0 |
| Environment | torch 2.11.0+cu130, CUDA 13.0, smp 0.6.0. `.venv` is an offline overlay of the frozen `/home/jc/bds-followup/.venv` and `.venv-kd` site-packages, plus nibabel 5.4.2 for the data step. |
| Frozen code | Every file in `scripts/kd_mc/`, `scripts/kd/` and `scripts/revision/`, and `src/bds_lite/`, `analysis/revision/expansion_statistics.py`, `recover_predictions.py`, `family_statistics.py` and `analysis/kd3/`, byte-identical to `ed08782`/`8ec14fe` |
| Exploratory models | D+KD-band seeds 1–3, `best.pt` sha256 `d3d704e4…b038` / `4b2fe9a1…364a` / `2c376074…3920`. Decoder-only seeds 1–3, `96d21b73…e56b` / `5d66a6fa…caa2` / `ba888f9f…7faf2`. Full hashes in `scripts/acdc_confirm/common.py`. |

## 9. Implementation and the pre-freeze equivalence check

**Why new code is needed.** The frozen trainers cannot run this design unchanged:
- `scripts/kd_mc/common.py` admits student seeds 1–3 and `λ_kd` = 1.0 only.
- The exploratory decoder-only trainer, `scripts/revision/train_run.py`, stops after epoch 1 while Experiment C's `revision/STOP_TRAINING` exists (that marker stays in place), and it scores the test set automatically at the end of every run.

No existing file is edited.

**New files**, frozen in the freeze commit:
- **`scripts/acdc_confirm/launch.py`**
  - Checks that the config equals the generator output (`common.validate`).
  - Then, for its own process only, rebinds `scripts.kd_mc.common.STUDENT_SEEDS` to this run's seed and `KD_SPEC["lambda_kd"]` to this run's `λ_kd`.
  - Then calls the frozen `scripts/kd_mc/train_mc.py::main()`.
  - Consequences:
    - both arms run through one trainer, which builds no test dataset;
    - the arms differ only in `λ_kd`; at 0, `KDCriterionMC` skips L_kd entirely;
    - the stop marker is `revision/STOP_KD_TRAINING`.
- **`common.py`, `register.py`:** 34 configs, generated from the frozen student config. Only `project.seed`, `revision.*` and (decoder-only) `kd.lambda_kd` differ.
- **The queue `revision/queue_acdc_confirm.yaml`** registers the sha256 of this file, every config, the manifests and every source file. It refuses to launch or resume on any change.
- **`run_queue.py`:** the frozen `run_queue_mc.main()` with three module bindings: entry point, protocol path, and the config check.
- **`supervise.py`:** a copy of `supervise_mc.py` that differs only in its queue script, log folder and start-event name. A unit test restores the original byte for byte.
- **`official_test.py`, `preprocessing_impact.py`, `declare_failed.py`, `integrity.py`, `evaluate_test.py`, `analyze.py`.** `run_queue.py` also skips runs declared failed under §5.
- **`tests/revision/test_acdc_confirm.py`** (15 tests):
  - fresh, alternating seeds;
  - configs differ from the frozen student only in the permitted keys;
  - the validator rejects any other change;
  - same seed gives identical initial weights in both arms;
  - the decoder-arm criterion equals the decoder-only loss bitwise;
  - the launcher changes no file and only the two constants;
  - the supervisor copy is exact;
  - stage-1 preprocessing; decision rules; the paired estimator;
  - the estimator reproduces the exploratory ACDC contrast (Δ 0.023873198, p 0.029263290);
  - the failure-evidence rules, the queue skip of declared-failed runs only, and achieved power equal to the planned power at N = 17.

**Equivalence check, run 2026-11-07** (`scripts/acdc_confirm/equivalence_check.py`, sha256 `c636a3ec…94bc`; report `revision/acdc_confirm/equivalence_check.json`):
- **Setup.**
  - Seed 1, 50 optimizer steps.
  - Path 1: `train_run.py` with the registered decoder-only config (`configs/revision/acdc_m2_aux_boundary_only_unweighted_seed1.yaml`).
  - Path 2: the launcher with `λ_kd` = 0 (`configs/revision/acdc_confirm/equivalence/acdc_equivcheck_decoder_lambda0_seed1.yaml`).
  - Path 3: `train_run.py` repeated.
  - Each path ran its unmodified `main()` in a fresh process. The only intervention was a wrapper around the module's `epoch_train`: the first epoch yielded 50 batches, then the state was dumped and the process exited.
  - Validation, checkpoint selection, the `STOP_TRAINING` check and the automatic test scoring all come after `epoch_train` returns, so none was reached.
  - No evaluation output was produced. As always, `train_run.py` builds its test dataset object and hashes the test files at start-up, but no model sees them.
- **Result, `train_run.py` vs launcher at `λ_kd` = 0:**
  - all 140 parameter and BatchNorm-buffer tensors **bitwise identical** at initialisation and after 50 steps (max abs diff 0.0);
  - AdamW state identical (max abs diff 0.0); scheduler state identical;
  - batch order identical; 0 AMP overflows in both.
  - The launcher's epoch-0 KD gradient probe is observational: it restores the buffers and RNG, which the bitwise identity confirms. Later probes (epochs 1, 25, …) use the same restoring code.
  - The only difference: the logged epoch-mean CE scalar differs by ≤ 1.5e-8. The repeat of `train_run.py` shows the same order of difference, so this is reduction-order jitter in a logged scalar, not in training.
  - The first run (before a cosmetic edit to the script) gave the same result: `equivalence_check_v0_superseded.json`.
- **Checkpoint selection.** Identical code in both paths:
  - `if val["val_dsc"] > best: best, best_epoch, best_state = val["val_dsc"], epoch, cpu_state(model)`, with `best = -1.0`;
  - the same `bds_lite.training.runner.validate_one_epoch`;
  - `model.load_state_dict(best_state)` after epoch 150;
  - the same resume state.
  - Only `train_run.py` additionally scores the test set at the end. The launcher writes a validation-only final marker (`test_evaluated: false`).
- **Conclusion.** The two paths are equivalent for training and selection. No difference requires an author decision.

## 10. Execution and monitoring

The exact commands are in `PLAN.md`, Phase 18.

- **Launch.** One queue under `scripts/acdc_confirm/supervise.py`. Run order as in §5. The launch, thermal-stop, cooling and resume rules are the frozen ones (`KD_MULTICLASS_PROTOCOL.md` §5).
- **May be monitored during training:**
  - training losses (L_seg, L_boundary, L_kd);
  - validation loss and validation DSC (`progress.json`, `history.json`);
  - GPU telemetry; queue and supervisor logs.
- **Validation sanity.** Finite values, and validation DSC changing over epochs. Used only to detect a broken run, never to select, stop or tune.
- **No test data during training.** No test prediction, metric or file, internal or official, is produced or opened before all 34 runs finish and pass integrity.

## 11. Decision rules (primary endpoint; seeds 201–217)

| Outcome | Criterion | Wording in the paper (held-out version) |
|---|---|---|
| **Success** | 95% lower bound > 0 (Δ > 0, two-sided p < 0.05) | "In a pre-registered replication with 17 fresh seeds per arm, evaluated on the 50 official ACDC test patients that played no part in the exploratory study, D+KD-band raised test Boundary F1 relative to decoder-only by Δ [95% CI]. ACDC was chosen after the exploratory three-dataset results; the result supports the ACDC gain on held-out patients of this dataset in this setting. It does not show transfer to other datasets, and without a D+KD-full control it does not show that the gain is specific to boundaries." Add "a gain above the smallest effect of interest" if the lower bound > +0.01, otherwise "of unclear size". |
| **Failure** | 95% upper bound < +0.01 (no relevant gain; includes harm) | "The pre-registered held-out replication did not reproduce the exploratory ACDC gain: with 17 fresh seeds per arm on the 50 official test patients, Δ = … [95% CI], which excludes a Boundary F1 gain of 0.01 or more in this setting. The exploratory ACDC result is not supported." |
| **Inconclusive** | Neither | "The pre-registered held-out replication was inconclusive: Δ = … [95% CI] neither excludes zero nor excludes a gain of 0.01. The ACDC gain remains not established." |

**Fallback wording** (if `PRIMARY_SET = "internal_test"`). Replace "evaluated on the 50 official ACDC test patients that played no part in the exploratory study" with "on the same 20 internal test patients used in the exploratory study (already seen)". Replace "supports the ACDC gain on held-out patients" with "shows that the ACDC gain reproduces across training seeds on this split; it does not show transfer to new patients".

- **All outcomes are reported**, together with secondary (a)–(d) and the ISIC2018 and Synapse exploratory results, whatever the direction.
- Secondary (a) and (b) can be described only as descriptive context. For example, a gain of the exploratory models on the official set does not count as a confirmation.
- Integrating the result into the manuscript is a separate author decision. The wording above is fixed in advance.

## 12. Prohibited actions

1. Evaluating any model (including the exploratory six and the teacher) on patients 101–150 before the freeze commit.
2. Inspecting any test prediction or metric (official or internal) of a replication run before all 34 runs finish and pass integrity, or running the test evaluation more than once.
3. Adding, removing or replacing seeds in response to any result or p-value. N = 17 is fixed; there is no interim analysis and no optional stopping.
4. Changing the primary endpoint, set, estimator, alpha, sidedness or decision thresholds, or promoting any secondary analysis to primary.
5. Changing any frozen item in §8: hyperparameters, KD weight, band rule, epochs, optimiser, schedule, selection rule, data, split, preprocessing or augmentation.
6. Retraining, replacing, modifying or evaluating the teacher or its cache.
7. Excluding runs on validation or test performance, or declaring a run failed other than by the §5 criteria and `declare_failed.py`.
8. Running KD-full, or extending ISIC2018 or Synapse, under this pre-registration.
9. Reading or using Experiment C runs (ACDC seeds 4–10).
10. Using the exploratory pooled cross-dataset interval as inferential evidence.

## 13. Limitations stated in advance

- **ACDC was selected after the results were seen.** The held-out set removes test-set reuse, but not that selection.
- **Stage-1 preprocessing of the held-out set is a validated reconstruction**, not the original third-party code.
  - On the development data, masks are reproduced exactly and images to within 0.015 z-units (the gate accepted by the author).
  - On the 10 validation patients, the reconstruction changed the 6 exploratory models' run-level Boundary F1 by at most 0.00011 and DSC by at most 0.00005, and the arm contrast by −0.00011 (Boundary F1). This is about 1% of the SESOI (0.01), and two orders of magnitude below the effects under test.
  - The difference is not zero, and it is unverified for the official patients, whose original third-party preprocessing does not exist.
- **The official test set comes from the same ACDC centre and protocol as the training data.** Agreement on it is evidence about held-out patients of this dataset, not about other scanners, centres or datasets.
- **The label check on the official set covers values and LV–myocardium adjacency only.** It does not validate the annotations themselves.
- **Slice-level 2D evaluation** averaged per patient; no volume-level metrics.
- **One teacher, trained once.** Teacher capacity, pretraining and independence vary together. There is no D+KD-full control, so boundary specificity cannot be assessed.
- **Trainer mix in the combined estimates.** The fresh decoder-only arm is trained via the launcher at `λ_kd` = 0, and the exploratory decoder-only runs via `train_run.py`. The two paths are bitwise identical over the first 50 steps, but not shown identical over 150 epochs. This affects only the combined (descriptive) estimates.
