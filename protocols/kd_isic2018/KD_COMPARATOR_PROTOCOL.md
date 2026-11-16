# Protocol: external-teacher boundary-band distillation comparator (ISIC2018)

**Status: FROZEN 2026-11-07, before any teacher or student training.** The freeze is the local git commit that first contains this file version together with the implementation in §13; it is the time stamp, as for `revision/EXPANSION_PROTOCOL.md`. Its hash cannot be written into the file it freezes, so it is recorded in `revision/kd_comparator/DECISIONS_AND_DEVIATIONS.md`.
- The sha256 of this file is registered in every KD config and queue file. The queue runner refuses to launch or resume a run if this file changes.
- Therefore, after the freeze, this file is never edited. Dated decisions (such as the Phase 2 decision) and deviations go in `revision/kd_comparator/DECISIONS_AND_DEVIATIONS.md`.

**History.**
- 2026-11-07 code audit against `HEAD` `e2e7bdc` (Appendix B).
- 2026-11-07 author decisions on OPEN-1, -2, -5 and -6 (§12).
- Implementation, engineering checks and freeze (§§9, 13).

No test metric of any new arm existed at the freeze. Only results already analysed in the manuscript were read (seeds 1–10 of the four ten-seed arms; Experiment C runs, seed ≥ 11, were not read).

Manuscript: PeerJ Computer Science CS-2026:06:142937, *When does boundary distillation help lightweight medical image segmentation?* (revision). The title is kept.

---

## 1. Purpose

The revised manuscript found no condition in which boundary distillation added to training-only boundary supervision. In particular, full BDS-Lite scored below the decoder-only arm on ISIC2018 (confirmed), and the transfer path (sibling-decoder feature distillation and gate) gave no established gain over the decoder.

This experiment tests one further, literature-motivated condition: **does band-restricted probability distillation from an external, independently trained, frozen teacher add to the same auxiliary boundary supervision?**

The design is a **BDKD-inspired comparator**, not a reproduction of BDKD-Net (§4.3). The experiment is not designed on the assumption that the result will be positive. All outcomes, including null and negative ones, are reported (§10).

## 2. Questions

- **Q1 (primary).** Decoder + KD-band vs decoder-only: does external-teacher boundary-band probability distillation provide additional Boundary F1 benefit on top of identical auxiliary boundary supervision?
- **Q2 (secondary, conditional on Q1 showing a gain).** Decoder + KD-band vs decoder + KD-full: is the gain related to the boundary-band restriction rather than to distillation over the whole image?
- **Background (descriptive only).** The new arm(s) against BDS-Lite and U-Net, using the existing runs.

## 3. Existing data and status of the test set

- Existing ten-seed ISIC2018 arms are reused and not retrained. Seeds 1–10 only; Experiment C runs (seed ≥ 11, `revision/queue_confirmation.yaml`) are excluded, and `analysis/revision/make_audit_assets.py::run()` refuses them.

  | Arm | Seeds 1–3 | Seeds 4–10 |
  |---|---|---|
  | Decoder-only | `results/revision_runs/isic2018_m2_aux_boundary_only_seed{1..3}/` | `results/revision_runs/isic2018_m2_aux_boundary_only_seed{4..10}/` |
  | U-Net | validation: `recovered/followup/results/followup_20261018_v2/isic2018_U_seed{1..3}/per_case.csv`; test: `revision/official_isic/test/U_seed{1..3}/per_case.csv` | `results/revision_runs/isic2018_m0_unet_seed{4..10}/` |
  | BDS-Lite | validation: `recovered/followup/results/followup_20261018_v2/isic2018_F_seed{1..3}/per_case.csv`; test: `revision/official_isic/test/F_seed{1..3}/per_case.csv` | `results/revision_runs/isic2018_m6_full_seed{4..10}/` |
  | U-Net+GSL | `results/revision_runs/isic2018_gsl_seed{1..3}/` | `results/revision_runs/isic2018_gsl_seed{4..10}/` |

  Paths are resolved by `analysis/revision/expansion_statistics.py::isic_path` (test) and `analysis/revision/make_audit_assets.py::validation_path` (validation). The KD analysis scripts use these same functions.
- **Partition and data.** Training: 2,075 images, `recovered/followup/splits/isic2018/train.json` (sha256 `393527ba797be787…`). Validation: 519 images, `recovered/followup/splits/isic2018/val.json` (sha256 `81329e5e8a2ac3bb…`). Split seed 2026. Arrays in `data/processed/isic2018/{train,val}/*.npz`: image 3×224×224 in [0, 1], mask 224×224, stored two-pixel `boundary` target.
- **Official test set** (1,000 images; archives `data/revision/isic2018_official/archives/ISIC2018_Task1-2_Test_Input.zip`, sha256 `e59ae1f69f4ed16f…`, and `ISIC2018_Task1_Test_GroundTruth.zip`, sha256 `2e8f6edce454a5bd…`; loader `analysis/revision/recover_predictions.py::Cohort`, split `official_test`). It has already been analysed in the manuscript for the existing arms. The choice of this comparator was motivated by the literature and by those earlier results, so the new experiment is **not** a validation on a test set never seen by the investigators. What it does guarantee is narrower:
  - the new arms' designs and hyperparameters are fixed before any of their test-set metrics exist;
  - no new-arm test metric is computed before §9 step 8. The training entry point builds no test dataset, and the queue refuses any run folder that contains a test output.
- The validation set (519 images) is the one used for checkpoint selection in all existing arms and in the new runs.

## 4. Arms

### 4.1 Common student recipe (identical to the existing ISIC2018 decoder-only arm)

The student is the decoder-only configuration of the manuscript, exactly as used for the existing ten decoder-only runs. Each student config equals `configs/revision/isic2018_m2_aux_boundary_only_seed4.yaml` in its `data`, `dataset`, `model`, `loss` and `trainer` blocks and its split manifests. Only `project.seed`, `revision.{purpose, protocol, protocol_sha256, arm}` and the added `kd` block differ. `scripts/kd/common.py::require_protocol` enforces this at every launch.

- **Model.** `bdslite_unet`, `base_channels: 32`, `boundary_classes: 1`, training-only boundary decoder `D_b`. The config has `boundary_gate: true` with `boundary_gate_scale: 0.0`, so the projection `P` and the gate are instantiated but inert. The gate is an exact identity, and `P` receives no gradient from any active loss (1,072 inert parameters). Functionally, the deployed network is a plain U-Net.
- **Losses.** `L_seg = L_ce + L_dice` (weights 1.0; CE averaged over all pixels of the batch; soft foreground Dice with sums over the whole batch, ε = 1e-6). `L_boundary = BCE + Dice` on the `D_b` logits, `λ_b = 0.05`. Its target is the stored morphological gradient `dilation XOR erosion` of the foreground mask (one iteration of SciPy's 3×3 cross element, a two-pixel band). No distillation of `q_b`, no surface loss, no class weights.
- **Optimizer and schedule.** AdamW, learning rate 3e-4, weight decay 1e-5, batch size 8, `CosineAnnealingLR(T_max=150)` stepped per epoch, 150 epochs. AMP fp16 autocast with a new `GradScaler` per epoch. Gradient-norm clipping at 1.0.
- **Data loading.** No online augmentation. Training shuffle with `torch.Generator().manual_seed(seed)`, 4 workers, `pin_memory`, `torch.set_num_threads(4)`.
- **Checkpoint rule.** The first maximum (strict `>`) of the per-image mean foreground DSC over the 519 validation images, evaluated every epoch.
- **Evaluator.** `scripts/revision/train_run.py::evaluate` → `analysis/revision/recover_predictions.py::case_metrics` (Boundary F1 with a 2-pixel tolerance), unchanged.
- **Exact source of the existing arm.**
  - Seeds 1–3: `configs/revision/isic2018_m2_aux_boundary_only_seed{1,2,3}.yaml`, commit `771eb49`.
  - Seeds 4–10: `configs/revision/isic2018_m2_aux_boundary_only_seed{4..10}.yaml`, commit `3f1c62b`.
  - All ten runs record identical source and data hashes; key files: `train_run.py` `10c0f722beb4bb4d…`, `losses.py` `67aaa72ac6beb8fe…`, `bds_lite.py` `38f29fb103238c7e…`.
- **New configs.** `configs/revision/kd/isic2018_kd_band_seed{1..10}.yaml` and `configs/revision/kd/isic2018_kd_full_seed{1..10}.yaml`. Their sha256 values are in the queue files (§13).

At deployment, `D_b` is removed, and the teacher is never part of inference. The deployed graph equals that of the deployed decoder-only model, so parameters and FLOPs are equal.

### 4.2 New arms

Both new arms add one term to the common recipe:

`L = L_seg + λ_b · L_boundary + λ_kd · L_kd`, with `λ_kd = 1.0`.

Notation (computed in fp32 with autocast disabled; `scripts/kd/kd_loss.py`):
- `p_s = softmax(z_s)[foreground]`, the student's foreground probability;
- `p_t` = the cached teacher foreground probability (float16 cache, cast to fp32);
- both clamped to [1e-6, 1 − 1e-6] inside the BCE.

**D+KD-band (primary new arm)**

- `M_t = 1[p_t > 0.5]` (strictly greater).
- `B = dilate(M_t) − erode(M_t)` with a 3×3 square structuring element, applied once, and a zero-valued image border (§12, OPEN-5).
- `L_kd = Σ_batch B · BCE(p_s, p_t) / (Σ_batch B + ε)`, ε = 1e-6. Both sums run over all pixels of all images in the batch.

**D+KD-full (control, Phase 2b only)**

- Identical to D+KD-band with `B ≡ 1`.

The two arms differ only in the distillation region. A consequence of the definition is that in D+KD-band each band pixel receives a larger share of the normalised loss than in D+KD-full. This is a property of restricting the region, not an extra design choice.

`L_kd` is added only to training batches, which carry `p_t`. Validation batches have no cache, so the term is skipped there. This affects only the logged validation loss; selection uses DSC.

### 4.3 Relation to BDKD-Net (Xia, Li & Sun, 2026, *J. Imaging* 12(7):306)

**Taken from BDKD-Net, as recorded in the author's earlier reading of the paper:**
- teacher hard mask from the teacher probability at threshold θ = 0.5;
- band = dilation minus erosion of that mask with width w = 3;
- probability-space BCE between student and teacher probabilities, restricted to the band and normalised by the band pixel count plus ε;
- weight 1.0 for this term, from the BDKD-Net composite loss `L_seg + 0.5·L_resp + 1.0·L_bd + 0.25·L_detail`.

**Not taken, so this is not a reproduction:**
- the response term `L_resp` and the detail-feature term `L_detail`;
- the PVTv2-B2 + EGFR teacher and PVTv2-B0 student;
- 352×352 input, data augmentation, the 30-epoch schedule with early stopping;
- the polyp data.

In BDKD-Net the student has no ground-truth boundary head. Here, the distillation is **added to** an auxiliary boundary decoder.

**Implementation choices made here, not BDKD-Net specifications** (§12, OPEN-5):
- reading w = 3 as the side of a 3×3 square element applied once;
- the zero-valued border;
- batch-level normalisation;
- ε = 1e-6 and the clamp;
- the behaviour of empty bands;
- float16 caching of `p_t`.

[CLAIM-CHECK] The paper's own definition of w (kernel side vs radius) was not re-read for this freeze: the publisher's site returned HTTP 403 to automated access on 2026-11-07, and no local copy exists. The manuscript must describe the band as this protocol defines it, not as "the BDKD-Net band", unless the paper is checked.

## 5. Teacher

- **Architecture.** `segmentation_models_pytorch` 0.6.0 `Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=2)` (24,436,514 parameters), wrapped by `scripts/kd/teacher.py::TeacherNet`.
  - ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]` are applied inside the wrapper. Stored images are in [0, 1], the input range declared by the weights. Student inputs are unchanged.
  - Encoder weights: `smp-hub/resnet34.imagenet` at revision `7a57b34f723329ff020b3f8bc41771163c519d0c`, `model.safetensors` sha256 `6549ccd46a2d20a0a74fca866cfc5c6f9b3672d468b0a92aa94a58545a61103e`. They were downloaded once into the local Hugging Face cache, are hash-checked before use, and are read offline (`HF_HUB_OFFLINE=1`).
- **Environment.** The frozen training venv `/home/jc/bds-followup/.venv` is left unchanged (verified by `pip freeze` before and after).
  - smp and its dependencies are installed in an overlay venv, `/home/jc/bds-followup/.venv-kd`. Its `_bds_frozen_base.pth` file makes the frozen venv's site-packages importable, so torch 2.11.0+cu130 and every other existing package resolve from the frozen venv.
  - Packages added by the overlay: `revision/kd_comparator/env_overlay_kd.txt`.
  - All KD commands run with `/home/jc/bds-followup/.venv-kd/bin/python`.
- **Training** (config `configs/revision/kd/isic2018_kd_teacher_seed100.yaml`). One run with seed 100: distinct from the student seeds 1–10, the Experiment C seeds 11–17 and the benchmark seed 0.
  - Same partition and arrays as the students, no augmentation, 150 epochs.
  - AdamW, learning rate 1e-4, weight decay 1e-5, batch size 8 (OPEN-6). Cosine schedule, AMP, clipping 1.0.
  - Loss `L_ce + L_dice`, the students' segmentation objective. This was a technical choice the protocol had left open; no boundary term is used.
  - Checkpoint: the first maximum of validation DSC, as for the students.
  - Same entry point and resume mechanism as the students (`scripts/kd/train_kd.py`).
- **No teacher replacement or tuning.** One teacher, trained once. If it fails the quality gate (OPEN-1), the experiment stops at Phase 1 and the failure is reported. There is no automatic replacement, retraining, or out-of-fold scheme.
- **Caching** (`scripts/kd/cache_teacher.py`, a queue step after teacher training).
  - After selection the teacher is frozen. Foreground probabilities for the 2,075 training images are computed once in true fp32, with cuDNN/cuBLAS TF32 disabled, and stored as float16 in manifest order: `results/revision_runs/isic2018_kd_teacher_cache/train_teacher_prob_fp16.npy`. The sha256 is recorded in `cache_complete.json`, and every student checks it.
  - No online augmentation is used, so the cache is exact.
  - Check: 20 training images drawn with `numpy.random.default_rng(20261107)` are recomputed one at a time. The maximum absolute difference to the cache must be < 1e-3.
  - TF32 is disabled because, with PyTorch's default TF32 convolutions, single-image and batch-of-8 "fp32" outputs differed by up to 0.017 in the engineering check, while the true-fp32 difference was 2.5e-4 (fp16 rounding).
  - Students read the cache, and the teacher is not on the GPU during student training.
- **Teacher diagnostics** (`results/revision_runs/isic2018_kd_teacher_cache/diagnostics.json`; training and validation images only):
  - teacher DSC, IoU, Boundary F1, HD95 and ASSD on the training and validation sets;
  - a 10-bin histogram and the mean binary entropy (bits) of `p_t` inside the band;
  - mean `|p_t − y|` inside the band and over the whole image;
  - fraction of images with an empty band; mean band size.

  These describe how much information the soft targets carry beyond the ground truth. Their only decision use is the quality gate. They do **not** trigger design changes.

## 6. Phases

### Phase 1 — exploratory feasibility (training and validation sets only)

- **Runs** (queue `revision/queue_kd_phase1.yaml`, in this order): the teacher; the cache step; D+KD-band seeds 1, 2, 3. Decoder-only is not retrained.
- **Data used:** training and validation sets only. No new-arm test metric is computed or read.
- **Checks** (`analysis/kd/phase1_validation.py`):
  1. the teacher quality gate (OPEN-1) and the teacher diagnostics (§5);
  2. training stability: finite losses, `L_kd` decreasing, and the gradient-norm ratio and cosine of `λ_kd·L_kd` and `λ_b·L_boundary` relative to `L_seg`. The ratio and cosine are measured on a fixed probe batch (the first 8 training images in manifest order), in fp32, before training and after epochs 1, 25, 50, 75, 100, 125 and 150. The probe restores RNG and BatchNorm buffers and does not change `.grad`.
  3. validation Boundary F1, DSC and HD95 (also IoU and ASSD) of the 3 runs against the 10 existing decoder-only runs, as difference of means with an interval (§7);
  4. resources: per-epoch time, peak allocated and reserved VRAM, maximum GPU temperature, thermal pauses (§8).
- **Status of the results:** exploratory. With 3 runs against 10 on the 519 validation images, the expected 95% half-width is ±0.013 to ±0.026 Boundary F1, and the 80%-power detectable difference is 0.017 to 0.034 (§7).
  - On existing contrasts, validation differences were about one sixth to one third of the corresponding test differences.
  - Phase 1 cannot establish or rule out plausible effects. Phase 1 results are never described as confirmed, and a non-positive result is never described as evidence of no effect.
- **Go/no-go for Phase 2: Option A** (OPEN-2). Proceed if and only if:
  - **F1:** the teacher passed the quality gate.
  - **F2:** every Phase 1 student completed 150 epochs with all logged training and validation losses finite, and `L_kd` decreased (mean over epochs 141–150 below the epoch-1 mean).
  - **F3:** all Phase 1 runs completed through the queue at batch size 8, or at the pre-specified fallback (§8).

  The direction or size of any Phase 1 validation estimate is not part of the decision. The decision is recorded with its date in `revision/kd_comparator/DECISIONS_AND_DEVIATIONS.md` before any Phase 2 run, and Phase 2a is then approved in `revision/queue_kd_phase2a.yaml`. Approving an entry changes only its `approved` flag; configs and hashes are unchanged.

### Phase 2 — confirmation and specificity control

- **Phase 2a: confirmation of Q1.** D+KD-band seeds 4–10 (7 fresh runs; `revision/queue_kd_phase2a.yaml`). The seed set is fixed and will not be extended.
- **Phase 2b: Q2 control.** D+KD-full seeds 1–10 (10 runs; `revision/queue_kd_phase2b.yaml`).
  - It runs **only if C1 is confirmed as a gain** (Δ > 0) in Phase 2a, because Q2 is asked only if Q1 shows a gain and is tested after Q1 in a fixed sequence (§7).
  - Its design is fixed now, so the decision to run it introduces no design adaptation.
  - If C1 is not confirmed as a gain, Phase 2b is not run. This is reported as the planned consequence of the fixed sequence, not as an omission.
- **Selection bias from the Phase 1 decision.** Under Option A the decision does not depend on efficacy. In addition:
  - the confirmatory C1 test uses only the 7 fresh seeds (4–10), which are independent of the Phase 1 runs;
  - the all-seed estimate (seeds 1–10) is reported alongside but labelled as including the Phase 1 runs;
  - Phase 1 results are reported whether or not Phase 2 runs.

## 7. Statistical analysis

### Estimator

As in the manuscript (`analysis/revision/expansion_statistics.py::contrast` and `unit_values`, frozen at `3f1c62b`, unchanged):
- Δ = difference of arm means of run means; positive favours the first arm; HD95 and ASSD signs reversed.
- `se = √(se_run² + se_unit²)`. `se_run` is the Welch SE of run means; `se_unit` is the SD over images of the per-image difference of arm means, divided by √(number of images).
- Two-sided 95% t interval. The Welch–Satterthwaite df comes from the run-level variances only.
- Only images with a finite value in every run of both arms enter a contrast. Images where both prediction and target are empty are excluded.
- Implemented in `analysis/kd/phase1_validation.py` (Phase 1) and `analysis/kd/phase2_statistics.py` (Phase 2).
- The Phase 2 code was exercised before the freeze in `--dry-run` mode on recorded arms only (BDS-Lite standing in for D+KD-band). It reproduced the manuscript's BDS-Lite − decoder contrast, −0.0321 [−0.0518, −0.0124].

### Seed labels and pairing

Student seeds 1–10 are labels. The analysis is unpaired, as in the manuscript.

Because initial weights and batch order are functions of the seed and of the model construction only, a D+KD-band run starts from the same initial weights, with the same shuffle order, as the decoder-only run with the same seed number. This is verified by the unit test `test_same_seed_gives_decoder_only_initial_weights`. Two steps preserve it:
- the training entry point builds no test dataset; building one was verified to draw no random numbers;
- the cache is read without drawing from any random generator.

This shared randomness induces a positive correlation that the unpaired analysis ignores. Ignoring it is conservative for C1 and C3, and no paired analysis is added.

### Contrasts (test set, Phase 2 only)

- **C1 (primary).** D+KD-band (fresh seeds 4–10) − decoder-only (seeds 1–10), Boundary F1. Tested at α = 0.05.
- **C3 (secondary).** D+KD-band (seeds 4–10) − D+KD-full (seeds 1–10), Boundary F1. Tested at α = 0.05 **only if C1 is confirmed as a gain**. This is a fixed-sequence procedure; familywise error is controlled at 0.05 without splitting α.
- **Descriptive, not tested:**
  - all-seed versions of C1 and C3;
  - DSC, IoU, HD95 and ASSD;
  - missed-structure rate (target present, prediction empty) and false-alarm rate (target empty, prediction present; not applicable when no target is empty);
  - all-case HD95/ASSD with the evaluator's penalty for one-sided empty cases;
  - the new arms against BDS-Lite and U-Net;
  - the same contrasts on the validation set.

### Status and effect labels

- **Confirmed:** p < 0.05 at the contrast's position in the sequence, the interval excludes zero, and the all-seed estimate has the same sign.
- **Otherwise:** not established.
- Effect labels use the manuscript's smallest effects of interest (0.01 for Boundary F1, DSC and IoU; 2 px for HD95; 0.5 px for ASSD) and its labels: relevant gain/harm; gain/harm of unclear size; no relevant effect; undetermined. Only "no relevant effect", meaning an interval wholly inside ±SESOI, is evidence of absence.

### Power (computed from recorded runs; seed counts and tests unchanged)

**Inputs (Appendix A).** These are the per-run Boundary F1 means of the four existing ten-seed arms, seeds 1–10.

| Arm | Validation (519) mean / run SD | Test (1,000) mean / run SD |
|---|---|---|
| Decoder-only | 0.5376 / 0.0048 | 0.3940 / 0.0186 |
| U-Net | 0.5344 / 0.0067 | 0.3734 / 0.0216 |
| BDS-Lite | 0.5277 / 0.0074 | 0.3620 / 0.0225 |
| U-Net+GSL | 0.5315 / 0.0102 | 0.3633 / 0.0257 |
| Pooled (root mean square of the four SDs) | 0.0075 | 0.0223 |

The 95% χ² interval for the decoder-only run SD from 10 runs is [0.0033, 0.0087] on validation and [0.0128, 0.0340] on test.

**Image-level SE term (`se_unit`).** It was obtained by placing existing arms in each planned seed configuration against the decoder-only arm.

| Design | Test, vs decoder (all ordered arm pairs) | Validation, vs decoder (all pairs) |
|---|---|---|
| 3 vs 10 | 0.0019–0.0022 (0.0014–0.0022) | 0.0024–0.0028 (0.0023–0.0029) |
| 7 vs 10 | 0.0014–0.0018 (0.0012–0.0018) | 0.0019–0.0027 (0.0019–0.0027) |
| 10 vs 10 | 0.00145–0.00154 (0.0011–0.0015) | 0.0019–0.0021 (0.0017–0.0021) |

The largest vs-decoder value was used for Phase 1 and C1, and the largest all-pairs value for C3.

**Method.** The expected 95% half-width is `t_{0.975,df} · se`. The MDE is the Δ with 80% power for a two-sided α = 0.05 test, computed from the noncentral t distribution with the estimator's Welch df by numerical integration. At these df it exceeds the normal approximation (e.g. 0.028 vs 0.026 for C1-fresh).
- The reference decoder-only arm keeps its observed SD.
- The new arm's SD is taken under four scenarios: equal to decoder-only; pooled; largest observed (U-Net+GSL); and the upper 95% limit of the decoder-only SD.
- For C3 both arms are new and both take the scenario SD.

**Expected 95% half-width / 80%-power MDE, Boundary F1:**

| Comparison (data) | New SD = decoder | = pooled | = largest | = decoder upper 95% |
|---|---|---|---|---|
| Phase 1, 3 vs 10 (validation; the Phase 1 analysis) | ±0.013 / 0.017 | ±0.019 / 0.026 | ±0.026 / 0.034 | ±0.022 / 0.029 |
| Phase 1 design on test, for reference only (not run) | ±0.038 / 0.050 | ±0.046 / 0.062 | ±0.055 / 0.073 | ±0.077 / 0.101 |
| C1 fresh, 7 vs 10 (test) | ±0.020 / 0.028 | ±0.023 / 0.032 | ±0.026 / 0.036 | ±0.033 / 0.045 |
| C1 all-seed, 10 vs 10 (test, descriptive) | ±0.018 / 0.025 | ±0.020 / 0.028 | ±0.022 / 0.030 | ±0.027 / 0.037 |
| C3, 7 vs 10 (test) | ±0.020 / 0.028 | ±0.024 / 0.034 | ±0.028 / 0.039 | ±0.036 / 0.051 |
| C1 fresh, 7 vs 10 (validation, descriptive) | ±0.008 / 0.011 | ±0.010 / 0.013 | ±0.011 / 0.016 | ±0.010 / 0.014 |

**Sensitivity to `se_unit`.** With the new SD equal to decoder-only:
- C1-fresh MDE: 0.028 with `se_unit = 0`, 0.030 with twice the observed value.
- Phase 1 validation MDE: 0.013 with `se_unit = 0`, 0.026 with twice the observed value.

**Check against the manuscript.** The recorded runs reproduce the manuscript's ten-run test contrasts and all six validation contrasts of Supplementary S13 (Appendix A).

**Assumptions.**
- Run means are approximately normal.
- New-arm SDs fall in the range of the existing arms, which is not guaranteed for a distilled arm.
- `se_unit` for a new arm is like that of existing arms against decoder-only.
- Half-widths are expected values; realised intervals vary with the realised SDs.
- C3 power is conditional on C1 having been confirmed as a gain.

**Implications:**
- Phase 1 is a feasibility check, not a test.
- C1 can reliably detect only gains of roughly 0.028–0.036 Boundary F1 on test if the new arm's run SD is like that of the existing arms, and about 0.045 if it is at the upper limit. True gains of 0.01–0.02 may well be missed. For comparison, the decoder − U-Net estimate is +0.021.
- C3 has a similar or larger MDE (0.028–0.039) and may remain undetermined even when C1 is confirmed.
- "No relevant effect" (interval inside ±0.01) is out of reach on test for C1 and C3.
- Seed counts are fixed in advance by cost, not chosen to reach significance, and will not be extended after any result is seen.

### Mechanism (secondary, descriptive; Phase 2 only, after §9 step 8)

`analysis/kd/mechanism_probe.py`: a linear boundary readout from `F_s`. It uses `analysis/revision/boundary_transfer_probe.py` unchanged, with its image subsets, pixel-sampling seeds (1000 + seed, 2000 + seed) and ridge settings. Fitting is on validation pixels and AP is computed on up to 300 test images.

**Seeds aligned with the recorded decoder-only probe:** D+KD-band seeds 1–3 against the decoder-only rows (arm M2, seeds 1–3) of `revision/tables/families/boundary_transfer_probe.csv`. The earlier plan (D+KD-band seeds 4–6) had no recorded decoder-only counterpart. The script refuses to run before the step-8 test evaluation of all ten D+KD-band runs exists.

## 8. Compute and resource constraints

**Hardware and environment.**
- One NVIDIA GeForce RTX 5060 Laptop GPU, 8,151 MiB, driver 595.99.02.
- Python 3.14.4; PyTorch 2.11.0+cu130, CUDA 13.0, cuDNN 9.19.0 (frozen venv, `revision/env_freeze.txt`).
- smp 0.6.0 and its dependencies in the overlay venv (§5).
- Every KD entry point refuses a torch or CUDA version other than 2.11.0+cu130 / 13.0.
- Inputs 224×224, AMP.

**Batch size.**
- Teacher and students use batch size 8.
- The pre-specified fallback was batch size 4 with gradient accumulation over 2 steps (effective batch 8, learning rate unchanged), applied to all new student runs if batch size 8 did not fit.
- **Benchmark result: batch size 8 fits for both** (peak allocated 696 MiB teacher, 1,220 MiB student; §13). The fallback is therefore not used and not implemented.

**Timing.**
- Benchmark training-epoch times: teacher 10.6 s, student 13.2 s (§13).
- Expected time without thermal pauses, including validation-size inference: about 0.5 h for the teacher and 0.6–0.7 h per student. Thermal pauses add to this; each existing decoder-only run had 11–32 of them.

**Logging and thermal handling.** The existing mechanism's rules are reused unchanged, through copies in `scripts/kd/`:
- `scripts/kd/run_queue_kd.py`, a copy of `scripts/revision/run_queue.py`:
  - launches only below 80 °C with under 1 GiB in use, at least 25 GiB of disk free, and no GPU process;
  - logs `nvidia-smi` every 10 s to `<run>/gpu_telemetry.csv`;
  - sends SIGTERM at ≥ 85 °C, under 20 GiB free disk, or when the stop marker exists.
- The trainer then finishes the current epoch, saves `last.pt` (model, optimizer, scheduler, best state, all RNG and shuffle-generator states, hashes) and exits 130.
- `scripts/kd/supervise_kd.py`, a copy of the `supervise_queue_v2.py` rules:
  - waits for ≤ 70 °C and under 1 GiB in use (at most 20 min);
  - resumes only verified thermal stops or the cool-launch guard;
  - halts on any other failure, on the stop marker, or after five consecutive thermal restarts without saved-epoch progress.
- `last.pt` is also written at epoch 1, every 10 epochs and at epoch 150. A resume refuses any change to config, code or data.
- The engineering check showed that a stop after epoch 1 followed by a resume gives model weights bitwise identical to an uninterrupted run. The logged epoch-mean cross-entropy differed by ≤ 1.5e-8 between two same-seed runs, interrupted or not: non-deterministic reduction of the logged loss value, which does not enter the gradient.
- **Differences from the existing mechanism:**
  - the stop marker is `revision/STOP_KD_TRAINING`. `revision/STOP_TRAINING` is the author's stop for Experiment C, stays in place, and also blocks the existing supervisor's cooling and classification functions;
  - events are logged in `revision/kd_comparator/supervision/events.jsonl`;
  - the KD supervisor shares the controller lock `revision/training_supervision/controller.lock`, and the queue shares `revision/.gpu.lock`, so no KD job runs beside another queue or supervisor;
  - per-epoch peak allocated and reserved VRAM (`torch.cuda.max_memory_*`) is logged in `history.json`.
- A pause and resume is logged and does not count as a deviation if training resumes from the saved state.

**Storage.** Teacher cache 2,075 × 224 × 224 × 2 bytes ≈ 0.21 GB.

## 9. Execution order

1. **Implementation (done).**
   - Training: `scripts/kd/{common, kd_loss, teacher, data, train_kd, cache_teacher, run_queue_kd, supervise_kd}.py`.
   - Tooling: `scripts/kd/register_kd.py` (configs and queues), `scripts/kd/benchmark_kd.py` (step 2).
   - Analysis: `analysis/kd/{phase1_validation, evaluate_test, phase2_statistics, mechanism_probe}.py`.
   - Unit tests: `tests/revision/test_kd_comparator.py`. These cover the band against SciPy with a square element and zero border; the strict threshold; B=1 reducing to full-image BCE; an empty band giving zero loss and zero gradient; `λ_kd = 0` reproducing the decoder-only loss exactly; validation batches skipping the term; config locks; same-seed initial weights; cache order; the supervisor's thermal classification; and the queue refusing test outputs.
   - Existing sources under `src/bds_lite/`, `scripts/revision/`, `analysis/revision/` and `configs/` are unchanged. No file is added to `scripts/kd/` after the freeze, because the trainer hashes that folder and a resume would refuse.
2. **Engineering benchmark (done before freezing; not part of the experiment).** `scripts/kd/benchmark_kd.py`:
   - seed 0; one training epoch each for the teacher and a D+KD-band student;
   - the student read a dummy cache (the ground-truth masks);
   - no validation or test metric was computed, and no model was saved;
   - recorded: loss finiteness, epoch time, peak VRAM and temperature (`revision/kd_comparator/benchmark.json`, §13).

   A separate smoke check of the real entry points ran in a scratch directory on 64 training images and 2 epochs. Validation was replaced by a stub, and the cache step's "validation" pass used training images. It verified fresh, stopped and resumed runs, the teacher path, and the cache step with the gate. It produced no validation or test metric and is not kept.
3. **Freeze.** `scripts/kd/register_kd.py --write` writes the configs and the three queue files, with the sha256 of this file, every config, every source and every manifest. This file, the code, configs, queues and tests are committed together.
4. **Phase 1 command** (one GPU job; detached so that it survives the terminal):

   ```bash
   cd /home/jc/bds-lite && mkdir -p revision/kd_comparator/supervision && \
   setsid nohup env PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 \
     /home/jc/bds-followup/.venv-kd/bin/python scripts/kd/supervise_kd.py \
     --queue revision/queue_kd_phase1.yaml \
     > revision/kd_comparator/supervision/supervisor_stdout.log 2>&1 &
   ```

   This trains the teacher, runs the cache step (diagnostics, gate, cache, check), then trains D+KD-band seeds 1–3. If the gate fails, the queue stops before the first student. To pause safely: `touch revision/STOP_KD_TRAINING`; to resume, remove it and rerun the command.
5. **Phase 1 analysis** (CPU, validation only): `PYTHONPATH=src:. /home/jc/bds-followup/.venv-kd/bin/python analysis/kd/phase1_validation.py`.
6. **Phase 2 decision** by Option A, recorded with its date in `revision/kd_comparator/DECISIONS_AND_DEVIATIONS.md` before any Phase 2 run.
7. **Phase 2a:** approve and run `revision/queue_kd_phase2a.yaml` with the same command. Completeness is checked by the queue: 150 epochs, the selected checkpoint at the first maximum of validation DSC, artifact hashes, and no test output.
8. **Only then** evaluate D+KD-band (all seeds) and the teacher on the test set (`analysis/kd/evaluate_test.py`), and run `analysis/kd/phase2_statistics.py` at the frozen commit, followed by `analysis/kd/mechanism_probe.py`.
9. **Phase 2b** if and only if C1 is confirmed as a gain: approve and run `revision/queue_kd_phase2b.yaml`, then `analysis/kd/evaluate_test.py --phase2b` and `phase2_statistics.py` for C3.

λ_kd, the band definition, θ, the teacher, its hyperparameters and the architecture are never changed in response to any validation or test result.

## 10. Pre-specified interpretation

| Outcome | What may be written | What may not be written |
|---|---|---|
| C1 confirmed gain, C3 confirmed | In this setting (ISIC2018, compact U-Net, no augmentation, one external pretrained teacher), adding boundary-band probability distillation to auxiliary boundary supervision improved Boundary F1, and restricting distillation to the band added to whole-image distillation. | That the band restriction is necessary in other settings; generality beyond this setting. |
| C1 confirmed gain, C3 not established | Distillation from the external teacher added to auxiliary boundary supervision in this setting; whether the band restriction matters is unresolved. | That the gain is boundary-specific. |
| C1 not established, interval mostly positive | A gain could not be confirmed; the estimate is X [CI]. Gains above the upper bound are unlikely; smaller gains are not excluded. | "No effect"; "boundary distillation does not work". |
| C1 labelled "no relevant effect" | No relevant gain beyond auxiliary boundary supervision was found in this setting, at the 0.01 threshold. | Generalisation to other teachers, datasets or designs. |
| C1 confirmed harm | Reported as found, with the teacher diagnostics. Phase 2b is not run. | — |
| Gain over U-Net only (descriptive), not over decoder-only | The pattern reported in the literature (gain over a student without boundary signal) appears, but distillation is not shown to exceed simple auxiliary supervision. | That distillation adds to auxiliary supervision. |
| Phase 1 only (no Phase 2) | Exploratory validation estimates from 3 runs, with intervals; reason for stopping. | Any confirmatory claim, positive or negative. |

**In all cases:**
- A positive result does not establish why the original BDS-Lite transfer failed. The new arm changes the teacher (independent, pretrained, larger) and the place where the loss acts (output probabilities instead of a projection) at the same time. "The sibling teacher was too weak" or "the projection absorbed the loss" may be described as consistent with the result, not as shown by it.
- All original results of the manuscript are kept unchanged. Every run of this protocol, including stopped phases and unfavourable results, is reported in the supplement.

## 11. Optional controls (not planned)

- **D+GT-band:** the same loss with `B` derived from the ground-truth mask instead of the teacher mask. Not needed for Q1 or Q2.
- **Out-of-fold teachers:** not used, because there is no pre-existing evidence that they are needed. The teacher diagnostics are reported so that readers can judge how far the soft targets differ from the ground truth.
- **ACDC and Synapse, other students, augmentation:** out of scope.

## 12. Decisions (resolved before the freeze) and deviations

**[OPEN-1] Teacher quality gate — resolved (author, 2026-11-07).**

The teacher passes if its validation Boundary F1 ≥ 0.5376352265487206 **and** its validation DSC ≥ 0.8631866601530269. These are the means of the ten existing decoder-only runs (seeds 1–10) of `final_metrics.json` `metrics.val.<metric>.mean`, over the 519 validation images at their DSC-selected checkpoints. The cache step recomputes them from the ten files and refuses to run if they differ.

The teacher values are taken from the teacher run's own validation evaluation, with the same evaluator and settings as the decoder-only runs and at its DSC-selected checkpoint.

Rationale:
- The teacher must be at least as good as the typical student it is meant to teach, on both the distilled boundary metric and overall overlap.
- The mean is used instead of the maximum of ten runs (the earlier candidate), which would require beating the luckiest seed. No margin is added.
- Validation data are also the teacher's selection data. Like the decoder runs' Boundary F1, the teacher's Boundary F1 is measured at a DSC-selected checkpoint, not maximised.
- A FAILED gate stops the experiment at Phase 1 and is reported.

**[OPEN-2] Phase 1 → Phase 2 decision — resolved: Option A (author, 2026-11-07).** Proceed on feasibility only (F1–F3 in §6), regardless of the direction of the Phase 1 validation estimate. The confirmatory C1 uses fresh seeds only.

**[OPEN-3] Power — resolved (2026-11-07).** Computed from recorded runs (§7, Appendix A); seed counts and testing design unchanged.

**[OPEN-4] Seeds — resolved (2026-11-07).**
- Teacher seed: 100. Benchmark seed: 0. Students: 1–10.
- Pairing: §7, "Seed labels and pairing".
- The seeding procedure is the existing one:
  - `seed_everything(seed)` sets `random`, `numpy`, `torch` and `torch.cuda` seeds, `cudnn.benchmark = False` and `cudnn.deterministic = True`;
  - it does **not** set `torch.use_deterministic_algorithms` or `CUBLAS_WORKSPACE_CONFIG`;
  - the shuffle generator is `torch.Generator().manual_seed(seed)`;
  - RNG and generator states are restored on resume.
- Empirically, rebuilt seed-1 runs of the existing arms replayed archived validation DSC exactly (`revision/expansion_gate/gate.json`), and a KD student resumed after a stop matched an uninterrupted run bitwise in its weights (§8).

**[OPEN-5] Band and loss details — resolved (author, 2026-11-07).**

*Decided by the author:*
- 3×3 square structuring element;
- batch-level normalisation (one sum over all band pixels of the batch, divided by the total band count plus ε);
- ε = 1e-6.

*Technical choices made with them, all implementation choices, not BDKD-Net specifications (§4.3):*
- w = 3 is read as the side of that element, applied once;
- dilation and erosion use a zero-valued border, as SciPy does and as the repository's boundary targets and Boundary F1 contours do, so a lesion touching the image border has a band along the border. The square-element band differs from the cross-element boundary target only in connectivity;
- the threshold is strictly greater than 0.5;
- an image with an empty band contributes nothing to either sum, and a batch with no band pixel gives `L_kd = 0` exactly (zero gradient). No training ground-truth mask is empty or entirely foreground, so empty bands arise only from teacher masks;
- `p_t` is cached as float16 (rounding ≤ 2.5e-4 in the check), and the band is computed from the cached values the students read.

On training ground-truth masks this band averages 979 pixels per image (1.95%).

**[OPEN-6] Teacher hyperparameters — resolved (author, 2026-11-07).**
- AdamW, learning rate 1e-4, batch size 8, weight decay 1e-5, with the pre-specified VRAM fallback (not needed; §8).
- Cosine schedule over 150 epochs, AMP and clipping 1.0, as for the students.
- Chosen before training and not tuned.

**[OPEN-7] Decoder-only source of truth — verified (2026-11-07).** Configs, commits, evaluator and per-run validation metrics match the manuscript, and no class weights were used on ISIC2018 (Appendix A, B).

**Technical issues resolved during implementation** (each a documented choice; none changes the design, seed counts or tests):
1. **No automatic test evaluation:** the KD entry point builds no test dataset, and test evaluation is the separate step 8 script.
2. **Stop marker:** a KD stop marker, because `revision/STOP_TRAINING` (Experiment C) stays in place.
3. **Overlay venv** for smp; the frozen venv is unchanged.
4. **Teacher loss:** `L_ce + L_dice`.
5. **True fp32 cache** (TF32 disabled).
6. **Phase 2b trigger:** C3/Phase 2b follow only a confirmed *gain* in C1, since Q2 is conditional on Q1 showing a gain.
7. **Mechanism seeds** aligned with the recorded decoder-only probe (seeds 1–3).
8. **Post-freeze records:** decisions and deviations go in a separate dated file, because this file is hash-checked.

### Deviations

Recorded after the freeze in `revision/kd_comparator/DECISIONS_AND_DEVIATIONS.md` (date, section, change, reason, and which results had been seen).

## 13. Freeze record

- **Date (UTC):** 2026-11-07, before the first teacher or student training step.
- **Commit:** the local commit that adds this file version; its hash is in `revision/kd_comparator/DECISIONS_AND_DEVIATIONS.md`.
- **Registered hashes:** `revision/queue_kd_phase1.yaml`, `revision/queue_kd_phase2a.yaml` and `revision/queue_kd_phase2b.yaml` hold the sha256 of this file, every config, every source file the trainer records (`src/bds_lite/**`, `scripts/revision/**`, `scripts/kd/**`, `analysis/revision/{recover_predictions, expansion_statistics, family_statistics}.py`) and both split manifests.
- **Implementation sha256 at the freeze:**

  | File | sha256 |
  |---|---|
  | `scripts/kd/__init__.py` | `ae87c9256133cf733925131431077d70816ddb014ae594e8b08541d07ca3c62d` |
  | `scripts/kd/benchmark_kd.py` | `018fedc135eeef9addfda963bf393e6cef182309556aef7945c7dd6bcb66303f` |
  | `scripts/kd/cache_teacher.py` | `cb4349f182368fc5c23065eb708cbe362253c398f8efe4b5a48b232937482029` |
  | `scripts/kd/common.py` | `08100b25c0063089f81cb1093c89c1b7e7c8fcc3e8e2c72b31d328fb71f5a440` |
  | `scripts/kd/data.py` | `97d74e4c4876b182540df9cb5edafcc756ee2d715f35f18591999d53537121be` |
  | `scripts/kd/kd_loss.py` | `0bf3271f62abe962f6cb26430cac1d82f75023891657315d169355d1521a2ec4` |
  | `scripts/kd/register_kd.py` | `d8baf8e81614bd4577242eaa868f5fca683ee1d673fc471ee2eb755c2d28faae` |
  | `scripts/kd/run_queue_kd.py` | `b44c940819283de1b616fb4992e6d80d3b0ffe578b73b9a62bd242a1357eda2a` |
  | `scripts/kd/supervise_kd.py` | `05ad84bc0ed2716eefb0ab2c8139b5f9b70a836e5da10af34b8e1f71ead9b384` |
  | `scripts/kd/teacher.py` | `fce53ad67885576300e61568f29c6a43ef5e001ca6b6061717c75e2bf5d12c90` |
  | `scripts/kd/train_kd.py` | `6d11ddb3bd5d8d2148ae804089fdf8a8193b441044e38be9d325126b84e362ac` |
  | `analysis/kd/evaluate_test.py` | `2526bebbc918c2f663d4aec497a588b23e73c496145fd57ec4f7a3e3af0068c5` |
  | `analysis/kd/mechanism_probe.py` | `54ca7da34f7052c0d021b86cdac07d7b24c7de9ee45460d4f6da777743e594f7` |
  | `analysis/kd/phase1_validation.py` | `36a4eac4fb8d130e9bfcc4d8204f39d5bd8bcfc8944953f4fd37a9208a6a7fa7` |
  | `analysis/kd/phase2_statistics.py` | `eb7f68b93426d407b0b06e8a3194c9308721e06bee1a48e7047bc9d77699f0bb` |
  | `tests/revision/test_kd_comparator.py` | `914c2c43c46ec490e052526f9dfd7b407a392a83bb592bcd24d2cb5f4e11b775` |
  | `scripts/revision/train_run.py` | `10c0f722beb4bb4d8cea9e07fd0b30e594df9d69d140f3973d171e6dd6186cc5` |
  | `scripts/revision/training_support.py` | `a90945ceca92ba158df21ce3334dc6f661376cb92aa2eac5684b85d2ed016cfc` |
  | `analysis/revision/recover_predictions.py` | `29700c6421d909bc687e528a7dbbe60e2aba93a882d92facb5c9b4163c56282d` |
  | `analysis/revision/expansion_statistics.py` | `17099233fb6d2b67d769687109b3cc29a7b9a2f6e432006cad38e7e1d3c6b120` |
  | `analysis/revision/boundary_transfer_probe.py` | `4a22cc4deafcb10e832ce38edc00aa8ef088b83b34b5500b5bb2e7983689719e` |
  | `src/bds_lite/training/losses.py` | `67aaa72ac6beb8fe520347e70d82a74b6fcb30b64ba1d8afd635973dd91e0407` |
  | `src/bds_lite/models/bds_lite.py` | `38f29fb103238c7e021d843d7a698bd86cb328c582febb2a4f52f67af36611cd` |
  | `src/bds_lite/utils/seed.py` | `a753ef1e19bc99266365590b3452ef9e99c0250b9edec5415b6d1db67c37262c` |

  The last eight files are existing, unchanged sources that the KD code imports.

- **Environment:**
  - frozen venv `/home/jc/bds-followup/.venv` (torch 2.11.0+cu130, CUDA 13.0, cuDNN 9.19.0), unchanged;
  - overlay `/home/jc/bds-followup/.venv-kd` with `segmentation_models_pytorch` 0.6.0 and its dependencies (`revision/kd_comparator/env_overlay_kd.txt`);
  - encoder weights `smp-hub/resnet34.imagenet` @ `7a57b34f…`, sha256 `6549ccd4…` (§5).
- **Benchmark** (`revision/kd_comparator/benchmark.json`; seed 0, one training epoch each, batch size 8, dummy cache, no metric):

  | | Training epoch | Validation-size inference | Peak allocated / reserved | Max temperature | Losses finite |
  |---|---|---|---|---|---|
  | Teacher | 10.6 s | 0.9 s | 696 / 852 MiB | 68 °C | yes |
  | Student | 13.2 s | 1.3 s | 1,220 / 1,828 MiB | 72 °C | yes |

  Batch size 8 fits; the fallback is not used.
- **Teacher cache sha256:** written by the cache step to `results/revision_runs/isic2018_kd_teacher_cache/cache_complete.json` and verified by every student.

## 14. Known limitations (stated in advance)

- One teacher, trained once. Teacher capacity, pretraining, and independence change together, and the location of the loss also differs from BDS-Lite, so none of these can be isolated.
- ISIC2018 only. No augmentation. One compact student.
- BDKD-Net settings come from polyp segmentation at a different resolution with augmentation, and are not tuned for ISIC2018. The band, its border and its normalisation are this protocol's implementation choices (§4.3).
- With ten or fewer runs per arm, only gains of roughly 0.028–0.036 Boundary F1 on test can be detected with 80% power in C1 (§7). Smaller gains may be missed, so a non-significant result is not evidence of no effect.
- The ISIC2018 official test set was analysed earlier in the manuscript, and the comparator was chosen with knowledge of those results.
- The teacher sees all training images, so its soft targets on those images may carry little information beyond the ground truth. This is reported through the diagnostics (§5), not corrected.
- Same-seed runs of the new and existing arms share initial weights and batch order; the unpaired analysis ignores this correlation (§7).

---

## Appendix A. Recorded per-run Boundary F1 (seeds 1–10) used in §7

Per-run means over images, computed with `expansion_statistics.py::unit_values` on the files listed in §3. These values were all read from existing result files; none is new.

**Validation (519 images), checkpoint selected by validation DSC:**

| Arm | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | s10 | Mean | SD (all) | SD (s4–10) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Decoder-only | 0.5405 | 0.5429 | 0.5455 | 0.5348 | 0.5424 | 0.5323 | 0.5328 | 0.5370 | 0.5350 | 0.5333 | 0.5376 | 0.0048 | 0.0035 |
| U-Net | 0.5293 | 0.5428 | 0.5370 | 0.5259 | 0.5325 | 0.5281 | 0.5386 | 0.5314 | 0.5322 | 0.5465 | 0.5344 | 0.0067 | 0.0070 |
| BDS-Lite | 0.5288 | 0.5307 | 0.5364 | 0.5269 | 0.5301 | 0.5216 | 0.5189 | 0.5277 | 0.5159 | 0.5398 | 0.5277 | 0.0074 | 0.0080 |
| U-Net+GSL | 0.5427 | 0.5313 | 0.5456 | 0.5211 | 0.5407 | 0.5149 | 0.5247 | 0.5358 | 0.5233 | 0.5354 | 0.5315 | 0.0102 | 0.0094 |

**Official test (1,000 images):**

| Arm | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | s10 | Mean | SD (all) | SD (s4–10) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Decoder-only | 0.3717 | 0.3706 | 0.3958 | 0.3835 | 0.4087 | 0.3911 | 0.3758 | 0.4207 | 0.4184 | 0.4040 | 0.3940 | 0.0186 | 0.0173 |
| U-Net | 0.3453 | 0.3671 | 0.3777 | 0.3548 | 0.3531 | 0.3628 | 0.3903 | 0.3797 | 0.3845 | 0.4184 | 0.3734 | 0.0216 | 0.0232 |
| BDS-Lite | 0.3498 | 0.3771 | 0.3740 | 0.3884 | 0.3506 | 0.3391 | 0.3629 | 0.3843 | 0.3175 | 0.3759 | 0.3620 | 0.0225 | 0.0258 |
| U-Net+GSL | 0.3701 | 0.3798 | 0.4015 | 0.3635 | 0.3756 | 0.3099 | 0.3404 | 0.3833 | 0.3606 | 0.3479 | 0.3633 | 0.0257 | 0.0246 |

**Decoder-only selection record:**
- Selected epochs (s1–s10): 57, 49, 56, 37, 52, 47, 56, 46, 61, 42.
- `best_val_dsc` (s1–s10): 0.8657, 0.8606, 0.8654, 0.8636, 0.8644, 0.8629, 0.8624, 0.8643, 0.8606, 0.8619.

**Ten-run validation contrasts (Boundary F1), recomputed (S13 shows them rounded to three decimals):**
- decoder − U-Net +0.0032 [−0.0036, +0.0100]
- BDS-Lite − U-Net −0.0067 [−0.0147, +0.0013]
- BDS-Lite − decoder −0.0099 [−0.0173, −0.0026]
- GSL − U-Net −0.0029 [−0.0119, +0.0061]
- GSL − BDS-Lite +0.0038 [−0.0053, +0.0130]
- GSL − decoder −0.0061 [−0.0150, +0.0028]

All six match S13 and `audit_validation_contrasts.csv`.

**Not verified in the audit:** the U-Net and BDS-Lite seed 1–3 training configurations, which come from the follow-up study (Supplementary S5, `revision/GITHUB_AUDIT.md`), and the GSL runs' configurations. Only their recorded per-image results were used.

## Appendix B. Code–protocol discrepancies found in the 2026-11-07 audit, and their resolution

| # | Earlier wording | What the code does | Resolution |
|---|---|---|---|
| D1 | "no projection `P`, no gate" | `P` and the gate are instantiated (scale 0.0, exact identity, no gradient to `P`) | Stated exactly in §4.1 |
| D2 | "checkpoint at maximum validation DSC" | First maximum, strict `>`, per-image mean DSC over 519 images | Stated in §§4.1, 5 |
| D3 | No new-arm test metric before step 8 | `train_run.py` evaluates the test set at the end of every run | The KD entry point builds no test dataset; separate step-8 script; queue refuses test outputs |
| D4 | "hook into the existing trainer" | Queue runners launch only registered entry points; sources are hash-checked | New KD queue runner and supervisor; existing sources unchanged |
| D5 | (implicit) | Validation batches carry no teacher cache | `L_kd` skipped without `p_t` (§4.2) |
| D6 | "peak allocated VRAM recorded by existing monitoring" | Only `nvidia-smi` `memory.used` recorded | Per-epoch `max_memory_allocated/reserved` in `history.json` |
| D7 | Benchmark with seed 0, no validation | Existing entry points validate every epoch and restrict seeds | Separate `benchmark_kd.py` |
| D8 | Phase 1 detectable difference from test-like SDs | Phase 1 is analysed on validation | Recomputed (§§6, 7) |
| D9 | Mechanism D+KD-band seeds 4–6 vs decoder-only | Recorded decoder-only probe covers seeds 1–3 only | Aligned to seeds 1–3 (§7) |
| D10 | Teacher weight decay unstated | — | 1e-5 (OPEN-6) |
| D11 | smp not installed | — | Overlay venv; frozen venv unchanged (§5) |
| D12 | (found in the engineering check) | TF32 convolutions make teacher outputs batch-size dependent (up to 0.017) | Cache computed with TF32 disabled (§5) |
| D13 | (found in the engineering check) | `revision/STOP_TRAINING` blocks the existing supervisor's functions | KD stop marker and KD supervisor (§8) |
