# Protocol: external-teacher boundary-band distillation on ACDC and Synapse (exploratory)

**Status: FROZEN 2026-11-07, before any ACDC or Synapse teacher or student training and before any new-arm test metric existed.** The freeze is the local commit that first contains this file version with the implementation listed in §9. Its hash is recorded in `revision/kd_multiclass/DECISIONS_AND_DEVIATIONS.md`.
- The sha256 of this file is registered in every config and in the queue, and the queue refuses to launch or resume if this file changes. This file is therefore never edited after the freeze.
- Dated decisions and deviations go in `revision/kd_multiclass/DECISIONS_AND_DEVIATIONS.md`.

**Relation to the ISIC2018 study.** This protocol extends the ISIC2018 comparator (`revision/KD_COMPARATOR_PROTOCOL.md`, frozen at `f7f771f`; three-seed exploratory design per `revision/KD_COMPARATOR_AMENDMENT_1.md`, `24e5396`) to the two multiclass datasets of the manuscript. The same method, teacher family, loss weight and band rule are used, adapted only where multiclass labels require it (§4). One unified analysis covers all three datasets (§7).

**Requested by** the author on 2026-11-07: three student seeds per dataset; one independently trained, frozen teacher per dataset; existing student recipes, budgets, selection, splits and evaluation preserved; no tuning.

Manuscript: PeerJ Computer Science CS-2026:06:142937, *When does boundary distillation help lightweight medical image segmentation?* (revision). Title kept.

## 1. Question

Does external-teacher boundary-band probability distillation (D+KD-band) add to the decoder-only arm's auxiliary boundary supervision on ACDC and Synapse? D+KD-band is also compared with U-Net and BDS-Lite.

The study is **exploratory**:
- three runs per arm;
- no confirmatory test and no "confirmed" status;
- no D+KD-full control, so no claim of boundary specificity can be made.

## 2. Data and existing references (verified 2026-11-07; not retrained)

| | ACDC | Synapse |
|---|---|---|
| Classes | 4 (background, RV, myocardium, LV) | 9 (background, 8 organs) |
| Images | 1-channel MRI, stored per-volume z-scored (volume mean 0.000, SD 0.987) | 1-channel CT, stored windowed to [0, 1] |
| Training / validation / test slices | 1,350 / 186 / 366 | 1,738 / 473 / 1,568 |
| Manifests (split seed 2026) | `recovered/followup/splits/acdc/{train,val,test}.json` | `recovered/followup/splits/synapse/{train,val,test}.json` |
| Decoder-only (seeds 1–3) | `results/revision_runs/acdc_m2_aux_boundary_only_unweighted_seed{1,2,3}/` (the corrected, unweighted runs) | `results/revision_runs/synapse_m2_aux_boundary_only_seed{1,2,3}/` |
| U-Net (seeds 1–3) | `revision/prediction_recovery/main_p1b_unet/acdc_seed{1,2,3}/test/per_case.csv` | `revision/prediction_recovery/main_p1b_unet/synapse_seed{1,2,3}/test/per_case.csv` |
| BDS-Lite (seeds 1–3) | `results/revision_runs/acdc_m6_full_unweighted_seed{1,2,3}/` | `revision/prediction_recovery/main_p1b_full/synapse_seed{1,2,3}/test/per_case.csv` |

- These are the manuscript's retrained references, located with `analysis/revision/expansion_statistics.py::cell_path` (cells M2, M0, M6), with the same evaluator (`recover_predictions.py::case_metrics`).
- Seeds 4–10 of these datasets belong to Experiment C, which was stopped by the author. They are not used, and their metrics are not read.
- The test sets of these references have been analysed in the manuscript. The new arm's design is fixed here before any of its test metrics exist, and its test set is evaluated once (§6).

## 3. Teacher (one per dataset)

- **Model.** `segmentation_models_pytorch` 0.6.0 `Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=K)`, with the same pinned and hash-checked encoder weights as ISIC2018 (`smp-hub/resnet34.imagenet` @ `7a57b34f…`, sha256 `6549ccd4…`). Wrapper: `scripts/kd_mc/teacher.py`.
- **Input adaptation** (no tuning). The single channel is replicated to three channels, so that the pretrained first convolution is used unchanged.
  - Synapse (in [0, 1], like ISIC2018) then receives ImageNet mean/std normalisation, as ISIC2018 does.
  - ACDC is stored per-volume z-scored. That is already the zero-mean, unit-variance scale that ImageNet normalisation produces, so it is passed unchanged.
- **Training.** One run per dataset, seed 100. The data, partition, arrays and checkpoint rule (first maximum of validation DSC) are those of the students.
  - Optimiser, schedule and budget equal the dataset's locked student recipe: AdamW, learning rate 1e-4, weight decay 1e-5, batch size 16, cosine over 150 epochs, AMP, clipping 1.0. The 1e-4 learning rate is also the ISIC2018 teacher's.
  - Loss `L_ce + L_dice`, unweighted (the students' segmentation objective).
  - Configs: `configs/revision/kd_mc/{acdc,synapse}_kd_teacher_seed100.yaml`.
- **No replacement or tuning.** One teacher per dataset, trained once.
- **Quality gate** (same rule as ISIC2018). The teacher's validation Boundary F1 and validation DSC are taken from its own `final_metrics.json` (mean over all slice–class rows, the same evaluator and definition as the decoder-only record). Each must be at least the mean of that dataset's decoder-only seeds 1–3.
  - If the gate fails, that dataset's students are not trained. This is reported, and the queue continues with the next dataset.
  - For reference, the decoder-only validation means are: ACDC Boundary F1 0.8356, DSC 0.7885; Synapse 0.8447, 0.8399 (exact values are recomputed by the cache step from the three files).
- **Cache** (`scripts/kd_mc/cache_mc.py`).
  - True fp32 inference (TF32 disabled). The K-class softmax for every training slice is stored as float16 in manifest order (`results/revision_runs/{dataset}_kd_teacher_cache/train_teacher_prob_fp16.npy`; about 0.54 GB for ACDC and 1.57 GB for Synapse), and its sha256 is checked by every student.
  - Check: 20 training slices drawn with `numpy.random.default_rng(20261107)` are recomputed one at a time; the maximum absolute difference must be < 1e-3.
  - Diagnostics, training and validation only:
    - per-class and macro DSC/Boundary F1 and all five metrics;
    - band size and empty-band fraction;
    - normalised categorical entropy of `p_t` in the band (10-bin histogram);
    - total-variation distance between `p_t` and the one-hot label, in the band and over the image.

## 4. Students and the multiclass loss

**Student recipe.** Each student config equals the dataset's registered decoder-only config (seed 1 template; the seed 1–3 configs differ only in the seed) in its `data`, `dataset`, `model`, `loss` and `trainer` blocks and its manifests:
- `bdslite_unet` with inert `P` and gate (scale 0);
- training-only `D_b` with `L_boundary` (BCE + Dice, union two-pixel target, `λ_b = 0.05`);
- `L_ce + L_dice` with no class weights;
- AdamW 1e-4, weight decay 1e-5, batch 16, 150 epochs, cosine, AMP, clipping 1.0;
- checkpoint at the first maximum of validation DSC.

Only `project.seed`, `revision.*` and the `kd` block differ, and this is enforced by `scripts/kd_mc/common.py::require_protocol`.

Seeds are 1, 2 and 3, as for the existing references. A student with seed s starts from the same initial weights and batch order as the decoder-only run with seed s; this is verified by a unit test. The analysis is unpaired, as in the manuscript, which is conservative.

**Loss.** `L = L_seg + λ_b · L_boundary + λ_kd · L_kd`, with `λ_kd = 1.0` (as ISIC2018). `L_kd` is defined in `scripts/kd_mc/loss.py` and computed in fp32 with autocast off:
- Teacher hard map `M_t = argmax_k p_t,k`.
- Band `B = ⋃_{c ≥ 1} [dilate(1[M_t = c]) − erode(1[M_t = c])]`, using a 3×3 square element applied once and a zero-valued border. The band is the union of the morphological gradients of all foreground classes; it covers class-to-class and class-to-background contours and excludes background, as the repository's multiclass boundary targets do (union over labels ≠ 0).
- Per-pixel loss: soft-target cross-entropy `−Σ_k p_t,k log p_s,k`, with both distributions clamped to [1e-6, 1 − 1e-6].
- `L_kd = Σ_batch B · CE / (Σ_batch B + ε)`, with ε = 1e-6. An empty band contributes nothing.
- **Reduction property.** For K = 2 this loss is exactly the ISIC2018 loss. The argmax equals the strict threshold at 0.5, and the two-class soft cross-entropy equals the binary BCE. This is verified by the unit test `test_two_class_case_reduces_exactly_to_the_isic_loss`. No temperature, class weighting or per-class normalisation is added; these would be new design choices without precedent in the ISIC2018 arm.

`L_kd` is skipped on validation batches, which carry no cache. This affects only the logged validation loss.

## 5. Implementation constraints (as for ISIC2018)

- The training entry point (`scripts/kd_mc/train_mc.py`) builds **no test dataset**. The manifests' test split is not loaded during training, and the queue refuses any run folder that contains a test output.
- Thermal handling and resume copy the existing rules (`scripts/kd_mc/run_queue_mc.py`, `scripts/kd_mc/supervise_mc.py`):
  - launch below 80 °C with under 1 GiB in use and at least 25 GiB of disk free;
  - SIGTERM at ≥ 85 °C, followed by an end-of-epoch checkpoint and exit 130;
  - cooling to ≤ 70 °C before relaunch; resume only verified thermal stops; halt after five consecutive no-progress restarts;
  - `last.pt` at epoch 1, every 10 epochs, at epoch 150 and on stop;
  - a resume refuses any change to config, code or data.
- Stop marker `revision/STOP_KD_TRAINING` (shared by the KD studies). `revision/STOP_TRAINING` (Experiment C) is left in place. The controller lock and GPU lock are shared, so this study never runs beside another queue.
- Per-epoch time and peak allocated/reserved VRAM are logged. Gradient probes of `λ_kd·L_kd` and `λ_b·L_boundary` relative to `L_seg` are taken on a fixed batch, the first 8 training slices in manifest order that contain any foreground label (many slices are background-only, which would give an empty band): before training and after epochs 1, 25, 50, 75, 100, 125 and 150.
- Environment: the frozen venv, unchanged, plus the smp overlay `/home/jc/bds-followup/.venv-kd` (as ISIC2018).
- Files in `scripts/kd_mc/` are not changed or added while runs are in progress, because the trainer hashes that folder.

## 6. Execution and test evaluation

1. **Queue** `revision/queue_kd_mc.yaml`, in priority order: ACDC teacher → ACDC cache step (gate) → ACDC students 1–3 → Synapse teacher → Synapse cache step → Synapse students 1–3. It is run by:

   ```bash
   cd /home/jc/bds-lite && mkdir -p revision/kd_multiclass/supervision && \
   setsid nohup env PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 \
     /home/jc/bds-followup/.venv-kd/bin/python scripts/kd_mc/supervise_mc.py \
     --queue revision/queue_kd_mc.yaml \
     > revision/kd_multiclass/supervision/supervisor_stdout.log 2>&1 &
   ```

2. **Integrity checks** (`analysis/kd3/integrity.py --dataset <d>`), after the three students of a dataset finish:
   - 150 contiguous epochs; validation-only final marker; artifact hashes;
   - config bytes and source hashes equal to the registration;
   - selected epoch equal to the first maximum of validation DSC;
   - finite losses; cache hash in each student's provenance; gate PASSED;
   - seeds 1–3 with differing first-epoch losses; no test output.

   Feasibility: `L_kd` decreased in every student (mean of epochs 141–150 below epoch 1). Runs that fail are reported, not replaced.
3. **Test evaluation, once** (`analysis/kd3/evaluate_test.py --dataset <d>`; refuses unless the integrity checks pass): three students and the teacher, using the unchanged `scripts/revision/train_run.py::evaluate` and the same test loader as the existing arms (registered test manifest, batch 16, 4 workers).
4. Validation data alone select checkpoints. No hyperparameter is changed in response to any result.

## 7. Analysis (unified across ISIC2018, ACDC and Synapse; `analysis/kd3/unified_analysis.py`)

Frozen with this protocol and before any new-arm test metric of any dataset existed.
- **Estimator.** The manuscript's frozen `expansion_statistics.py::contrast` and `unit_values`, unchanged:
  - per-image values; for ACDC/Synapse, the label macro per slice, then the mean over slices of each patient/case unit;
  - Δ = difference of arm means of run means;
  - SE combines the Welch SE of run means with the unit-level SE; Welch df; 95% t interval;
  - HD95/ASSD signs reversed, so positive favours D+KD-band.
- **References per dataset.**
  - ACDC/Synapse: decoder-only, U-Net and BDS-Lite, seeds 1–3.
  - ISIC2018: seeds 1–10 (primary), plus seeds 1–3 (seed-matched sensitivity), per Amendment 1.
- **Primary (exploratory).** D+KD-band − decoder-only, test Boundary F1, per dataset. The p value is reported unadjusted and Holm-adjusted over the three datasets.
- **Descriptive:**
  - all five metrics (DSC, IoU, Boundary F1, HD95, ASSD) against all three references;
  - per-seed values and mean ± SD; seed consistency (how many D+KD-band runs exceed the decoder-only mean; the minimum and maximum separation);
  - per-class deltas for ACDC and Synapse (DSC, Boundary F1, HD95);
  - missed-structure and false-alarm rates;
  - validation-set contrasts against decoder-only;
  - an inverse-variance pooled D+KD-band − decoder-only per metric across the three datasets, with Cochran's Q (exploratory);
  - teacher test metrics;
  - training cost (epoch time, total epoch hours, peak VRAM, thermal pauses, teacher cost) and the deployed parameter count, which equals decoder-only by construction.
- **Readings** (no "confirmed"):
  - interval wholly beyond ±SESOI (0.01 for DSC/IoU/Boundary F1, 2 px for HD95, 0.5 px for ASSD): exploratory relevant gain/harm;
  - interval excluding zero only: exploratory gain/harm of unclear size;
  - interval inside ±SESOI: no relevant effect;
  - otherwise: not established.
- **All datasets and all metrics are reported**, whatever their direction.

**Expected precision** (from the recorded seed 1–3 runs, same estimator; new-arm SD equal to the decoder-only SD, up to the largest reference SD). Expected 95% half-width / 80%-power detectable difference, 3 vs 3 runs:
- ACDC: Boundary F1 ±0.024–0.026 / 0.032–0.035; DSC ±0.025–0.032 / 0.033–0.042.
- Synapse: Boundary F1 ±0.055–0.067 / 0.075–0.090; DSC ±0.052–0.068 / 0.070–0.092.
- ISIC2018 (3 vs 10): ±0.038–0.055 / 0.050–0.073.

Only large effects can be distinguished from zero, so a "not established" reading is not evidence of no effect.

## 8. Interpretation rules (fixed in advance)

- A dataset with an exploratory gain on Boundary F1 against decoder-only may be described as "a gain in this exploratory three-seed comparison, in this setting". It may not be described as confirmed, general, or boundary-specific.
- Consistency across datasets is described only from the per-dataset readings and the pooled estimate with its heterogeneity. A favourable dataset is never reported without the others.
- Gains against U-Net without a gain against decoder-only repeat the manuscript's central caution: they do not show that distillation adds to simple auxiliary supervision.
- Any positive result changes the teacher (independent, pretrained, larger) and the place where the loss acts (probabilities instead of features) at the same time. Mechanistic explanations of BDS-Lite's earlier failure are "consistent with" at most.
- The manuscript is not modified by this study; integrating any result is a separate author decision.

## 9. Freeze record

- **Date (UTC):** 2026-11-07, before any ACDC/Synapse teacher or student training step.
- **Commit:** the commit that adds this file version (hash in `revision/kd_multiclass/DECISIONS_AND_DEVIATIONS.md`).
- **Registered hashes:** `revision/queue_kd_mc.yaml` holds the sha256 of this file, every config, every source file the trainer records (`src/bds_lite/**`, `scripts/revision/**`, `scripts/kd/**`, `scripts/kd_mc/**`, and the evaluator and estimator files) and the manifests.
- **Implementation and analysis sha256 at the freeze:**

  | File | sha256 |
  |---|---|
  | `scripts/kd_mc/__init__.py` | `ca28539f3859e22c186c4061d3577a72b4aeb0d10f16abcb425eb36b6bae83e9` |
  | `scripts/kd_mc/benchmark_mc.py` | `57c7c6b3e69012d032c51c46e54348a92cd30edb685d49a1e937e85673f431bd` |
  | `scripts/kd_mc/cache_mc.py` | `76196bc1eb26708fa971ffee76774c1412b82d5962cc7aa27fac8396222f669b` |
  | `scripts/kd_mc/common.py` | `2a02fa2c605d9b14e44083bc5d1fa9e68f2e243a237f6b99a1c902a41fe1cfc5` |
  | `scripts/kd_mc/data.py` | `6fa73a250ca906e208ea306964cea15fcfcafbfd6a69622fa8a8af89ca8c2f39` |
  | `scripts/kd_mc/loss.py` | `1b6d572b4f56ad21d8070043f888998b018b3894995f95ba4249862d52ebe309` |
  | `scripts/kd_mc/register_mc.py` | `6dc24f3916ee1803eb0e03e79b27dd828dffd1e5c41d68988d7c82115032018c` |
  | `scripts/kd_mc/run_queue_mc.py` | `e63da7faa0ddb5d2665786c4222181d484cf75fbe0cbaa7022c6f1637915e404` |
  | `scripts/kd_mc/supervise_mc.py` | `4a0de33f0479955d1b042bd294f8af29d0869715c37a5d2694a435a2bd8366c6` |
  | `scripts/kd_mc/teacher.py` | `ad991dffae908e69bd97f243947b93ee5742330e700733b984fdb3e47466a686` |
  | `scripts/kd_mc/train_mc.py` | `e137df0c790e53bedf7cd44dfe53da9a319b36d7d067c0c4774b0dc911f56634` |
  | `analysis/kd3/evaluate_test.py` | `ee6a2a4adbfa16131d86e6fd018c6d5d2b43926082e73367219c70a15b60a2e6` |
  | `analysis/kd3/integrity.py` | `b3358a181f8c680a7e1a58cbe55c8bac33059c41bd37d9859d7ff094c69dd361` |
  | `analysis/kd3/unified_analysis.py` | `8fd6823665d03585021e8acd9a5c9bb700efa27a9ea6f3b442bcffce4abb39a4` |
  | `tests/revision/test_kd_multiclass.py` | `2b4796b253593311813c90c6fc482970fa10606ad51ba31b326d481297f3ac4c` |
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
  | `scripts/revision/train_run.py` | `10c0f722beb4bb4d8cea9e07fd0b30e594df9d69d140f3973d171e6dd6186cc5` |
  | `analysis/revision/recover_predictions.py` | `29700c6421d909bc687e528a7dbbe60e2aba93a882d92facb5c9b4163c56282d` |
  | `analysis/revision/expansion_statistics.py` | `17099233fb6d2b67d769687109b3cc29a7b9a2f6e432006cad38e7e1d3c6b120` |
  | `analysis/revision/make_audit_assets.py` | `8fd017205832af185a9df8eee1140bddb082c4aa790c864b62a848da3f365cef` |

  `scripts/kd/*` (ISIC2018, frozen at `f7f771f`) and the last four files are existing sources, imported unchanged.

- **Benchmark** (`revision/kd_multiclass/benchmark.json`; seed 0, one training epoch, dummy one-hot cache, no metric):

  | Dataset | Model | Batch | Training epoch | Peak allocated / reserved | Max temperature | Losses finite |
  |---|---|---|---|---|---|---|
  | acdc | teacher (24,436,804 parameters) | 16 | 6.6 s | 1,043 / 1,126 MiB | 66 °C | yes |
  | acdc | student (2,197,493 parameters) | 16 | 8.6 s | 2,086 / 2,522 MiB | 70 °C | yes |
  | synapse | teacher (24,437,529 parameters) | 16 | 7.2 s | 1,512 / 1,620 MiB | 73 °C | yes |
  | synapse | student (2,197,658 parameters) | 16 | 11.5 s | 2,325 / 2,804 MiB | 74 °C | yes |

  Batch size 16 fits every model. No memory fallback is needed, and none is implemented.

## 10. Limitations stated in advance

- Three runs per arm. The intervals are wide (§7), and nothing here is confirmatory.
- One teacher per dataset, trained once. Teacher capacity, pretraining and independence change together.
- The multiclass adaptation (argmax band over foreground classes, soft cross-entropy) is this protocol's generalisation, not a published specification. It reduces exactly to the ISIC2018 loss for two classes.
- The teacher sees all training slices, so its soft targets may carry little beyond the labels. This is reported through the diagnostics.
- The ACDC U-Net reference is low in the retrained set (manuscript caveat), which inflates any contrast against it.
- No D+KD-full control, so boundary specificity is not assessable.
- Slice-level 2D evaluation, as in the manuscript's main analysis; volume-level metrics are not computed.
- An engineering smoke check before the freeze (scratch directory, 64 training slices, 2 epochs, validation stubbed) verified fresh, stopped and resumed students (bitwise-identical weights), the teacher path and the cache step with its gate on both datasets. It produced no validation or test metric and was discarded.
