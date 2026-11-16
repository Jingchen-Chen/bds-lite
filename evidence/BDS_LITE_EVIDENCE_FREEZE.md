# BDS-Lite — final scientific evidence reference v1.0.0

## 1. Authority, cutoff and preservation scope

**Authoritative scientific reference for subsequent Fable/Opus manuscript-only revision.**
Raw-evidence cutoff: **2026-11-10 20:31:26 UTC**. Derived verification and document
assembly occur after that cutoff; they do not backdate new raw evidence.
Source checkout: `5b2ff7a52edf906f16fdea52d7c6a9b4bad282d8`, branch `revision-followup`,
plus the explicitly preserved working-tree source snapshots and audit additions.
The repository contained previously staged editorial changes; the freeze commits
exclude manuscript and reviewer/editor-letter changes. Those files are not the
scientific source of truth and have not been rewritten in this phase.

Governing claim boundary: **`revision/CLAIM_BOUNDARY_v4.md`**. The author explicitly
authorized its adoption and the AGENTS.md pointer update. V2, V3, the archived BSPC
anchor and the provisional reference are retained as historical records. If a
presentation, generated number, old audit or historical boundary conflicts with
this reference, use the current verified evidence and disclose the discrepancy.
Do not edit original outcomes, protocols or earlier decisions to fit the prose.

**Freeze scope: finalized statistical-evidence preservation, not complete portable
training/inference reproduction.** The historical selected archive is retained,
and an additive final supplement preserves previously ignored primary/structured
records, E3 official-test case outputs, source snapshots and verification receipts.
SHA-256 is checked against actual bytes and archive members. Large datasets,
checkpoints, prediction maps and caches are not all copied into independently
backed-up storage. The dependency manifest identifies current local bytes and
missing historical files; a tag cannot recover absent bytes. See §§18 and20.

**Evidence labels.** VERIFIED CORRECT means an explicitly stated independent
calculation, implementation check or hash check passed; its scope is stated.
VERIFIED ERROR identifies a demonstrated historical/current defect, not merely
a scientific limitation. UNRESOLVED means the evidence cannot identify the
answer; INCOMPLETE means required provenance, artifacts or measurements are
missing. Confirmatory/exploratory/post hoc/superseded/excluded describe study
status, independently of arithmetic verification. REPORTED tables are traceable
saved results whose original predictions/feature extraction were not independently
reconstructed here; they are not cleared for a new headline inference.

**New final-stage evidence.** E3 training ended at 20:18:57 UTC (20/20 runs).
Its registered official-test evaluation had not been performed when this task
started. This task completed that existing evaluation step, once per run, from
20:24:49.674126 to20:30:09.794298 UTC, after registered integrity checks passed.
It added new E3 evaluation directories and a new record only; no E1/E2 or training
output was overwritten. Analysis was isolated under `revision/evidence_freeze/`
to avoid rewriting the earlier frozen aggregate tables. No new training or new
experimental design was launched. Experiment C remains incomplete/blinded and
excluded, despite all current follow-up training now being complete.

## 2. Executive Scientific Summary

- C01: in-network boundary-feature transfer plus gate did not establish a Boundary F1 gain over the training-only boundary decoder under the prespecified three-dataset analysis. Without the surface term the pooled Δ is +0.0092 [−0.0010, +0.0194]; the dataset estimates vary and the pooled interval depends on the chosen estimator.
- C02: on ISIC2018 in the retrained set, the complete recipe scores below the decoder-only reference in the 10-run Boundary F1 comparison (−0.0321 [−0.0518, −0.0124]); fresh-seed Holm p=0.0314. The full recipe contains several changes, so the deficit is not attributable to distillation alone, and the earlier pipeline gives a different, uncertain result.
- C03–C04: exploratory external-teacher band KD had positive point estimates on all three datasets, none Holm-significant. The later pre-registered ACDC replication on 50 previously unused official test patients found +0.0178 [+0.0098, +0.0258] versus decoder-only, with 17 runs per arm. The lower bound falls below the 0.01 prespecified threshold, so the gain is **of unclear size**.
- C05–C06: a completed, registered-before-training but secondary exploratory follow-up reused those already examined patients. Full-image KD and compact-teacher band KD each had positive corrected contrasts versus decoder-only. Whether band restriction or the pretrained teacher's combined capacity/pretraining adds to the gain remains unresolved after Holm adjustment; do not claim equality, necessity, or a mechanism.
- C13: under the registered D4 augmentation, E3 band KD minus decoder-only Boundary F1 is −0.003708 [−0.011777, +0.004360], p=0.346144, ten runs per arm on the same 50 examined patients. A positive difference is not established; neither equivalence nor harm is established. This exploratory result narrows the unaugmented positive finding and cannot retroactively change its registered decision.
- C08–C12: estimator choice can change interval exclusions; a post hoc adjustment does not replace the frozen decision. Gate interventions change some labels, external literature uses different protocols, and older contaminated ISIC results and Experiment C test metrics are excluded.

## 3. Study Design and Experimental Inventory

The eight stage-1 cells encode U = compact U-Net, D = auxiliary boundary decoder, S = signed-distance surface loss, F = in-network feature transfer, G = gate. M0=U; M1=U+S; M2=D; M7=D+S; M3=D+F+G; M4=D+G+S; M5=D+F+S (gate scale 0); M6=D+F+G+S, the complete BDS-Lite recipe. M4 and M5 have different omissions; M5 is a training configuration and is not the same as setting a trained M6 gate to identity after training.

| Evidence group | Completed design and replication | Status / use |
|---|---|---|
| Stage 1 matched ablations | 8 cells × 3 datasets × seeds 1–3 = 72 cell-runs | VERIFIED for selected contrasts; inferential study informed by earlier work |
| Matched comparators | U+GSL, U+Boundary Loss, Attention U-Net, adapted LB-UNet × 3 datasets × 3 seeds = 36 runs | VERIFIED absolute values; controlled comparators, adapted LB-UNet |
| ISIC extension | U, D, full, GSL × 10 seeds = 40 runs (9 overlap with stage 1 and 3 with the comparator group); fresh seeds 4–10 | VERIFIED 30 contrasts; fresh-seed Holm family |
| External teacher stage2 | D+KD-band3 seeds/dataset; three teachers, seed100; references reused | VERIFIED registered arithmetic; exploratory; no pooled cross-dataset inference |
| ACDC stage 3 | D and D+KD-band × 17 fresh seeds 201–217 on 50 official patients | VERIFIED; pre-registered primary endpoint; ACDC selected post hoc |
| ACDC follow-up E1/E2 | full KD and compact-teacher band KD × 17 seeds, plus compact teacher seed 100; stage-3 D/band reused | VERIFIED; exploratory on same already examined patients |
| Other completed analyses | gate identity, sensitivity, duplicate sensitivity, old checkpoint re-evaluation, slice-stack volumes, resource and mechanistic probes | REPORTED unless individually identified; post hoc where applicable |
| E3 D4 augmentation | D and band KD, seeds201–210, 10 runs/arm; 20/20 completed and evaluated once | VERIFIED registered arithmetic; exploratory, same examined patients; §10.4 |
| Experiment C | 14 completed markers, one partial and55 unstarted/unavailable of70 slots | INCOMPLETE / EXCLUDED; no outcome is inspected or used |

The nonoverlapping core design comprises **237 training slots**, including four
teachers:72 matched ablation,36 comparators,28 additional ISIC,9 exploratory KD
students,3 external teachers,34 confirmation students,34 E1/E2 students,1 compact
teacher and20 E3 students. Reused references count once. This is a design count,
not the number of statistically independent realizations in all historical files.

The independently reconciled repository inventory contains **659 typed records**:
**526 nonblinded completed training records** (517 with150-epoch trajectory
evidence and nine EGE records supported by historical completion/epoch150
checkpoint filenames), plus **14 completed Experiment C markers excluded from
science**. Thus540 completed-record claims exist across repository history,
including aliases, re-evaluations and exact reproductions; **540 must not be
called independent experiments or models**. Independent uniqueness of that
all-history total is not established. The old ledger's379 tensor-distinct/~358
independent estimates remain REPORTED, not rehashed certification.

The remaining rows retain55 unstarted/unavailable C slots, one C partial,24
withdrawn designs,14 short diagnostics, two historical partial records, one
saved failed launch, six equivalence diagnostics, four caches and12 analysis
records. Full run-by-run methodology/dataset/arm/seed/epoch/partition/evidence
paths are in `revision/evidence_freeze/final_experiment_inventory.csv`; reconciliation
and family counts are in `final_inventory_notes.md`. Historical completion does
not validate a historical efficacy claim. Negative, failed, superseded and
blinded records are preserved in the register rather than silently discarded.

## 4. Datasets and Evaluation Protocols — Methods Reference

| Dataset and evaluation | Training | Validation | Test | Unit and source |
|---|---|---|---|---|
| ISIC2018 Task 1, skin-lesion binary segmentation | 2,075 RGB images | 519 development images | 1,000 official held-out images with masks; separate official validation set of 100 is not the training validation split | Image. `recovered/followup/splits/isic2018/{train,val}.json`; `revision/official_isic/test/`; `revision/OFFICIAL_ISIC_FINDINGS.md`. |
| ACDC cardiac cine MRI, 3 foreground classes | 70 patients, 1,350 slices | 10 patients, 186 slices | Stage 1/2: 20 internal patients, 366 slices. Stage 3/follow-up: official patients 101–150, 50 patients, 1,076 slices, separate from exploration. | Patient for recipe comparisons; sample=slice, foreground labels RV, myocardium, LV. `recovered/followup/splits/acdc/*.json`; `preregistration/acdc_confirmation.md`. |
| Synapse abdominal CT, 8 foreground class IDs | 14 cases, 1,738 slices | 4 cases, 473 slices | 12 held-out cases, 1,568 slices | Case for recipe comparisons. Original-label semantics require care; `revision/SYNAPSE_LABEL_SEMANTICS.md`; `recovered/followup/splits/synapse/*.json`. |

Counts are from saved manifests and registered official test records; ACDC/Synapse are patient/case-level partitions, not independent-slice splits. Processed inputs are 224×224; continuous image interpolation and nearest-neighbor label interpolation. ACDC is stored per-volume z-scored, and Synapse CT windowed to [0,1]; ISIC is RGB. Students consume stored normalization; the ImageNet teacher normalizes ISIC and Synapse, replicates single-channel medical images to three channels, and passes stored ACDC z-scores unchanged. The main, confirmation and E1/E2 runs used no online augmentation; E3 uses the registered D4 transforms of images, masks, boundary targets and cached probabilities together. Dataset IDs and exact image hashes were checked for ISIC split overlap; lesion/patient independence is not proved. Near-duplicate sensitivity is §11.

The official ACDC NIfTI preprocessing was reconstructed because the original third-party training-data converter was unavailable. On 1,536 development slices the masks matched bit-exactly and images differed by at most 0.014784 z-units; on 10 validation patients the maximum run-level differences were 0.00011 Boundary F1 and 0.00005 DSC. Impact on official patients cannot be directly measured. `revision/acdc_confirm/preprocessing_dev_gate.json`, `preprocessing_impact.json`, `preregistration/acdc_confirmation.md`.

Checkpoint selection uses the first maximum validation DSC: foreground-class Dice per slice/image, then mean across slices/images, with both-empty classes scored as1. This differs from final both-empty-excluded reporting and patient/case recipe contrasts. Alternative checkpoint rankings are unestablished. For example, Synapse D seed1 has2,506 both-empty rows among3,784; selection DSC0.8357645 versus0.5137190 after dropping those class rows (the latter is not a patient-level test score). Official test labels did not select checkpoints in stage3; later study/arm selection did use prior test results as disclosed. However, ACDC was selected for confirmation after exploratory results, and follow-up was designed after official ACDC test analysis; these are not independent prospective replications. Earlier ISIC validation numbers measured largely on training images are withdrawn (§15). Any exact dataset release/third-party converter version not supported by a local record is **PENDING**, not guessed.

## 5. Architectures and Model Variants

Compact backbone: four-level 2D U-Net, base width 32, with shared encoder and segmentation decoder (`src/bds_lite/models/unet.py`, `src/bds_lite/models/bds_lite.py`). D adds a training-only auxiliary boundary decoder, fed by the encoder, with boundary BCE+Dice and shared-encoder gradients. Its forward output is not used in the deployed segmentation output. For the ACDC confirmation D has one union-boundary output, gate present but scale 0, no SDF, no feature transfer and no external KD. Do not equate the same configuration across different runs, test sets or historical training pipelines.

In-network feature transfer: boundary features have 16 channels; a 1×1 projection maps 32-channel segmentation features to 16. `mean SmoothL1(P(F_s), stopgrad(F_b))`, weight 0.05, transfers toward detached boundary features. The gate reads the segmentation projection, not the boundary decoder: `F_out = F_s [1 + α(sigmoid(W_g P(F_s)+b_g) − 1/2)]`; α=0.25 in full runs, α=0 is identity. The interval of the modulation is [0.875,1.125] at α=0.25; this algebraic bound is not evidence of zero output difference. `revision/methods_verified.md`; `src/bds_lite/models/bds_lite.py`; `src/bds_lite/training/losses.py`.

External-teacher band KD adds a training-only loss to D, leaving its inference architecture unchanged relative to D by construction. The raw eager D/KD forward still computes the projection, scale-zero gate and an additional segmentation-head call; algebraic identity is not automatic graph pruning. A plain-U-Net execution graph requires an explicit pruned wrapper/export, and zero measured eager overhead is not established. The external teacher is a separately trained `segmentation_models_pytorch` 0.6.0 U-Net with ImageNet-initialized ResNet-34 encoder, trained once per dataset (seed 100). ACDC teacher: 24,436,804 parameters; training student: 2,197,493; ratio ≈11.1 relative to the full training model. Relative to the explicitly pruned ACDC U-Net graph (1,926,532 parameters), the teacher ratio is≈12.7; do not mix these denominators. The compact teacher uses the student's architecture and D recipe, no pretraining, seed 100, and no external-teacher quality gate. The original external teacher passed a validation DSC and Boundary F1 gate; the compact teacher's gate is not applicable. The teacher, capacity and pretraining are not separated by this design.

GSL, Boundary Loss, Attention U-Net and LB-UNet are matched-protocol comparators, not identical to all published implementations. LB-UNet is adapted to multiclass output and union-boundary supervision; original genetic key-point maps, augmentation, input size and training schedule were not reproduced. Upstream code/license is not redistributed. Legacy GSL Phase16 checkpoints are not interchangeable with later trajectories; `revision/GSL_LINEAGE_AUDIT.md` is historical and must not be used to deny the later matched comparator runs.

## 6. Training and Distillation Methodology

Shared unaugmented retrained recipe (E3 adds only registered D4 augmentation):150 epochs; AdamW, weight decay 10⁻⁵; cosine schedule, mixed precision, gradient-norm clip 1.0; no online augmentation. ISIC batch 8, learning rate 3×10⁻⁴; ACDC/Synapse batch 16, learning rate 10⁻⁴. CE weight 1 and foreground Dice loss weight 1; D boundary BCE+Dice weight 0.05; full in-network distillation weight 0.05; SDF surface weight 1; full gate scale 0.25. Source-of-record resolved configs and histories are included in the archive; parameters are configuration-specific, not inferred from a model label. `revision/ABLATION_PROTOCOL.md`, `revision/EXPANSION_PROTOCOL.md`, `configs/revision/`, saved `config.json`.

The auxiliary target is the union of per-foreground-class dilation XOR erosion using SciPy's default 2D connectivity-1 cross, one iteration; it is not the all-ones-square band used by external KD. Boundary BCE is evaluated over the full target map. SDF is EDT(outside)−EDT(inside), negative inside, per image/class normalized by its maximum absolute magnitude; empty classes are masked. Surface loss is the mean of foreground probability times normalized signed distance over valid image/class maps; it is signed and can be negative.

For external KD, teacher softmax `pᵗ` is cached float16 for training images (ACDC 1,350×4×224×224), probabilities are evaluated in fp32 and clamped to [10⁻⁶,1−10⁻⁶], no temperature. Let `B` be the union over foreground classes of dilation minus erosion of the teacher argmax masks using one 3×3 square operation, zero border. With student probabilities `pˢ`,

```text
L_KD = Σ_n,u B_n(u) [−Σ_k pᵗ_nk(u) log pˢ_nk(u)] / [Σ_n,u B_n(u) + 10⁻⁶]
L_total = L_CE + L_Dice + 0.05 L_boundary + λ_KD L_KD,  λ_KD = 1.
```

For D+KD-full, `scripts/kd_followup/launch.py::full_region` returns ones and replaces the band function in that process only. Thus the same teacher, cache and loss weight average soft CE across every pixel, including background and empty slices. D+KD-band(compact) changes the teacher/cache, hence its probabilities and predicted band geometry. That comparison entangles capacity, pretraining, training realization, calibration and band geometry; it is not an isolated causal capacity/pretraining contrast. The shared-input sibling decoder likewise has a distinct objective/representation; no additional external data does not prove redundant information. For representative seed 201, model, data, loss and trainer dictionaries are identical across the band/full/compact student configs; the KD block changes only `mode` or `cache_run`, respectively. Loss magnitude and gradient can nevertheless differ because the normalization region changes. `scripts/kd_mc/loss.py`; `scripts/kd_followup/launch.py`; `revision/followup_registration.md`.

## 7. Metrics and Statistical Methods

DSC=2|P∩G|/(|P|+|G|); IoU=|P∩G|/|P∪G|; both higher is better. Boundary F1 is the harmonic mean of matched-boundary precision and recall, higher is better. Per-class boundaries use dilation XOR erosion with SciPy's four-connected cross; matches use two cross-dilation steps (Manhattan neighborhood), not a Euclidean-radius disk. HD95/ASSD use erosion-defined inner surfaces and Euclidean transforms. The teacher KD band instead uses a3×3 square. Restricting the loss to that band does not restrict eventual prediction changes elsewhere through shared parameters. HD95 is the 95th percentile of bidirectional surface distances, ASSD their average; both lower is better, reported in pixels. For the primary analysis, both-empty foreground pairs are excluded; one-sided empty overlap/BF1 is near zero and its undefined distance is excluded. The all-case sensitivity assigns the image diagonal to one-sided empty distances; it is a penalty, not a measured distance. `src/bds_lite/evaluation/metrics.py`; `analysis/revision/family_statistics.py`.

**Two distinct summaries:** the absolute stage-1 tables average class-macro scores across *images/slices* within a run and then summarize seed means; their SD is sample SD across seeds. Recipe-level contrasts first average class-macro slices within patient/case and weight patients/cases equally. These differ even using the same predictions: for ACDC M0 BF1, slice-weighted 0.7828 versus patient-weighted 0.7858; for Synapse M0, 0.5059 versus case-weighted 0.5300. ISIC units are images so the two coincide. These are weighting differences, not silently interchangeable estimates.

Let `y_aru` be the score in arm `a`, run `r`, unit `u`, after class/slice aggregation, and `m_ar` its mean over units defined in all compared runs. The frozen two-arm estimator is:

```text
Δ = mean_r(m_Ar) − mean_r(m_Br)
SE² = s_A²/n_A + s_B²/n_B + Var_u(mean_r(y_Aru) − mean_r(y_Bru))/U
CI = Δ ± t_(0.975,ν) SE,
ν = (v_A+v_B)² / [v_A²/(n_A−1) + v_B²/(n_B−1)],  v_a=s_a²/n_a.
```

The df uses run terms only; the common-unit term is included in SE. The shared run-by-unit residual enters both terms, inflating the expected SE² under the stated variance model; this alone does not guarantee interval coverage under misspecification. Stage-1 factorial/transfer contrasts use a pooled within-cell run variance, `df=Σ_c(n_c−1)`, and the same per-unit contrast variance. Cross-dataset stage-1 summaries inverse-variance-weight the three estimates and use normal intervals; Q has two degrees of freedom and limited power. ACDC accounts for 84.3% of weight for the no-surface transfer BF1 row. The exploratory stage-2 pooled KD interval is not inferential evidence for transfer to new datasets.

Primary stage-2 BF1 tests use Holm across three datasets; ISIC extension uses fresh seeds 4–10, six pairwise contrasts and Holm within each metric; stage-3 uses one pre-registered primary BF1 contrast; follow-up C1–C4 use Holm across four BF1 tests, and secondary metrics are descriptive with Holm within metric. SESOI are engineering thresholds 0.01 for DSC/IoU/BF1, 2 px HD95, 0.5 px ASSD. A CI wholly within ±SESOI gives the registered no-relevant-effect label, distinct from an interval that merely crosses zero. Shared seed numbers do not automatically make the prespecified analysis paired; paired sensitivity is separately identified. Image-level Wilcoxon/bootstrap analyses condition on fixed models and are not independent training-run replication. `analysis/revision/expansion_statistics.py`; `analysis/revision/family_statistics.py`; `preregistration/acdc_confirmation.md`; `revision/followup_registration.md`.

## 8. Main Experimental Results

### 8.1 Three-run matched absolute results — VERIFIED

The following are slice/image-weighted per-run means ± sample SD across seeds 1–3, from `revision/tables/families/method_summary.csv`, independently recomputed from each run's case-level CSV (165 mean/SD metric cells checked; maximum mismatch below 10⁻¹⁰). ISIC=official 1,000 images, ACDC=internal 20 patients, Synapse=12 cases. Absolute values across these datasets and partitions are not a common performance scale. Here comparator C1 is U-Net+Boundary Loss, C3 is Attention U-Net, and C4 is adapted LB-UNet; those labels are unrelated to follow-up comparison IDs C1–C4 in §10.3.

| Dataset | Arm | DSC ↑ | IoU ↑ | BF1 ↑ | HD95 px ↓ | ASSD px ↓ |
|---|---|---|---|---|---|---|
| isic2018 | M0 | 0.8410 ± 0.0033 | 0.7501 ± 0.0044 | 0.3634 ± 0.0165 | 28.27 ± 1.32 | 9.65 ± 0.35 |
| isic2018 | M1 | 0.8350 ± 0.0010 | 0.7422 ± 0.0029 | 0.3320 ± 0.0343 | 28.85 ± 0.44 | 10.34 ± 0.04 |
| isic2018 | M2 | 0.8402 ± 0.0017 | 0.7513 ± 0.0021 | 0.3794 ± 0.0142 | 28.33 ± 0.92 | 9.79 ± 0.26 |
| isic2018 | M3 | 0.8396 ± 0.0044 | 0.7504 ± 0.0054 | 0.3746 ± 0.0218 | 27.88 ± 1.20 | 9.68 ± 0.26 |
| isic2018 | M4 | 0.8402 ± 0.0032 | 0.7508 ± 0.0049 | 0.3613 ± 0.0241 | 26.91 ± 0.44 | 9.71 ± 0.11 |
| isic2018 | M5 | 0.8379 ± 0.0039 | 0.7479 ± 0.0045 | 0.3597 ± 0.0009 | 27.11 ± 0.53 | 9.84 ± 0.24 |
| isic2018 | M6 | 0.8374 ± 0.0051 | 0.7466 ± 0.0081 | 0.3670 ± 0.0150 | 28.55 ± 2.18 | 9.86 ± 0.47 |
| isic2018 | GSL | 0.8419 ± 0.0037 | 0.7536 ± 0.0045 | 0.3838 ± 0.0161 | 27.55 ± 1.28 | 9.61 ± 0.26 |
| isic2018 | C1 | 0.8411 ± 0.0055 | 0.7510 ± 0.0092 | 0.3634 ± 0.0441 | 26.93 ± 2.29 | 9.63 ± 0.58 |
| isic2018 | C3 | 0.8417 ± 0.0023 | 0.7526 ± 0.0039 | 0.3746 ± 0.0251 | 28.03 ± 0.39 | 9.93 ± 0.14 |
| isic2018 | C4 | 0.8569 ± 0.0085 | 0.7722 ± 0.0120 | 0.3494 ± 0.0359 | 19.69 ± 0.82 | 8.32 ± 0.48 |
| acdc | M0 | 0.7420 ± 0.0080 | 0.6600 ± 0.0064 | 0.7828 ± 0.0013 | 7.40 ± 1.08 | 2.48 ± 0.36 |
| acdc | M1 | 0.7488 ± 0.0104 | 0.6675 ± 0.0113 | 0.7920 ± 0.0111 | 7.03 ± 0.22 | 2.50 ± 0.35 |
| acdc | M2 | 0.7583 ± 0.0080 | 0.6763 ± 0.0088 | 0.7993 ± 0.0080 | 8.88 ± 0.85 | 2.78 ± 0.24 |
| acdc | M3 | 0.7634 ± 0.0017 | 0.6829 ± 0.0035 | 0.8092 ± 0.0031 | 7.48 ± 0.56 | 2.45 ± 0.38 |
| acdc | M4 | 0.7545 ± 0.0019 | 0.6730 ± 0.0035 | 0.7936 ± 0.0048 | 8.20 ± 0.81 | 2.62 ± 0.46 |
| acdc | M5 | 0.7521 ± 0.0237 | 0.6725 ± 0.0246 | 0.7961 ± 0.0225 | 7.44 ± 0.18 | 2.54 ± 0.27 |
| acdc | M6 | 0.7584 ± 0.0132 | 0.6772 ± 0.0139 | 0.8006 ± 0.0097 | 8.67 ± 1.28 | 2.67 ± 0.32 |
| acdc | GSL | 0.7620 ± 0.0090 | 0.6766 ± 0.0097 | 0.8061 ± 0.0056 | 7.39 ± 1.63 | 2.28 ± 0.47 |
| acdc | C1 | 0.7620 ± 0.0102 | 0.6832 ± 0.0124 | 0.8073 ± 0.0109 | 5.85 ± 0.42 | 2.06 ± 0.18 |
| acdc | C3 | 0.7568 ± 0.0028 | 0.6737 ± 0.0051 | 0.7957 ± 0.0044 | 8.73 ± 1.08 | 2.52 ± 0.56 |
| acdc | C4 | 0.5098 ± 0.0255 | 0.4053 ± 0.0217 | 0.4721 ± 0.0241 | 14.39 ± 0.10 | 5.53 ± 0.14 |
| synapse | M0 | 0.4870 ± 0.0277 | 0.4388 ± 0.0281 | 0.5059 ± 0.0261 | 8.65 ± 0.48 | 3.05 ± 0.24 |
| synapse | M1 | 0.4945 ± 0.0152 | 0.4450 ± 0.0130 | 0.5122 ± 0.0165 | 8.75 ± 0.21 | 2.99 ± 0.10 |
| synapse | M2 | 0.5062 ± 0.0176 | 0.4557 ± 0.0163 | 0.5246 ± 0.0191 | 8.73 ± 0.89 | 3.11 ± 0.22 |
| synapse | M3 | 0.5285 ± 0.0292 | 0.4741 ± 0.0263 | 0.5488 ± 0.0329 | 8.85 ± 0.79 | 3.10 ± 0.45 |
| synapse | M4 | 0.4977 ± 0.0091 | 0.4477 ± 0.0097 | 0.5175 ± 0.0066 | 8.41 ± 0.33 | 3.07 ± 0.17 |
| synapse | M5 | 0.4809 ± 0.0147 | 0.4336 ± 0.0126 | 0.4987 ± 0.0151 | 8.60 ± 0.37 | 2.92 ± 0.12 |
| synapse | M6 | 0.4957 ± 0.0120 | 0.4464 ± 0.0112 | 0.5157 ± 0.0137 | 9.04 ± 0.57 | 3.14 ± 0.20 |
| synapse | GSL | 0.5325 ± 0.0262 | 0.4791 ± 0.0237 | 0.5555 ± 0.0273 | 8.31 ± 0.12 | 2.90 ± 0.17 |
| synapse | C1 | 0.5072 ± 0.0188 | 0.4553 ± 0.0174 | 0.5282 ± 0.0173 | 8.54 ± 0.38 | 3.03 ± 0.07 |
| synapse | C3 | 0.5217 ± 0.0095 | 0.4700 ± 0.0105 | 0.5382 ± 0.0098 | 8.32 ± 0.36 | 2.97 ± 0.14 |
| synapse | C4 | 0.2137 ± 0.0330 | 0.1683 ± 0.0263 | 0.1770 ± 0.0350 | 26.05 ± 5.74 | 13.20 ± 4.15 |

### 8.2 ISIC2018 extension: 10 runs per arm — VERIFIED

Means below weight each run's defined test images equally; SD is across seeds.
They were recalculated from `analysis/revision/expansion_statistics.py::isic_path`
and the preserved per-case outputs. Each run's defined-image set can differ
for distances; these standalone means therefore need not subtract to the
pairwise contrast below, which uses images defined in every run of the two
arms. In particular, the older `revision/manuscript/generated/tab_ten_seed.tex`
uses comparator-specific shared sets for its distance means, giving different
values; neither table is a substitute for documenting that eligibility rule.
The 30 pairwise contrasts × all/fresh estimators were independently recomputed
from the per-case files, maximum absolute difference 4.45×10⁻¹⁶. The familiar
3-seed values above differ because seeds 4–10 add runs.

| Arm | DSC | IoU | BF1 | HD95 px | ASSD px |
|---|---|---|---|---|---|
| unet | 0.8420 ± 0.0046 | 0.7528 ± 0.0065 | 0.3734 ± 0.0216 | 28.18 ± 1.66 | 9.74 ± 0.41 |
| decoder | 0.8436 ± 0.0042 | 0.7561 ± 0.0058 | 0.3940 ± 0.0186 | 27.65 ± 1.14 | 9.63 ± 0.32 |
| full | 0.8387 ± 0.0067 | 0.7484 ± 0.0099 | 0.3620 ± 0.0225 | 28.21 ± 2.30 | 9.93 ± 0.67 |
| gsl | 0.8399 ± 0.0040 | 0.7499 ± 0.0064 | 0.3633 ± 0.0257 | 27.58 ± 0.85 | 9.74 ± 0.27 |

All-seed Δ and 95% intervals below use ten runs per arm. Fresh Δ and Holm p use seeds 4–10, seven per arm, six contrasts within each metric. Distance contrasts reverse the sign; positive favors the first arm. “Confirmed” means the registered fresh-seed rule, not a new held-out patient set.

| Metric | First − second | All Δ [95% CI] | Fresh Δ | Fresh Holm p | Status |
|---|---|---|---|---|---|
| dsc | decoder − unet | +0.0016 [-0.0028, +0.0059] | +0.0026 | 1.0000 | not established |
| dsc | full − unet | -0.0033 [-0.0090, +0.0023] | -0.0031 | 1.0000 | not established |
| dsc | full − decoder | -0.0049 [-0.0104, +0.0006] | -0.0057 | 0.6068 | not established |
| dsc | gsl − unet | -0.0021 [-0.0064, +0.0022] | -0.0034 | 0.8953 | not established |
| dsc | gsl − full | +0.0012 [-0.0043, +0.0067] | -0.0003 | 1.0000 | not established |
| dsc | gsl − decoder | -0.0037 [-0.0078, +0.0004] | -0.0060 | 0.1608 | not established |
| iou | decoder − unet | +0.0033 [-0.0027, +0.0093] | +0.0042 | 0.8091 | not established |
| iou | full − unet | -0.0043 [-0.0125, +0.0038] | -0.0047 | 0.8091 | not established |
| iou | full − decoder | -0.0077 [-0.0156, +0.0003] | -0.0089 | 0.4876 | not established |
| iou | gsl − unet | -0.0028 [-0.0090, +0.0034] | -0.0055 | 0.6859 | not established |
| iou | gsl − full | +0.0015 [-0.0066, +0.0096] | -0.0008 | 0.8702 | not established |
| iou | gsl − decoder | -0.0061 [-0.0121, -0.0002] | -0.0097 | 0.0986 | supported by all seeds, not confirmed by fresh seeds |
| boundary_f1 | decoder − unet | +0.0207 [+0.0014, +0.0399] | +0.0226 | 0.2599 | supported by all seeds, not confirmed by fresh seeds |
| boundary_f1 | full − unet | -0.0114 [-0.0323, +0.0095] | -0.0178 | 0.4040 | not established |
| boundary_f1 | full − decoder | -0.0321 [-0.0518, -0.0124] | -0.0405 | 0.0314 | confirmed |
| boundary_f1 | gsl − unet | -0.0101 [-0.0326, +0.0124] | -0.0232 | 0.2885 | not established |
| boundary_f1 | gsl − full | +0.0013 [-0.0216, +0.0242] | -0.0054 | 0.6984 | not established |
| boundary_f1 | gsl − decoder | -0.0308 [-0.0522, -0.0093] | -0.0459 | 0.0137 | confirmed |
| hd95 | decoder − unet | +0.50 [-0.89, +1.88] | +0.79 | 1.0000 | not established |
| hd95 | full − unet | -0.07 [-1.98, +1.85] | +0.06 | 1.0000 | not established |
| hd95 | full − decoder | -0.56 [-2.33, +1.21] | -0.72 | 1.0000 | not established |
| hd95 | gsl − unet | +0.57 [-0.75, +1.89] | +0.50 | 1.0000 | not established |
| hd95 | gsl − full | +0.62 [-1.11, +2.36] | +0.42 | 1.0000 | not established |
| hd95 | gsl − decoder | +0.07 [-0.96, +1.09] | -0.30 | 1.0000 | not established |
| assd | decoder − unet | +0.09 [-0.28, +0.46] | +0.21 | 1.0000 | not established |
| assd | full − unet | -0.21 [-0.74, +0.33] | -0.19 | 1.0000 | not established |
| assd | full − decoder | -0.30 [-0.81, +0.21] | -0.41 | 1.0000 | not established |
| assd | gsl − unet | -0.01 [-0.37, +0.35] | -0.05 | 1.0000 | not established |
| assd | gsl − full | +0.19 [-0.31, +0.70] | +0.13 | 1.0000 | not established |
| assd | gsl − decoder | -0.11 [-0.42, +0.20] | -0.28 | 0.8584 | not established |

## 9. Ablation Results

The 3-dataset, 3-run factorial contrasts below use patient/case weighting and the prespecified pooled within-cell variance, not the slice-weighted absolute means in §8. Positive favors the named first addition. For each dataset, the two transfer contrasts were independently reproduced across all five metrics from primary files (30 cells, max error 3.56×10⁻¹⁵). All120 dataset-level and40 pooled stage-1 linear contrasts were independently recalculated from primary case files in the final4095-check verifier; additive component-model coefficients retain their separately stated model assumptions. Source `revision/tables/families/expansion_interaction.csv`; analysis `analysis/revision/expansion_statistics.py`; C01.

| Boundary F1 effect | ISIC | ACDC | Synapse | Pooled |
|---|---|---|---|---|
| interaction | +0.0322 [-0.0279, +0.0922] | -0.0118 [-0.0342, +0.0107] | -0.0180 [-0.0730, +0.0370] | -0.0079 [-0.0245, +0.0088] |
| surface without decoder | -0.0313 [-0.0927, +0.0300] | +0.0086 [-0.0119, +0.0291] | +0.0038 [-0.0491, +0.0567] | +0.0045 [-0.0084, +0.0174] |
| surface with decoder | +0.0008 [-0.0374, +0.0391] | -0.0032 [-0.0223, +0.0160] | -0.0142 [-0.0581, +0.0297] | -0.0039 [-0.0152, +0.0073] |
| decoder without surface | +0.0160 [-0.0194, +0.0514] | +0.0159 [-0.0014, +0.0332] | +0.0150 [-0.0456, +0.0756] | +0.0159 [+0.0053, +0.0265] |
| decoder with surface | +0.0482 [-0.0149, +0.1113] | +0.0042 [-0.0192, +0.0276] | -0.0030 [-0.0371, +0.0311] | +0.0058 [-0.0072, +0.0189] |
| distillation+gate without surface | -0.0048 [-0.0468, +0.0373] | +0.0106 [-0.0052, +0.0263] | +0.0215 [-0.0515, +0.0944] | +0.0092 [-0.0010, +0.0194] |
| distillation+gate with surface | -0.0132 [-0.0522, +0.0258] | +0.0034 [-0.0175, +0.0243] | +0.0059 [-0.0283, +0.0400] | +0.0011 [-0.0104, +0.0125] |
| distillation+gate interaction with surface | -0.0085 [-0.0561, +0.0391] | -0.0072 [-0.0283, +0.0140] | -0.0156 [-0.0855, +0.0544] | -0.0080 [-0.0238, +0.0079] |

The M7 decoder+surface cell is not included in the older `method_summary.csv`.
The following slice/image-weighted means ± sample SD were calculated directly
from its three per-case CSVs; the run-level interaction contrasts above use
patient/case weighting. Source paths:
`results/revision_runs/{dataset}_m7_decoder_surface_seed{1,2,3}/per_case_test.csv`,
preserved in the evidence archive. **VERIFIED**.

| Dataset | DSC ↑ | IoU ↑ | BF1 ↑ | HD95 px ↓ | ASSD px ↓ |
|---|---|---|---|---|---|
| ISIC2018 | 0.8412 ± 0.0027 | 0.7530 ± 0.0036 | 0.3802 ± 0.0189 | 28.59 ± 2.26 | 9.91 ± 0.41 |
| ACDC | 0.7568 ± 0.0095 | 0.6755 ± 0.0110 | 0.7979 ± 0.0084 | 7.36 ± 0.17 | 2.40 ± 0.32 |
| Synapse | 0.4878 ± 0.0151 | 0.4388 ± 0.0132 | 0.5090 ± 0.0163 | 8.96 ± 0.82 | 3.11 ± 0.33 |

The additive component-model coefficients are not isolated causal effects under interactions. Projection/readout and gate probes (§13) do not establish how training changed segmentation features. The complete-recipe ISIC deficit cannot be assigned to F because S and G are also present.

## 10. External-Teacher Experiments

### 10.1 Original three-dataset exploratory study — VERIFIED

Band-KD seeds 1–3; decoder-only references ISIC seeds 1–10 (unpaired, selected in the original analysis), ACDC/Synapse seeds 1–3. BF1 is the only Holm-adjusted family across datasets. All 15 displayed contrasts were independently recalculated from primary per-case files, with maximum absolute discrepancy 2.23×10⁻¹⁶; the three Holm values follow from the three raw BF1 p values. The later balanced ISIC 3-vs-3 presentation was selected after seeing results and must remain post hoc. Source `revision/kd_unified/contrasts.csv`; `analysis/kd3/unified_analysis.py`; C03.

| Dataset | Metric | KD-band mean ± SD | D mean ± SD | Δ [95% CI] | Holm p |
|---|---|---|---|---|---|
| isic2018 | dsc | 0.8440 ± 0.0118 | 0.8436 ± 0.0042 | +0.0005 [-0.0276, +0.0286] | — |
| isic2018 | iou | 0.7564 ± 0.0160 | 0.7561 ± 0.0058 | +0.0004 [-0.0376, +0.0384] | — |
| isic2018 | boundary_f1 | 0.4317 ± 0.0406 | 0.3940 ± 0.0186 | +0.0376 [-0.0565, +0.1317] | 0.4963 |
| isic2018 | hd95 | 29.50 ± 1.73 | 27.53 ± 1.13 | -1.97 [-5.88, +1.93] | — |
| isic2018 | assd | 9.65 ± 0.63 | 9.55 ± 0.31 | -0.10 [-1.57, +1.37] | — |
| acdc | dsc | 0.7767 ± 0.0063 | 0.7586 ± 0.0081 | +0.0181 [-0.0012, +0.0374] | — |
| acdc | iou | 0.6982 ± 0.0069 | 0.6763 ± 0.0090 | +0.0219 [+0.0007, +0.0432] | — |
| acdc | boundary_f1 | 0.8256 ± 0.0065 | 0.8018 ± 0.0075 | +0.0239 [+0.0040, +0.0438] | 0.0878 |
| acdc | hd95 | 6.65 ± 0.53 | 8.91 ± 0.75 | +2.27 [-0.00, +4.54] | — |
| acdc | assd | 2.11 ± 0.22 | 2.83 ± 0.27 | +0.72 [-0.08, +1.51] | — |
| synapse | dsc | 0.5296 ± 0.0245 | 0.5242 ± 0.0191 | +0.0054 [-0.0479, +0.0586] | — |
| synapse | iou | 0.4786 ± 0.0217 | 0.4725 ± 0.0179 | +0.0062 [-0.0417, +0.0540] | — |
| synapse | boundary_f1 | 0.5601 ± 0.0259 | 0.5450 ± 0.0203 | +0.0150 [-0.0423, +0.0724] | 0.5001 |
| synapse | hd95 | 7.73 ± 0.49 | 8.55 ± 0.89 | +0.82 [-1.16, +2.81] | — |
| synapse | assd | 2.69 ± 0.21 | 3.10 ± 0.23 | +0.42 [-0.18, +1.01] | — |

### 10.2 ACDC official-test confirmation — VERIFIED

50 official patients, 17 runs per arm (201–217), patient-level estimation. These are means of run means over patients, unlike §8.1 slice means. Primary result C04 is unchanged by follow-up. The pre-registered decision is success **of unclear size** because the lower BF1 bound +0.0098 is below the 0.01 SESOI. Two-sided p=0.00007842. The matched-seed sensitivity BF1 CI is [+0.0106,+0.0250], but does not replace the unpaired primary. Source `revision/acdc_confirm/results/results.json`; `results/acdc_confirm_eval/evaluation_record.json`; `scripts/acdc_confirm/analyze.py`; `preregistration/acdc_confirmation.md`.

| Metric | Band mean ± SD | D mean ± SD | Δ [95% CI] | p |
|---|---|---|---|---|
| dsc | 0.7485 ± 0.0070 | 0.7349 ± 0.0081 | +0.0136 [+0.0061, +0.0211] | 0.00087 |
| iou | 0.6763 ± 0.0079 | 0.6579 ± 0.0089 | +0.0183 [+0.0105, +0.0262] | 0.00004 |
| boundary_f1 | 0.7975 ± 0.0069 | 0.7797 ± 0.0083 | +0.0178 [+0.0098, +0.0258] | 0.00008 |
| hd95 | 7.67 ± 1.24 | 8.89 ± 2.15 | +1.22 [-0.09, +2.54] | 0.06701 |
| assd | 2.46 ± 0.32 | 2.84 ± 0.47 | +0.39 [+0.05, +0.72] | 0.02412 |

Primary run ranges overlap: band 0.7842–0.8078 versus D 0.7645–0.7949; the minimum band minus maximum D is −0.0107. All 17 band runs exceed the D arm mean, not every D run. Secondary per-structure BF1 differences (unadjusted and descriptive): RV +0.0263 [+0.0104,+0.0422], myocardium +0.0193 [+0.0116,+0.0269], LV +0.0109 [+0.0030,+0.0187]; no causal anatomy explanation follows. The primary estimator, 25 result rows including follow-up, and all20 follow-up Holm values (four contrasts across five metrics) were independently recomputed from the per-case CSVs and record hashes; max discrepancy 2.23×10⁻¹⁶.

### 10.3 Completed full-image and compact-teacher controls — VERIFIED, exploratory

Same 50 patients previously inspected, reused stage-3 D and band runs, 17 fresh control runs per arm, seeds 201–217. C1=band−full; C2=full−D; C3=compact-teacher band−D; C4=external-teacher band−compact-teacher band. BF1 Holm p is over C1–C4. Other metrics use separate Holm families and are descriptive. Nominal 95% intervals are not multiplicity-adjusted. Source `revision/followup/results/contrasts.csv`, `analysis/followup/analyze.py`, `revision/followup_registration.md`; C05–C06.

| ID | Metric | First mean ± SD | Second mean ± SD | Δ [95% CI] | Holm p |
|---|---|---|---|---|---|
| C1 | dsc | 0.7485 ± 0.0070 | 0.7449 ± 0.0076 | +0.0036 [-0.0038, +0.0109] | 0.4966 |
| C2 | dsc | 0.7449 ± 0.0076 | 0.7349 ± 0.0081 | +0.0100 [+0.0035, +0.0165] | 0.0149 |
| C3 | dsc | 0.7452 ± 0.0056 | 0.7349 ± 0.0081 | +0.0103 [+0.0030, +0.0175] | 0.0219 |
| C4 | dsc | 0.7485 ± 0.0070 | 0.7452 ± 0.0056 | +0.0033 [-0.0024, +0.0091] | 0.4966 |
| C1 | iou | 0.6763 ± 0.0079 | 0.6700 ± 0.0080 | +0.0062 [-0.0014, +0.0138] | 0.2085 |
| C2 | iou | 0.6700 ± 0.0080 | 0.6579 ± 0.0089 | +0.0121 [+0.0054, +0.0188] | 0.0035 |
| C3 | iou | 0.6712 ± 0.0062 | 0.6579 ± 0.0089 | +0.0133 [+0.0059, +0.0207] | 0.0035 |
| C4 | iou | 0.6763 ± 0.0079 | 0.6712 ± 0.0062 | +0.0050 [-0.0012, +0.0113] | 0.2085 |
| C1 | boundary_f1 | 0.7975 ± 0.0069 | 0.7903 ± 0.0086 | +0.0073 [-0.0005, +0.0150] | 0.0945 |
| C2 | boundary_f1 | 0.7903 ± 0.0086 | 0.7797 ± 0.0083 | +0.0106 [+0.0034, +0.0177] | 0.0196 |
| C3 | boundary_f1 | 0.7915 ± 0.0063 | 0.7797 ± 0.0083 | +0.0118 [+0.0037, +0.0200] | 0.0196 |
| C4 | boundary_f1 | 0.7975 ± 0.0069 | 0.7915 ± 0.0063 | +0.0060 [+0.0001, +0.0119] | 0.0945 |
| C1 | hd95 | 7.67 ± 1.24 | 9.03 ± 1.14 | +1.36 [+0.34, +2.38] | 0.0429 |
| C2 | hd95 | 9.03 ± 1.14 | 8.89 ± 2.15 | -0.14 [-1.42, +1.14] | 0.9475 |
| C3 | hd95 | 8.39 ± 1.55 | 8.89 ± 2.15 | +0.51 [-0.92, +1.93] | 0.9475 |
| C4 | hd95 | 7.67 ± 1.24 | 8.39 ± 1.55 | +0.72 [-0.34, +1.78] | 0.5260 |
| C1 | assd | 2.46 ± 0.32 | 2.83 ± 0.25 | +0.37 [+0.10, +0.64] | 0.0340 |
| C2 | assd | 2.83 ± 0.25 | 2.84 ± 0.47 | +0.02 [-0.29, +0.33] | 0.9118 |
| C3 | assd | 2.64 ± 0.39 | 2.84 ± 0.47 | +0.20 [-0.18, +0.59] | 0.7508 |
| C4 | assd | 2.46 ± 0.32 | 2.64 ± 0.39 | +0.18 [-0.14, +0.50] | 0.7508 |

C2 and C3 are positive in the registered BF1 family; C1 and C4 are unresolved. C4 has an unadjusted CI barely above zero but adjusted p=0.0945; do not confuse the nominal interval with an adjusted CI. The full-image control does not prove the band has no contribution. C1's secondary HD95 and ASSD intervals exclude zero after their metric-specific Holm tests, but these descriptive endpoints cannot establish a boundary-specific primary effect. One teacher per type, same test patients, loss-region normalization and unseparated pretraining/capacity limit inference.

The 68 confirmation/follow-up student evaluation files and prediction hashes matched their two evaluation records. Teacher-cache check maximum absolute errors were 0.0002492 for the external teacher and 0.0002500 for the compact teacher, both below 0.001; compact-teacher quality gate was not applicable. The analysis is not a randomized comparison of independently trained teachers. Follow-up local registration preceded its own training; the original external timestamp requirement was removed before training and its pending proofs were not externally verified.

### 10.4 E3: completed registered augmentation control — VERIFIED, exploratory

All20 runs (D_aug and KD_band_aug, seeds201–210) completed150 epochs, with
first-maximum validation selection and exact registered source/config/provenance
checks. Augmentation is one uniformly drawn D4 transform per training sample,
applied identically to image, mask, boundary and cached teacher probabilities;
there is no interpolation or test-time augmentation. These are transformed
cached probabilities, not new teacher predictions recomputed on augmented
images. Registration§3,§9.5 fixes the single primary BF1 contrast, no Holm
adjustment. Four secondary metrics are descriptive. E3a/E3b use10 augmented
versus17 earlier unaugmented runs and are descriptive, not independent replication.

Official patients101–150 were already examined. E3 is not confirmatory and
cannot change the original stage-3 decision. Original aggregates under
`revision/followup/results/` remain unchanged. Sources:
`results/followup_eval/evaluation_record_e3.json` (20 runs,40 artifact hashes),
`revision/evidence_freeze/final_e3_analysis.{json,csv}`, per_seed/per_class CSVs,
`analysis/revision/final_e3_analysis.py`. All15 metric contrasts matched an
independent implementation (195 scalar checks, max8.88e-16).

| ID | Metric | First mean±SD | Second mean±SD | Oriented Δ [nominal95% CI] | Two-sided p |
|---|---|---|---|---|---|
| E3 | dsc | 0.7893±0.0050 | 0.7964±0.0065 | -0.0071 [-0.0142,+0.0000] | 0.0512959 |
| E3 | iou | 0.7230±0.0052 | 0.7305±0.0071 | -0.0076 [-0.0149,-0.0002] | 0.0450246 |
| E3 | boundary_f1 | 0.8432±0.0060 | 0.8469±0.0074 | -0.0037 [-0.0118,+0.0044] | 0.346144 |
| E3 | hd95 | 5.07±0.33 | 5.24±0.78 | +0.17 [-0.56,+0.91] | 0.617409 |
| E3 | assd | 1.62±0.12 | 1.69±0.20 | +0.06 [-0.17,+0.29] | 0.568258 |
| E3a | dsc | 0.7964±0.0065 | 0.7349±0.0081 | +0.0615 [+0.0461,+0.0770] | 2.9732e-08 |
| E3a | iou | 0.7305±0.0071 | 0.6579±0.0089 | +0.0726 [+0.0568,+0.0884] | 2.40165e-09 |
| E3a | boundary_f1 | 0.8469±0.0074 | 0.7797±0.0083 | +0.0672 [+0.0500,+0.0844] | 6.72993e-08 |
| E3a | hd95 | 5.24±0.78 | 8.89±2.15 | +3.65 [+2.13,+5.18] | 5.71983e-05 |
| E3a | assd | 1.69±0.20 | 2.84±0.47 | +1.16 [+0.72,+1.60] | 1.26122e-05 |
| E3b | dsc | 0.7893±0.0050 | 0.7485±0.0070 | +0.0409 [+0.0279,+0.0538] | 1.01421e-06 |
| E3b | iou | 0.7230±0.0052 | 0.6763±0.0079 | +0.0467 [+0.0333,+0.0602] | 1.82539e-07 |
| E3b | boundary_f1 | 0.8432±0.0060 | 0.7975±0.0069 | +0.0457 [+0.0309,+0.0604] | 2.0759e-06 |
| E3b | hd95 | 5.07±0.33 | 7.67±1.24 | +2.60 [+1.56,+3.64] | 4.31516e-05 |
| E3b | assd | 1.62±0.12 | 2.46±0.32 | +0.84 [+0.48,+1.19] | 6.02511e-05 |

E3 first/second=band_aug/D_aug; E3a=D_aug/D; E3b=band_aug/band.
The primary BF1 point estimate is negative, but its interval crosses zero and
extends below−0.01: **positive benefit not established, harm not established,
equivalence not established**. Secondary IoU has a nominal negative interval
(p≈0.045); it does not replace the primary endpoint or support an adjusted
confirmatory harm claim. Both augmented arms have higher descriptive means
than their earlier unaugmented references; these comparisons do not isolate a
mechanism or constitute a registered augmentation×KD interaction test.

## 11. Sensitivity and Robustness Analyses

**Shared variance and non-monotonicity (REPORTED, post hoc).** The same run-by-unit residual enters both nominal variance components. With scaled shared-term subtraction and nonnegative component truncation, the pooled no-surface transfer BF1 intervals are: k=0 [−0.0010,+0.0194]; k=0.5 [+0.0011,+0.0181]; k=1 [+0.0009,+0.0182]; k=1.5 [−0.0010,+0.0195]. The endpoint is non-monotonic because truncation alters dataset variances and inverse-variance weights. At k=1 without truncation the pooled interval is [+0.0041,+0.0161], a different assumption. Stage-3 prespecified [+0.0098,+0.0258] becomes [+0.0104,+0.0253] at k=1; it does **not** change the prespecified size reading. `revision/restructure_posthoc/`; `revision/manuscript/generated/tab_s20_pooled.tex`; `revision/CLAIM_BOUNDARY_v2_NOTE_20261109.md`.

Alternative post hoc crossed random effects and hierarchical run×patient bootstrap BF1 intervals, respectively: stage 3 [+0.0105,+0.0251] and [+0.0101,+0.0263]; C1 [+0.0002,+0.0144] and [−0.0004,+0.0151]; C4 [+0.0008,+0.0112] and [−0.0001,+0.0123]. Their differences are not a reason to switch the frozen estimator or Holm decisions. `revision/followup/results/posthoc_reference_estimators.csv`.

**Near-duplicate ISIC sensitivity (REPORTED):** for original submitted checkpoints re-evaluated on the clean official1,000-image test set (not the withdrawn519-image contaminated evaluation), fixed-model BF1 Δ is +0.0307 [+0.0257,+0.0356] for all 1,000 and +0.0240 [+0.0185,+0.0293] after the strictest r>0.90 exclusion (833 images). In the separate retrained family it is +0.0036 [+0.0001,+0.0072] for all and +0.0025 [−0.0013,+0.0065] for 833. These fixed-model bootstrap intervals do not establish recipe-level generalization. `revision/tables/families/isic_duplicate_sensitivity.csv`.

**Gate and loss sensitivity (REPORTED):** ISIC gate scale 0/0.1/0.25/0.5/1.0 BF1 means were 0.3597/0.3686/0.3670/0.3576/0.3681 (three runs each; non-monotonic). Synapse rows use one run each and are not equivalent replication. Single-seed ISIC distillation-weight sensitivity at 0.01/0.05/0.2 gives BF1 0.3915/0.3498/0.3768; not a tuning recommendation or inferential ranking. `revision/tables/families/sensitivity_means.csv`.

**Gate identity (REPORTED, conditional fixed checkpoints):** replacing the trained gate by identity changed BF1 by −0.001410 [−0.001605,−0.001209] on 1,000 ISIC images, +0.000438 [+0.000230,+0.000658] on 20 ACDC patients, and −0.000094 [−0.000218,+0.000039] on 12 Synapse cases. Some rank-test p values are small despite small magnitude, so outputs are not exactly invariant. Conditional intervals are inside the 0.01 engineering threshold, not a general equivalence or hardware claim. `revision/GATE_IDENTITY_FINDINGS.md`.

## 12. Computational and Efficiency Results

ISIC 3-channel, 2-class profile, REPORTED from `revision/tables/resource_profile.csv` and `analysis/revision/profile_resources.py`; GPU values are median FP32 milliseconds per batch on the recorded RTX 5060 laptop, CPU is one-thread milliseconds. GFLOPs uses the released hook convention; another column reports a different counting convention.

| Graph | Training params | Deployed params | GFLOPs | GPU batch 1 ms | GPU batch 16 ms | CPU 1-thread ms |
|---|---|---|---|---|---|---|
| U-Net | 1,927,042 | 1,927,042 | 15.970 | 1.66 | 36.75 | 129.9 |
| BDS-Lite gate present | 2,198,003 | 1,928,114 | 16.079 | 1.79 | 42.26 | 134.8 |
| BDS-Lite gate collapsed | — | 1,927,042 | 15.970 | 1.66 | 36.84 | 131.1 |
| Attention U-Net | 1,948,773 | 1,948,773 | 16.281 | 1.86 | 41.05 | 136.0 |
| adapted LB-UNet | 55,874 | 55,874 | 0.144 | 3.56 | 3.91 | 6.5 |

The gate-retaining graph has 1,072 more deployed ISIC parameters and the measured latency differs from U-Net in this specific profile; parameter/FLOP counts alone do not imply runtime. ACDC D and band-KD student training graph 2,197,493 parameters; the external teacher has 24,436,804. Teacher training was 150 epochs, selected epoch 13, summed epoch time 0.243 h. Confirmation 17-run summed epoch time: band KD 7.625 h, D 7.522 h, difference 0.104 h (~1.4%). The original end-to-end teacher-cache duration is unrecorded. A later post hoc benchmark is available:median27.277531s over three passes of1,350 training images, batch16; it includes teacher forward/softmax/float16 copy/diagnostics but excludes loading, validation, saving and the20-image check (`results/acdc_confirm_eval_posthoc/teacher_cache_timing.json`). A separate KD-student inference-latency measurement is unavailable. D and KD share the same raw architecture; literal plain-U-Net graph equivalence requires explicit pruning as explained in§5. No hardware-general claim follows. `revision/kd_unified/costs.csv`; `revision/kd_multiclass/benchmark.json`; `revision/acdc_confirm/results/results.json`.

## 13. Qualitative Evidence and Mechanistic Probes

REPORTED, exploratory, not causal. Distilled vs undistilled teacher-head boundary AP at projected `q_s`: ISIC 0.123 vs 0.035, ACDC 0.191 vs 0.070, Synapse 0.104 vs 0.012; readout change on pre-projection `F_s`: +0.009 [−0.007,+0.025], +0.001 [−0.048,+0.050], −0.006 [−0.060,+0.047]. Gate modulation band/off-band 0.75, 1.08, 1.11. These observations are consistent with several explanations, including a learned projection reading existing information; they do not prove one. `revision/tables/families/boundary_transfer_probe.csv`, `gate_structure.csv`, `revision/manuscript/generated/tab_s_mechanism.tex`.

Single-seed diagnostic mean distillation-vs-segmentation gradient cosines were 0.030, 0.003, 0.009 for ISIC, ACDC and Synapse; fractions negative 0.455, 0.463, 0.483. The ACDC diagnostic was class-weighted and these are not causal conflict tests. `revision/tables/gradient_summary.csv`; `analysis/revision/gradient_residual_audit.py`.

Post hoc ACDC error-location summaries: among D errors, 55% RV, 80% myocardium, 67% LV lay within two pixels of the reference contour; stage-3 BF1 gains were +0.0263, +0.0193, +0.0109, respectively. Per-structure Spearman correlations for band KD were −0.06, −0.09, +0.15. Multiple descriptive correlations, shared reference, patient heterogeneity and potential regression to the mean preclude causal interpretation. `revision/followup/results/posthoc_error_location.csv`.

Qualitative contours are selected illustrations, not a test of relative frequency. Failure modes include small/absent structures and one-sided empty predictions; surface-distance complete-case selection can hide misses. `revision/figures/qualitative_manifest_v3.json`; `revision/SURFACE_ERROR_FINDINGS.md`.

## 14. Published Literature Comparisons

REPORTED from `revision/lit_numbers.csv` and source PDFs identified there; not independently rechecked for this freeze. Published ISIC lightweight results are not evaluated on the matched protocol; Synapse literature uses 18 training cases rather than 14, different 3D-volume aggregation, pretraining, resolution and augmentation; ACDC splits differ. Do not directly rank, compute significance across studies, equate HD labels to our pixel HD95, or treat adapted LB-UNet as the original method.

| Dataset / published model | Published DSC | Protocol caveat |
|---|---|---|
| ISIC2018 EGE-UNet | 0.8946 | different split, 256 px, augmentation |
| ISIC2018 LB-UNet | 0.8954 | published binary formulation, different protocol |
| ACDC TransUNet / Swin-Unet | 0.8971 / 0.9000 | published splits, volume-level values |
| Synapse TransUNet / Swin-Unet | 0.7748 / 0.7913 | 18 training cases; volume-level DSC; reported HD 31.69 / 21.55 |

Published HD units/implementation are disputed: source captions and code use different descriptions; keep original source labels and do not assert millimeters for our resized 2D metrics. The original Swin-Unet ECCV-workshop chapter (doi10.1007/978-3-031-25066-8_9) was unavailable in the earlier audit; this is distinct from LB-UNet's MICCAI2024 source. Exact published values require primary-document verification before new use; `revision/RESTRUCTURE_PHASE1_AUDIT.md`, `revision/validation/published_metric_protocol_audit.json`.

## 15. Negative, Inconclusive, and Superseded Findings

- **SUPERSEDED/EXCLUDED:** earlier ISIC validation values 0.5597 vs 0.5520 are contaminated by substantial train-image reuse; they cannot be manuscript efficacy evidence. Official 1,000-image test evaluations replace them. `revision/CLAIM_BOUNDARY_v2.md` §6; `revision/OFFICIAL_ISIC_FINDINGS.md`.
- **REPORTED:** earlier lost-pipeline full−D BF1 pooled +0.0067 [−0.0082,+0.0216], ISIC alone +0.0255 [−0.0782,+0.1293]. It does not replicate the retrained ISIC deficit; original trainer is missing, so compare arms within each pipeline. `revision/tables/families/historical_april.csv`.
- **VERIFIED:** full-vs-D ISIC deficit and null fresh-seed U-vs-full contrasts are in §8.2. GSL-vs-full BF1 is not established in the matched ten-run analysis; no claim that BDS-Lite exceeds GSL.
- **REPORTED:** in a fixed-checkpoint official ISIC F−U comparison, ASSD was worse by raw +0.2457 px, Holm p=0.00634; BF1 +0.003608 [0.000068,0.007207], Holm p=0.308742. Image-level rank tests and bootstrap mean intervals have different estimands and need not agree. `revision/OFFICIAL_ISIC_FINDINGS.md`.
- **REPORTED:** stacked 2D predictions assessed as volumes gave full−U DSC +0.0142 [0.0027,0.0299], Holm p=0.0577 on ACDC and +0.0138 [0.0067,0.0205], Holm p=0.0205 on Synapse; surface-distance findings were mixed. These are image-stack analyses of 2D models in index units, not evidence for 3D training, physical measurements or cross-dataset transfer. `revision/VOLUME_FINDINGS.md`.
- **EXCLUDED:** Experiment C completed only a subset, with test outputs under a blinding/amendment boundary. No C test result is inspected or used. E3 is complete and its inconclusive primary result is included in§10.4; the earlier incomplete-status wording is superseded. No superseded or unavailable checkpoint is silently substituted. Historical GSL long-schedule credibility phase E did not produce a valid completed result.

## 16. Claim Boundaries and Interpretation Rules

`revision/CLAIM_BOUNDARY_v4.md` is the final governing boundary. C01–C14 apply to manuscript statements. In particular: name the appropriate decoder-only reference; never equate a full-recipe deficit to an in-network mechanism; keep stage-2 exploratory, stage-3 ACDC-selected confirmatory, and E1/E2/E3 same-patient exploratory; do not claim the band is unnecessary or causal, nor that teacher size/pretraining separately explains results. Preserve the primary +0.0098 lower bound and “of unclear size”. Do not turn a nonsignificant result into equivalence. Never claim an external-KD versus in-network effect difference without a valid direct design. Secondary or post hoc results do not change the primary. Forbidden scope includes unsupported novelty, cross-dataset, 3D, clinical-use and hardware generalizations.

## 17. Evidence Traceability Matrix

| Claim ID | Claim / comparator | Supporting evidence and analysis | Verification | Scope / limitations |
|---|---|---|---|---|
| C01 | F+G vs D, ±S | expansion_interaction.csv; expansion_statistics.py; seeds1–3; final_statistics_audit.json | VERIFIED dataset and pooled arithmetic | three datasets; pooled fixed-effect; CI crosses zero |
| C02 | full vs D on ISIC | expansion_contrasts.csv; expansion_statistics.py; seeds 1–10, fresh 4–10 | VERIFIED | complete recipe, retrained set only |
| C03 | band KD vs D, three datasets | revision/kd_unified/contrasts.csv; analysis/kd3/unified_analysis.py | VERIFIED | 3 KD seeds, ISIC 10 references; Holm 3 datasets |
| C04 | band KD vs D, official ACDC | revision/acdc_confirm/results/results.json; scripts/acdc_confirm/analyze.py; seeds 201–217 | VERIFIED | 50 same-centre patients, post hoc dataset selection; effect size unclear |
| C05 | full / compact band vs D | revision/followup/results/contrasts.csv; analysis/followup/analyze.py; seeds 201–217 | VERIFIED | exploratory, previously examined patients, one teacher each |
| C06 | band vs full / teacher type | same follow-up artifacts; C1 and C4 | VERIFIED | Holm p=0.0945; no equivalence |
| C07 | secondary / class metrics | revision/acdc_confirm/results/contrasts.csv and per_class.csv | VERIFIED overall and nine confirmation class contrasts; other class summaries as labeled | descriptive, distances 2D pixels |
| C08 | shared variance sensitivity | revision/restructure_posthoc/; posthoc_reference_estimators.csv | Ten crossed-model/bootstrap rows independently VERIFIED; other saved sensitivity summaries REPORTED | post hoc; no decision replacement |
| C09 | mechanistic probes | revision/tables/families/boundary_transfer_probe.csv; gradient_summary.csv | REPORTED | observational, some single-seed |
| C10 | withdrawn / excluded experiments | revision/OFFICIAL_ISIC_FINDINGS.md; revision/CONFIRMATION_PROTOCOL.md | REPORTED / EXCLUDED | do not inspect Experiment C tests |
| C11 | model resource graph | revision/tables/resource_profile.csv; kd_multiclass/benchmark.json | REPORTED | specific configuration and measurement hardware |
| C12 | published / adapted references | revision/lit_numbers.csv; COMPARATOR_PROTOCOL.md | REPORTED | different protocols and adapted implementations |
| C13 | E3 band_aug−D_aug | final_e3_analysis.csv/json; frozen followup registration§3,§9.5; new evaluation record | VERIFIED15 contrasts/195 checks | exploratory;10runs/arm;one D4 regime; benefit/harm/equivalence unresolved |
| C14 | Inference architecture/cost | bds_lite.py:168–176; profile_resources.py::GateCollapsed; resource_profile.csv | VERIFIED static execution scope; saved timing arithmetic | D/KD raw graphs equal each other; plain-U-Net graph requires pruning; no fresh latency measurement |

## 18. Known Limitations and Unresolved Questions

1. A full **training/inference reproducibility** freeze is not established: source datasets, some predictions, model checkpoints and float16 caches are ignored/local or licensed and not independently backed up by the Git commit; original pipeline training code is lost. Archived CSVs support reanalysis of selected results, not reconstruction of predictions or training. Hash records cannot recover missing bytes.
2. AGENTS.md now points to the author-authorized final v4 boundary. Historical v2/v3 wording remains preserved; it must not be read as current evidence status.
3. Official ACDC preprocessing differs by reconstruction within observed tolerance; effect on official patients is unmeasurable. The 50 patients are from the same dataset/centre, and selection followed exploratory findings. No claim of cross-dataset transfer.
4. Follow-up reuses examined patients and reference runs. The local registration has no verified external timestamp before training; full-vs-band loss scale differs. One teacher per architecture; pretraining and capacity are entangled. Unadjusted confidence intervals and Holm p values answer different questions.
5. ISIC lesion-identity independence is not established even though image IDs and bytes were checked. ACDC spacing/frames and Synapse anatomy naming constrain interpretation. Published comparator protocols and source distances differ.
6. E3 is complete and included, but remains exploratory on examined patients and one D4 regime. It does not establish that augmentation eliminates a KD effect: no registered interaction test was performed, and the difference between significant and nonsignificant results is not itself a significant difference. Experiment C remains excluded and requires an amended analysis before any outcome is read.
7. Several analyses and diagnostics in §11–15 are REPORTED rather than independently re-executed. Archived paths must not be confused with historical, missing or replaced checkpoint provenance. The missing registered E3 evaluation was completed once during this task; existing confirmation/E1/E2 test outputs were not re-evaluated. Raw AP-feature reconstruction, all historical predictions and published literature were not independently rerun/recovered.

## 19. Manuscript Revision Reference

| Section | Evidence / claim IDs to use |
|---|---|
| Abstract | C01, C04–C06,C13: primary ACDC BF1 exact CI and unclear size; distinguish exploratory follow-up, unresolved band/teacher contributions and E3's unestablished augmented contrast. |
| Introduction and contributions | C01, C03–C06: frame tested comparisons and qualifications; no unsupported novelty or universal benefit. |
| Related Work | C12 and §14: literature protocols, teacher distinction, adapted comparator caveats. |
| Methods | §4–7: model graphs, gate, losses, morphology, teacher/caches, preprocessing, estimands and estimator. |
| Experimental Setup | §3–7 and §10: partitions, seeds, selection, pre-registration timeline, multiplicity and scope. |
| Results | §8–10; C01–C07: retain matched absolute means, 10-run negative finding, stage-2/3 and follow-up with exact CIs. |
| Discussion | §11, §13, §15: estimator sensitivity, possible mechanisms as hypotheses, negative and contradictory evidence. |
| Limitations | §18: held-out reconstruction, selected dataset, reused patients, teacher confounding, missing artifacts, statistical design. |
| Conclusion | C01–C06,C13: studied 2D settings, magnitude uncertainty, no band-specific or teacher-specific causal conclusion. |
| Supplementary Material | §9, §11–15, §20: ablations, per-class and secondary results, full sensitivity, computational and provenance records. |

## 20. Final verification and preservation record

### Independent verification scope

- `analysis/revision/verify_final_evidence.py`:4,095 scalar checks, zero
  mismatches, max absolute discrepancy3.752553823233029e-14 (tolerance1e-10),
 681 unique primary inputs hashed. Includes25 ACDC/E1/E2 contrasts,20 Holm values,
 nine confirmation class contrasts,60 ISIC estimates/60 Holm values,120 stage-1
 dataset/40 pooled contrasts,15 original KD contrasts, two three-dataset display
 families,165 absolute mean/SD cells, ten post hoc reference rows,68 histories,
 218 evaluation artifact hashes,15 cost histories,544 raw KD-probe records and
 36 gradient-summary rows.
- `analysis/revision/final_e3_audit.py`:20 completed150-epoch histories, exact
  source/config/provenance/artifact checks and selected checkpoint epoch/config.
  Its dated20:25 observation that evaluation was missing is superseded by the
  later completed evaluation record, not silently rewritten.
- `analysis/revision/final_e3_analysis.py`:15 contrasts,195 independent scalar
  comparisons, max8.88e-16;40 evaluation artifacts and20 checkpoints hash-match.
  E3 BF1 is a single registered exploratory primary test, no Holm adjustment.
- `analysis/revision/verify_final_predictions.py`:all20 E3 outputs pass ID,
  shape/dtype/label checks; four uniformly spaced slices/run give240 class rows
  and1,200 independent metric comparisons from saved predictions/reference masks.
  Maximum metric difference0;96 undefined-distance agreements each for HD95/ASSD.
  This is a sample, not full prediction-to-metric reconstruction.
- E3 D4 implementation:all eight codes agree exactly with independent NumPy
  transforms for four fields (32 array checks); mask/teacher alignment and square
  KD-band commutation pass. Training only is augmented. Epoch-time sums:
  D_aug3.751343h; band_aug3.812927h, excluding cooling/wall time.
- Supplemental duplicate/gate/volume/resource/mechanistic summaries were checked
  against saved tables. Original latency, feature extraction, every historical
  prediction and external publication were not rerun. These verification limits
  travel with the evidence labels; existing diagnostic output is not proof of a
  causal explanation.

### Layered statistical preservation

The original `primary-evidence-20261110.tar.gz` and its1,108-file SHA manifest
remain byte-identical. `final-supplement-20261110.tar.gz` adds structured inputs
and source snapshots missing from that layer, including the previously ignored
E3 case-level outputs and frozen source/configuration/provenance histories.
The companion content manifest gives a SHA-256 for each original relative path.
`final_preservation.json` records exact archive byte sizes, hashes and combined
member counts. Both layers are verified member-by-member and compared with local
source bytes; large dependencies are explicitly outside the archive.

**Verified final preservation:** base1,108 members plus final supplement1,724
members = **2,832 distinct structured evidence/source files**. The supplement
is27,892,992 bytes; base48,720,203 bytes. Both archive membership and every
stored digest match the corresponding selected live bytes. The archived-only
statistical checker reproduces40 ACDC metric contrasts and20 Holm values without
large dependencies (max numerical discrepancy9.29e-14):

```bash
.venv/bin/python analysis/revision/verify_archived_statistics.py
```

The full provenance/hash/prediction audit scripts also verify local large files
and therefore require the dependencies listed separately. Their failure in a
CSV-only checkout would not mean the archived statistical arithmetic is missing.
The two archival layers deliberately do not claim to reproduce all historical
training, feature extraction or original publication values.

| Artifact | SHA-256 |
|---|---|
| `primary-evidence-20261110.tar.gz` | `5fc0d861d8fc1e02446a4dbf97914ed63e5e0b7b6a2231f3c9ad0632903c7f5e` |
| `final-supplement-20261110.tar.gz` | `570f26fd916afa9b241729d71c07253e7e90f742b483c20c4aa280a7b657906e` |
| `primary-evidence-20261110.sha256` | `21a9d06818f6f63462a94c7088090ed314293193f32c8c288a236018868970ae` |
| `final-supplement-20261110.sha256` | `b1e80161c99423d30fc48f613511d6518738b42ba1fcb54abace68d66323a64e` |

Verification command, with no model execution or experiment-result writes:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python analysis/revision/preserve_final_evidence.py
```

A new empty scratch directory may receive both archives in sequence; verify
against both content manifests. Do not extract over live experiments. Source and
registered-analysis files are included, so statistical reanalysis need not depend
on the earlier provisional document's prose. The final reference/boundary and
archive checksums are also recorded in `final_checksums.csv` and in the tagged
Git tree; the checksums file excludes itself and its mirrored completion-report manifest to avoid circular hashing.

### Large dependencies and exact reproduction limits

`final_dependency_manifest.csv` independently hashes10,067 existing local files,
18,360,963,870 logical bytes (17.100GiB), and explicitly identifies nine missing
historical Phase16 GSL checkpoints. Included identity checks cover9,538 processed
files, four official ISIC archives,12 raw Synapse H5 files, four teacher caches,
508 nonblinded checkpoint paths and the pinned pretrained encoder. These files
are **local-only, not preserved in the statistical archives or independently
backed up by this tag**. Raw ACDC NIfTI/source families outside the enumerated
paths are not exhaustively inventoried. Prediction-map hashes are in evaluation
records/receipts, but their bytes are not all archived. See the exact manifest
and `final_dependency_notes.md`, not an inferred whole-repository backup claim.

The pinned encoder's SHA matches its registered object; retrieval and lawful data
access still depend on the providers. The local Python overlay imports packages
from external `/home/jc/bds-followup/` environments and ACDC tooling. A complete
portable lockfile, external source retrieval, recovered original trainer/converter,
missing historical checkpoints and authorized independent storage are not
established. No package installation/environment change, remote upload or Zenodo
deposition is claimed. The tag finalizes the documented statistical scope only.

### Git scope and ongoing manuscript work

Version tag: **`bds-lite-evidence-v1.0.0`**, annotated locally, no push. Its target
identifies the final freeze commit; do not infer a self-referential commit hash
from this file. Governance is adopted in an author-authorized `agents:` commit;
freeze artifacts use a separate scoped documentation commit. The author's
repository-local identity is Jingchen Chen, `jingchen@localhost` (authorized
fallback; no matching historical repository email was found). Global Git settings
are unchanged.

The dirty working tree contained earlier staged manuscript/letter corrections.
This phase neither rewrites nor sweeps those edits into the freeze commits.
The tagged authoritative reference and source snapshots separate science from
presentation. The current manuscript still says E3 was incomplete at an earlier
editorial cutoff and needs manuscript-only revision; this is a known presentation
lag, not missing final evidence. Future writers must use this reference/v4 and
report remaining limitations. Full release readiness and author approval remain
separate from readiness to revise the manuscript from validated evidence.

## 21. Historical and ancillary experimental register

This table covers every inventoried family, not only the currently favorable or
primary arms. Detailed configurations, seeds, partitions, paths and completion
support are in the companion CSV. Historical records may be aliases or exact
reproductions. Completion alone does not validate their outcomes or override
superseded protocols. Historical or checkpoint-only efficacy values are not
promoted to new inferential claims by inclusion in this register.

| Family | Records | Datasets / seeds | Recorded status / scientific treatment |
|---|---:|---|---|
| ACDC confirmation | 34 | acdc; 201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217 | completed_150_epoch_record:34 |
| ACDC index-grid region vs paper-GSL | 6 | acdc (recovered 80-case dev data); 1,2,3 | completed_150_epoch_record:6 |
| ACDC phase-reduction control | 6 | acdc (recovered 80-case dev data); 1,2,3 | completed_150_epoch_record:6 |
| ACDC three-index context control | 3 | acdc (recovered 80-case dev data); 1,2,3 | completed_150_epoch_record:3 |
| E1 long-schedule GSL | 1 | isic2018; 1 | partial_record:1 |
| EGE-UNet compact baseline | 9 | acdc,isic2018,synapse; 1,2,3 | ledger_complete_epoch150_checkpoint_present:9 |
| ISIC D4 augmentation baseline | 3 | isic2018; 1,2,3 | completed_150_epoch_record:3 |
| ISIC U/S/B/F factorial | 12 | isic2018; 1,2,3 | completed_150_epoch_record:12 |
| ISIC boundary-specificity control | 3 | isic2018; 1,2,3 | completed_150_epoch_record:3 |
| ISIC pathway controls | 6 | isic2018; 1,2,3 | completed_150_epoch_record:6 |
| Phase 16 matched GSL | 9 | acdc,isic2018,synapse; 1,2,3 | completed_150_epoch_record:9 |
| Stage 1 expansion | 40 | acdc,isic2018,synapse; 1,10,2,3,4,5,6,7,8,9 | completed_150_epoch_record:40 |
| Stage 2 Experiment C | 70 | acdc,isic2018,synapse; 10,11,12,13,14,15,16,17,4,5,6,7,8,9 | completed_blinded_excluded:14; not_started_or_unavailable:55; started_incomplete_blinded_excluded:1 |
| WP2/WP3 Tier B | 23 | acdc,isic2018,synapse; 1,2,3 | completed_150_epoch_record:23 |
| WP2/WP4 queue A | 46 | acdc,isic2018,synapse; 1,2,3 | completed_150_epoch_record:46 |
| WP4 queue B | 27 | acdc,isic2018,synapse; 1,2,3 | completed_150_epoch_record:27 |
| boundary readout probes | 1 | ;  | completed_output_present:1 |
| cohort shift | 1 | ;  | completed_output_present:1 |
| distance-missingness sensitivity | 1 | ;  | completed_output_present:1 |
| error location | 1 | ;  | completed_output_present:1 |
| exploratory external KD students | 9 | acdc,isic2018,synapse; 1,2,3 | completed_150_epoch_record:9 |
| exploratory external teacher | 3 | acdc,isic2018,synapse; 100 | completed_150_epoch_record:3 |
| exploratory teacher cache | 3 | acdc,isic2018,synapse;  | cache_complete:3 |
| failed launch | 1 | isic2018; 1 | failed_saved_error_record:1 |
| family O reproducibility grid | 117 | acdc,isic2018,synapse; 1,2,3 | completed_150_epoch_record:117 |
| follow-up E1/E2 students | 34 | acdc; 201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217 | completed_150_epoch_record:34 |
| follow-up E3 augmentation | 20 | acdc; 201,202,203,204,205,206,207,208,209,210 | completed_150_epoch_record:20 |
| follow-up compact teacher/cache | 2 | acdc; 100 | cache_complete:1; completed_150_epoch_record:1 |
| gate identity intervention | 1 | ;  | completed_output_present:1 |
| gate structure probes | 1 | ;  | completed_output_present:1 |
| gradient probes | 1 | ;  | completed_output_present:1 |
| near-duplicate sensitivity | 1 | ;  | completed_output_present:1 |
| p1 gate pilot (single seed) | 15 | acdc,isic2018,synapse; 2026 | completed_150_epoch_record:15 |
| p1b gate-free multi-seed | 36 | acdc,isic2018,synapse; 1,2,3 | completed_150_epoch_record:36 |
| p2 recent-loss baselines (GSL, Boundary DoU) | 19 | acdc,isic2018,synapse; 1,2,3 | completed_150_epoch_record:18; partial_record:1 |
| p2b BDS-Lite(no gate)+GSL | 9 | acdc,isic2018,synapse; 1,2,3 | completed_150_epoch_record:9 |
| reference estimators | 1 | ;  | completed_output_present:1 |
| reproduction gate | 12 | ; 1 | short_diagnostic:12 |
| resource profiles | 1 | ;  | completed_output_present:1 |
| shared-variance sensitivity | 1 | ;  | completed_output_present:1 |
| single-seed exploration | 37 | acdc,isic2018,synapse; 2026,2027,2028 | completed_150_epoch_record:37 |
| smoke | 2 | ; 7 | short_diagnostic:2 |
| stacked-volume evaluation | 1 | ;  | completed_output_present:1 |
| trainer equivalence diagnostic | 6 | acdc; 1 | equivalence_diagnostic:6 |
| withdrawn registration | 24 | acdc,isic2018,synapse; 1,2,3 | withdrawn_never_run:24 |

Historical efficacy numerical files are retained with their chronology and
exclusion labels. Examples requiring preservation include the lost-pipeline
full−D contrast, original-checkpoint clean-test re-evaluations, failed/aborted
long-schedule GSL, class-weighted ACDC diagnostics, single-seed weight sweeps,
index-grid phase/context controls and adapted EGE/LB-UNet records. These are not
interchangeable with current matched primary contrasts. Nine missing historical
GSL checkpoints do not authorize substituting later weights; later valid GSL
retraining remains its own documented family.

## 22. Final contributions, revision plan and audit issue dispositions

The defensible contribution is a controlled assessment of additions to auxiliary
boundary supervision, with explicit negative/inconclusive evidence and replicated
training runs: (1) in-network transfer remains unestablished under the registered
analysis; (2) external-teacher band KD has a positive unaugmented ACDC confirmation
of unclear size; (3) exploratory full-image/compact-teacher controls leave band
and teacher-property attribution unresolved; (4) completed D4-augmentation E3
does not establish the positive KD difference in that regime. None establishes
a universal recipe, causal mechanism or transfer beyond the studied conditions.

Recommended main evidence displays:

1. Study-design table with chronology, data partitions, seed counts, endpoints,
   registered versus post hoc choices, and reused patients/references.
2. Stage-1 dataset-specific and pooled transfer contrasts, plus the retrained
   ISIC full−D negative result and its fresh-seed Holm test.
3. Original exploratory KD, unaugmented ACDC confirmation, E1/E2 controls and E3
   in clearly separated panels. Keep E3's null primary beside the positive
   unaugmented result. Do not turn their significance difference into interaction.
4. Resource table with measured graph/hardware, raw versus explicitly pruned
   execution, teacher cost and the post hoc cache-benchmark scope.
5. Supplementary complete secondary/class tables, estimator/duplicate/distance
   sensitivities, error-location/representation probes, historical register and
   source/dependency manifests. Preserve adverse and unestablished rows.

Section guidance in§19 is supplemented as follows: Methods must state exact
metric morphology, selection aggregation and inference pruning; Results must
include the E3 primary and distinguish its descriptive IoU; Discussion must keep
teacher configurations, cohort/preprocessing and loss-region scale entangled;
Limitations must retain reported-only historical verification and portable
reproduction gaps. Abstract/Conclusion should state that the positive KD result
is conditional on the studied unaugmented setting and that augmentation E3 did
not establish the positive difference. No manuscript or reviewer-letter edit is
part of this freeze phase.

Prior audit issues were rechecked. Historical ISIC overlap is verified and
withdrawn; band influence and informational redundancy are corrected as
interpretations, not new measurements; validation-selection mismatch and the
duplicated residual remain disclosed; C4's nominal/adjusted difference is correct;
E3's missing planned evaluation was completed without overwriting old results;
stale cache timing and plain-U-Net execution claims are narrowed; historical
counts are reconciled without claiming unique realizations. All material unresolved
provenance and reproducibility issues remain explicit.

## 23. Independent final audit and handoff

Independent review receipts are `final_content_review.md` and
`final_evidence_review.md`; their verification scope is retained. No material
numerical/scientific inconsistency remains within the independently checked
principal evidence. The full final-statistics pass gives4,095 checks and the
isolated E3 analysis195; the prediction sample adds1,200 metric comparisons.
All reported-only limitations and incomplete historical/provenance dependencies
remain explicit rather than being promoted to verified outcomes.

Required repository checks passed: Ruff format (438 files), Ruff lint, full
CPU-only pytest147 passed/one optional skip. No test was weakened or source
experiment repaired to make a check pass. Archive-only reanalysis and both
member-by-member preservation checks pass. `final_consistency_audit.json`
records preservation of the starting manuscript/letters and experimental sources,
claim/table consistency, and checksum verification scope. The existing manuscript
and response PDFs were not rebuilt in this freeze phase, as presentation edits
are deferred to the next task.

Tooling defects in new audit helpers were resolved without altering scientific
assertions: inventory empty-path/CSV schema handling; new-verifier explanatory
string lint; initial preservation-helper unused import/string length. Exact
checks, continuations, outputs and committed paths are in PLAN.md's phase status.
The prior Git identity blocker was resolved with the author's repository-local
name and explicitly authorized placeholder email; no global configuration or
remote operation is performed.

**Ready for manuscript-only revision from this scientific reference and v4.**
That readiness does not mean every historical result is independently verified,
mechanisms are resolved, release/deposition is complete, or full portable
training/inference reproduction is available. Fable and Opus must retain the
negative E3 result and scope qualifications, preserve registered decisions, and
version any future scientific correction instead of silently changing evidence.
