# PROVISIONAL DRAFT — BDS-Lite completed-evidence inventory — 2026-11-10

## 1. Provisional metadata and scope

**Provisional status (author instruction, 2026-11-10).** This is a draft inventory, not a finalized evidence freeze or release. It remains provisional until all experiments finish, including E3. The existing archive and checksum files are preserved without regeneration; no freeze commit, freeze tag or push is performed by the scientific-correction task. Scientific wording corrections are documented separately in `revision/SCIENTIFIC_AUDIT_CORRECTIONS_20261110.md`.

**Evidence cutoff: 2026-11-10 12:46:21 UTC.** Source commit `5b2ff7a52edf906f16fdea52d7c6a9b4bad282d8` (branch `revision-followup`). The cutoff is an evidence boundary, not the later commit time. The author explicitly authorized `revision/CLAIM_BOUNDARY_v3.md` for this snapshot. V2 is preserved unchanged; `AGENTS.md` still names v2 and requires a human edit to make future agent sessions automatically see this precedence. Give future sessions the author's later authorization until then.

This document is a provisional traceability reference for **selected completed evidence at the stated cutoff**, subject to the classifications below; it has no finalized freeze authority. It covers results, analysis choices, configurations, and claim limits, not a ban on prose or presentation changes. No experiment was started for this audit. E3 augmentation was running/resuming at the cutoff, and all E3 results, including subsequently completed runs, are excluded. Experiment C is incomplete and its test metrics remain blind/excluded. Later results are not incorporated here; any eventual finalization requires author authorization after all experiments finish.

**Verification labels.** VERIFIED = independently recalculated from primary case outputs or independently checked against recorded hashes, with scope stated; REPORTED = faithfully transcribed from the cited source but not independently recalculated here; SUPERSEDED = displaced by corrected analysis; EXCLUDED = unsuitable or outside the cutoff; PENDING = unresolved. Validating the 68-run ACDC patient-level inputs and all 165 three-seed descriptive metric cells does not make every earlier result or causal explanation verified. All 95% intervals are two-sided unless stated. Δ means first arm minus second; for HD95 and ASSD signs of contrasts are reversed so positive favors the first arm. Absolute distance scores are not sign reversed.

**Reproducibility qualification.** The Git snapshot preserves 1,108 selected files including primary case-level CSVs, recorded run configurations/provenance and locked split manifests in a deterministic 48,720,203-byte archive. Model checkpoints, cached teacher probabilities, original public/restricted datasets, and prediction maps are not all committed or copied into an independent durable store. They remain local/externally obtained dependencies, some licensed; therefore this is a **provisional statistical-evidence archive, not a complete independent training-and-inference reproduction package**. Hashes and missing dependencies are listed in §20. Do not infer that a Git tag alone preserves the images, checkpoints, or preprocessing environment.

## 2. Executive Scientific Summary

- C01: in-network boundary-feature transfer plus gate did not establish a Boundary F1 gain over the training-only boundary decoder under the prespecified three-dataset analysis. Without the surface term the pooled Δ is +0.0092 [−0.0010, +0.0194]; the dataset estimates vary and the pooled interval depends on the chosen estimator.
- C02: on ISIC2018 in the retrained set, the complete recipe scores below the decoder-only reference in the 10-run Boundary F1 comparison (−0.0321 [−0.0518, −0.0124]); fresh-seed Holm p=0.0314. The full recipe contains several changes, so the deficit is not attributable to distillation alone, and the earlier pipeline gives a different, uncertain result.
- C03–C04: exploratory external-teacher band KD had positive point estimates on all three datasets, none Holm-significant. The later pre-registered ACDC replication on 50 previously unused official test patients found +0.0178 [+0.0098, +0.0258] versus decoder-only, with 17 runs per arm. The lower bound falls below the 0.01 prespecified threshold, so the gain is **of unclear size**.
- C05–C06: a completed, registered-before-training but secondary exploratory follow-up reused those already examined patients. Full-image KD and compact-teacher band KD each had positive corrected contrasts versus decoder-only. Whether band restriction or the pretrained teacher's combined capacity/pretraining adds to the gain remains unresolved after Holm adjustment; do not claim equality, necessity, or a mechanism.
- C08–C12: estimator choice can change interval exclusions; a post hoc adjustment does not replace the frozen decision. Gate interventions change some labels, external literature uses different protocols, and older contaminated ISIC results and Experiment C test metrics are excluded.

## 3. Study Design and Experimental Inventory

The eight stage-1 cells encode U = compact U-Net, D = auxiliary boundary decoder, S = signed-distance surface loss, F = in-network feature transfer, G = gate. M0=U; M1=U+S; M2=D; M7=D+S; M3=D+F+G; M4=D+G+S; M5=D+F+S (gate scale 0); M6=D+F+G+S, the complete BDS-Lite recipe. M4 and M5 have different omissions; M5 is a training configuration and is not the same as setting a trained M6 gate to identity after training.

| Evidence group | Completed design and replication | Status / use |
|---|---|---|
| Stage 1 matched ablations | 8 cells × 3 datasets × seeds 1–3 = 72 cell-runs | VERIFIED for selected contrasts; inferential study informed by earlier work |
| Matched comparators | U+GSL, U+Boundary Loss, Attention U-Net, adapted LB-UNet × 3 datasets × 3 seeds = 36 runs | VERIFIED absolute values; controlled comparators, adapted LB-UNet |
| ISIC extension | U, D, full, GSL × 10 seeds = 40 runs (9 overlap with stage 1 and 3 with the comparator group); fresh seeds 4–10 | VERIFIED 30 contrasts; fresh-seed Holm family |
| External teacher stage 2 | D+KD-band 3 seeds on each dataset; 3 independent teachers, seed 100; references overlap stage 1/extension | REPORTED; exploratory; no pooled cross-dataset inference |
| ACDC stage 3 | D and D+KD-band × 17 fresh seeds 201–217 on 50 official patients | VERIFIED; pre-registered primary endpoint; ACDC selected post hoc |
| ACDC follow-up E1/E2 | full KD and compact-teacher band KD × 17 seeds, plus compact teacher seed 100; stage-3 D/band reused | VERIFIED; exploratory on same already examined patients |
| Other completed analyses | gate identity, sensitivity, duplicate sensitivity, old checkpoint re-evaluation, slice-stack volumes, resource and mechanistic probes | REPORTED unless individually identified; post hoc where applicable |
| E3 augmentation; Experiment C | E3 ongoing; C incomplete, test boundary remains in force | PENDING / EXCLUDED; no preliminary results |

Counts above are intended arm-run slots, not unique models: references are deliberately reused, and the 40 ISIC runs include 9 stage-1 and 3 comparator runs. Excluding this overlap, the core inventory comprises **217 distinct intended training runs**, including four teachers: 72 ablation, 36 comparator, 28 additional ISIC, 9 exploratory KD students, 3 external teachers, 34 confirmation students, 34 follow-up students and 1 compact teacher. Historical, diagnostic and sensitivity runs are additional; the count is a design inventory, not an independently enumerated all-repository total. A separate run ledger is in `revision/audit/project_audit_20261107/experiment_ledger.csv` (where present), source configs and saved case files in the snapshot manifest.

## 4. Datasets and Evaluation Protocols — Methods Reference

| Dataset and evaluation | Training | Validation | Test | Unit and source |
|---|---|---|---|---|
| ISIC2018 Task 1, skin-lesion binary segmentation | 2,075 RGB images | 519 development images | 1,000 official held-out images with masks; separate official validation set of 100 is not the training validation split | Image. `recovered/followup/splits/isic2018/{train,val}.json`; `revision/official_isic/test/`; `revision/OFFICIAL_ISIC_FINDINGS.md`. |
| ACDC cardiac cine MRI, 3 foreground classes | 70 patients, 1,350 slices | 10 patients, 186 slices | Stage 1/2: 20 internal patients, 366 slices. Stage 3/follow-up: official patients 101–150, 50 patients, 1,076 slices, separate from exploration. | Patient for recipe comparisons; sample=slice, foreground labels RV, myocardium, LV. `recovered/followup/splits/acdc/*.json`; `preregistration/acdc_confirmation.md`. |
| Synapse abdominal CT, 8 foreground class IDs | 14 cases, 1,738 slices | 4 cases, 473 slices | 12 held-out cases, 1,568 slices | Case for recipe comparisons. Original-label semantics require care; `revision/SYNAPSE_LABEL_SEMANTICS.md`; `recovered/followup/splits/synapse/*.json`. |

Counts are from saved manifests and registered official test records; ACDC/Synapse are patient/case-level partitions, not independent-slice splits. Processed inputs are 224×224; continuous image interpolation and nearest-neighbor label interpolation. ACDC is stored per-volume z-scored, and Synapse CT windowed to [0,1]; ISIC is RGB. Students consume stored normalization; the ImageNet teacher normalizes ISIC and Synapse, replicates single-channel medical images to three channels, and passes stored ACDC z-scores unchanged. The main runs used no online augmentation. Dataset IDs and exact image hashes were checked for ISIC split overlap; lesion/patient independence is not proved. Near-duplicate sensitivity is §11.

The official ACDC NIfTI preprocessing was reconstructed because the original third-party training-data converter was unavailable. On 1,536 development slices the masks matched bit-exactly and images differed by at most 0.014784 z-units; on 10 validation patients the maximum run-level differences were 0.00011 Boundary F1 and 0.00005 DSC. Impact on official patients cannot be directly measured. `revision/acdc_confirm/preprocessing_dev_gate.json`, `preprocessing_impact.json`, `preregistration/acdc_confirmation.md`.

Checkpoints were selected by maximum validation DSC, first occurrence of a tie; official test labels did not select checkpoints in stage 3. However, ACDC was selected for confirmation after exploratory results, and follow-up was designed after official ACDC test analysis; these are not independent prospective replications. Earlier ISIC validation numbers measured largely on training images are withdrawn (§15). Any exact dataset release/third-party converter version not supported by a local record is **PENDING**, not guessed.

## 5. Architectures and Model Variants

Compact backbone: four-level 2D U-Net, base width 32, with shared encoder and segmentation decoder (`src/bds_lite/models/unet.py`, `src/bds_lite/models/bds_lite.py`). D adds a training-only auxiliary boundary decoder, fed by the encoder, with boundary BCE+Dice and shared-encoder gradients. Its forward output is not used in the deployed segmentation output. For the ACDC confirmation D has one union-boundary output, gate present but scale 0, no SDF, no feature transfer and no external KD. Do not equate the same configuration across different runs, test sets or historical training pipelines.

In-network feature transfer: boundary features have 16 channels; a 1×1 projection maps 32-channel segmentation features to 16. `mean SmoothL1(P(F_s), stopgrad(F_b))`, weight 0.05, transfers toward detached boundary features. The gate reads the segmentation projection, not the boundary decoder: `F_out = F_s [1 + α(sigmoid(W_g P(F_s)+b_g) − 1/2)]`; α=0.25 in full runs, α=0 is identity. The interval of the modulation is [0.875,1.125] at α=0.25; this algebraic bound is not evidence of zero output difference. `revision/methods_verified.md`; `src/bds_lite/models/bds_lite.py`; `src/bds_lite/training/losses.py`.

External-teacher band KD adds a training-only loss to D, leaving its inference graph unchanged by construction. The external teacher is a separately trained `segmentation_models_pytorch` 0.6.0 U-Net with ImageNet-initialized ResNet-34 encoder, trained once per dataset (seed 100). ACDC teacher: 24,436,804 parameters; training student: 2,197,493; ratio ≈11.1. The compact teacher uses the student's architecture and D recipe, no pretraining, seed 100, and no external-teacher quality gate. The original external teacher passed a validation DSC and Boundary F1 gate; the compact teacher's gate is not applicable. The teacher, capacity and pretraining are not separated by this design.

GSL, Boundary Loss, Attention U-Net and LB-UNet are matched-protocol comparators, not identical to all published implementations. LB-UNet is adapted to multiclass output and union-boundary supervision; original genetic key-point maps, augmentation, input size and training schedule were not reproduced. Upstream code/license is not redistributed. Legacy GSL Phase16 checkpoints are not interchangeable with later trajectories; `revision/GSL_LINEAGE_AUDIT.md` is historical and must not be used to deny the later matched comparator runs.

## 6. Training and Distillation Methodology

Shared retrained recipe: 150 epochs; AdamW, weight decay 10⁻⁵; cosine schedule, mixed precision, gradient-norm clip 1.0; no online augmentation. ISIC batch 8, learning rate 3×10⁻⁴; ACDC/Synapse batch 16, learning rate 10⁻⁴. CE weight 1 and foreground Dice loss weight 1; D boundary BCE+Dice weight 0.05; full in-network distillation weight 0.05; SDF surface weight 1; full gate scale 0.25. Source-of-record resolved configs and histories are included in the archive; parameters are configuration-specific, not inferred from a model label. `revision/ABLATION_PROTOCOL.md`, `revision/EXPANSION_PROTOCOL.md`, `configs/revision/`, saved `config.json`.

The auxiliary target is the union of per-foreground-class dilation XOR erosion using SciPy's default 2D connectivity-1 cross, one iteration; it is not the all-ones-square band used by external KD. Boundary BCE is evaluated over the full target map. SDF is EDT(outside)−EDT(inside), negative inside, per image/class normalized by its maximum absolute magnitude; empty classes are masked. Surface loss is the mean of foreground probability times normalized signed distance over valid image/class maps; it is signed and can be negative.

For external KD, teacher softmax `pᵗ` is cached float16 for training images (ACDC 1,350×4×224×224), probabilities are evaluated in fp32 and clamped to [10⁻⁶,1−10⁻⁶], no temperature. Let `B` be the union over foreground classes of dilation minus erosion of the teacher argmax masks using one 3×3 square operation, zero border. With student probabilities `pˢ`,

```text
L_KD = Σ_n,u B_n(u) [−Σ_k pᵗ_nk(u) log pˢ_nk(u)] / [Σ_n,u B_n(u) + 10⁻⁶]
L_total = L_CE + L_Dice + 0.05 L_boundary + λ_KD L_KD,  λ_KD = 1.
```

For D+KD-full, `scripts/kd_followup/launch.py::full_region` returns ones and replaces the band function in that process only. Thus the same teacher, cache and loss weight average soft CE across every pixel, including background and empty slices. D+KD-band(compact) changes the cache to the compact teacher. For representative seed 201, model, data, loss and trainer dictionaries are identical across the band/full/compact student configs; the KD block changes only `mode` or `cache_run`, respectively. Loss magnitude and gradient can nevertheless differ because the normalization region changes. `scripts/kd_mc/loss.py`; `scripts/kd_followup/launch.py`; `revision/followup_registration.md`.

## 7. Metrics and Statistical Methods

DSC=2|P∩G|/(|P|+|G|); IoU=|P∩G|/|P∪G|; both higher is better. Boundary F1 is the harmonic mean of matched-boundary precision and recall at two resized-grid pixels, higher is better. HD95 is the 95th percentile of bidirectional surface distances, ASSD their average; both lower is better, reported in pixels. For the primary analysis, both-empty foreground pairs are excluded; one-sided empty overlap/BF1 is near zero and its undefined distance is excluded. The all-case sensitivity assigns the image diagonal to one-sided empty distances; it is a penalty, not a measured distance. `src/bds_lite/evaluation/metrics.py`; `analysis/revision/family_statistics.py`.

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

The 3-dataset, 3-run factorial contrasts below use patient/case weighting and the prespecified pooled within-cell variance, not the slice-weighted absolute means in §8. Positive favors the named first addition. For each dataset, the two transfer contrasts were independently reproduced across all five metrics from primary files (30 cells, max error 3.56×10⁻¹⁵). The pooling and remaining contrasts are REPORTED. Source `revision/tables/families/expansion_interaction.csv`; analysis `analysis/revision/expansion_statistics.py`; C01.

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

Primary run ranges overlap: band 0.7842–0.8078 versus D 0.7645–0.7949; the minimum band minus maximum D is −0.0107. All 17 band runs exceed the D arm mean, not every D run. Secondary per-structure BF1 differences (unadjusted and descriptive): RV +0.0263 [+0.0104,+0.0422], myocardium +0.0193 [+0.0116,+0.0269], LV +0.0109 [+0.0030,+0.0187]; no causal anatomy explanation follows. The primary estimator, 25 result rows including follow-up, and all four-metric Holm families were independently recomputed from the per-case CSVs and record hashes; max discrepancy 2.23×10⁻¹⁶.

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

## 11. Sensitivity and Robustness Analyses

**Shared variance and non-monotonicity (REPORTED, post hoc).** The same run-by-unit residual enters both nominal variance components. With scaled shared-term subtraction and nonnegative component truncation, the pooled no-surface transfer BF1 intervals are: k=0 [−0.0010,+0.0194]; k=0.5 [+0.0011,+0.0181]; k=1 [+0.0009,+0.0182]; k=1.5 [−0.0010,+0.0195]. The endpoint is non-monotonic because truncation alters dataset variances and inverse-variance weights. At k=1 without truncation the pooled interval is [+0.0041,+0.0161], a different assumption. Stage-3 prespecified [+0.0098,+0.0258] becomes [+0.0104,+0.0253] at k=1; it does **not** change the prespecified size reading. `revision/restructure_posthoc/`; `revision/manuscript/generated/tab_s20_pooled.tex`; `revision/CLAIM_BOUNDARY_v2_NOTE_20261109.md`.

Alternative post hoc crossed random effects and hierarchical run×patient bootstrap BF1 intervals, respectively: stage 3 [+0.0105,+0.0251] and [+0.0101,+0.0263]; C1 [+0.0002,+0.0144] and [−0.0004,+0.0151]; C4 [+0.0008,+0.0112] and [−0.0001,+0.0123]. Their differences are not a reason to switch the frozen estimator or Holm decisions. `revision/followup/results/posthoc_reference_estimators.csv`.

**Near-duplicate ISIC sensitivity (REPORTED):** for the submitted historical F−U comparison, fixed-model BF1 Δ is +0.0307 [+0.0257,+0.0356] for all 1,000 and +0.0240 [+0.0185,+0.0293] after the strictest r>0.90 exclusion (833 images). In the separate retrained family it is +0.0036 [+0.0001,+0.0072] for all and +0.0025 [−0.0013,+0.0065] for 833. These fixed-model bootstrap intervals do not establish recipe-level generalization. `revision/tables/families/isic_duplicate_sensitivity.csv`.

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

The gate-retaining graph has 1,072 more deployed ISIC parameters and the measured latency differs from U-Net in this specific profile; parameter/FLOP counts alone do not imply runtime. ACDC D and band-KD student training graph 2,197,493 parameters; the external teacher has 24,436,804. Teacher training was 150 epochs, selected epoch 13, summed epoch time 0.243 h. Confirmation 17-run summed epoch time: band KD 7.625 h, D 7.522 h, difference 0.104 h (~1.4%). Teacher cache generation time and a separate KD inference-latency measurement are not available. No hardware-general claim follows. `revision/kd_unified/costs.csv`; `revision/kd_multiclass/benchmark.json`; `revision/acdc_confirm/results/results.json`.

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

Published HD units/implementation are disputed: source captions and code use different descriptions; keep original source labels and do not assert millimeters for our resized 2D metrics. The original ECCV-workshop chapter was unavailable in the earlier audit, so its exact published LB-UNet/Swin-related numbers should be checked against the primary document before submission; `revision/RESTRUCTURE_PHASE1_AUDIT.md`, `revision/validation/published_metric_protocol_audit.json`.

## 15. Negative, Inconclusive, and Superseded Findings

- **SUPERSEDED/EXCLUDED:** earlier ISIC validation values 0.5597 vs 0.5520 are contaminated by substantial train-image reuse; they cannot be manuscript efficacy evidence. Official 1,000-image test evaluations replace them. `revision/CLAIM_BOUNDARY_v2.md` §6; `revision/OFFICIAL_ISIC_FINDINGS.md`.
- **REPORTED:** earlier lost-pipeline full−D BF1 pooled +0.0067 [−0.0082,+0.0216], ISIC alone +0.0255 [−0.0782,+0.1293]. It does not replicate the retrained ISIC deficit; original trainer is missing, so compare arms within each pipeline. `revision/tables/families/historical_april.csv`.
- **VERIFIED:** full-vs-D ISIC deficit and null fresh-seed U-vs-full contrasts are in §8.2. GSL-vs-full BF1 is not established in the matched ten-run analysis; no claim that BDS-Lite exceeds GSL.
- **REPORTED:** in a fixed-checkpoint official ISIC F−U comparison, ASSD was worse by raw +0.2457 px, Holm p=0.00634; BF1 +0.003608 [0.000068,0.007207], Holm p=0.308742. Image-level rank tests and bootstrap mean intervals have different estimands and need not agree. `revision/OFFICIAL_ISIC_FINDINGS.md`.
- **REPORTED:** stacked 2D predictions assessed as volumes gave full−U DSC +0.0142 [0.0027,0.0299], Holm p=0.0577 on ACDC and +0.0138 [0.0067,0.0205], Holm p=0.0205 on Synapse; surface-distance findings were mixed. These are image-stack analyses of 2D models in index units, not evidence for 3D training, physical measurements or cross-dataset transfer. `revision/VOLUME_FINDINGS.md`.
- **EXCLUDED:** Experiment C completed only a subset, with test outputs under a blinding/amendment boundary. No C test result is inspected or used. E3 incomplete at cutoff, results pending. No superseded or unavailable checkpoint is silently substituted. Historical GSL long-schedule credibility phase E did not produce a valid completed result.

## 16. Claim Boundaries and Interpretation Rules

`revision/CLAIM_BOUNDARY_v3.md` is the author-authorized claim boundary for this snapshot. C01–C12 apply to manuscript statements. In particular: name the appropriate decoder-only reference; never equate a full-recipe deficit to an in-network mechanism; keep stage-2 exploratory, stage-3 ACDC-selected confirmatory, and E1/E2 same-patient exploratory; do not claim the band is unnecessary or causal, nor that teacher size/pretraining separately explains results. Preserve the primary +0.0098 lower bound and “of unclear size”. Do not turn a nonsignificant result into equivalence. Never claim an external-KD versus in-network effect difference without a valid direct design. Secondary or post hoc results do not change the primary. Forbidden scope includes unsupported novelty, cross-dataset, 3D, clinical-use and hardware generalizations.

## 17. Evidence Traceability Matrix

| Claim ID | Claim / comparator | Supporting evidence and analysis | Verification | Scope / limitations |
|---|---|---|---|---|
| C01 | F+G vs D, ±S | expansion_interaction.csv; expansion_statistics.py; seeds 1–3 | VERIFIED dataset cells; pooled REPORTED | three datasets; pooled fixed-effect; CI crosses zero |
| C02 | full vs D on ISIC | expansion_contrasts.csv; expansion_statistics.py; seeds 1–10, fresh 4–10 | VERIFIED | complete recipe, retrained set only |
| C03 | band KD vs D, three datasets | revision/kd_unified/contrasts.csv; analysis/kd3/unified_analysis.py | VERIFIED | 3 KD seeds, ISIC 10 references; Holm 3 datasets |
| C04 | band KD vs D, official ACDC | revision/acdc_confirm/results/results.json; scripts/acdc_confirm/analyze.py; seeds 201–217 | VERIFIED | 50 same-centre patients, post hoc dataset selection; effect size unclear |
| C05 | full / compact band vs D | revision/followup/results/contrasts.csv; analysis/followup/analyze.py; seeds 201–217 | VERIFIED | exploratory, previously examined patients, one teacher each |
| C06 | band vs full / teacher type | same follow-up artifacts; C1 and C4 | VERIFIED | Holm p=0.0945; no equivalence |
| C07 | secondary / class metrics | revision/acdc_confirm/results/contrasts.csv and per_class.csv | VERIFIED overall; per-class REPORTED | descriptive, distances 2D pixels |
| C08 | shared variance sensitivity | revision/restructure_posthoc/; posthoc_reference_estimators.csv | REPORTED | post hoc; no decision replacement |
| C09 | mechanistic probes | revision/tables/families/boundary_transfer_probe.csv; gradient_summary.csv | REPORTED | observational, some single-seed |
| C10 | withdrawn / excluded experiments | revision/OFFICIAL_ISIC_FINDINGS.md; revision/CONFIRMATION_PROTOCOL.md | REPORTED / EXCLUDED | do not inspect Experiment C tests |
| C11 | model resource graph | revision/tables/resource_profile.csv; kd_multiclass/benchmark.json | REPORTED | specific configuration and measurement hardware |
| C12 | published / adapted references | revision/lit_numbers.csv; COMPARATOR_PROTOCOL.md | REPORTED | different protocols and adapted implementations |

## 18. Known Limitations and Unresolved Questions

1. A full **training/inference reproducibility** freeze is not established: source datasets, some predictions, model checkpoints and float16 caches are ignored/local or licensed and not independently backed up by the Git commit; original pipeline training code is lost. Archived CSVs support reanalysis of selected results, not reconstruction of predictions or training. Hash records cannot recover missing bytes.
2. `AGENTS.md` still points to v2. The author expressly authorized v3 here, but only the author can edit that file. Future automated instruction discovery must be updated by the author, or v3 and the later author instruction must be supplied explicitly.
3. Official ACDC preprocessing differs by reconstruction within observed tolerance; effect on official patients is unmeasurable. The 50 patients are from the same dataset/centre, and selection followed exploratory findings. No claim of cross-dataset transfer.
4. Follow-up reuses examined patients and reference runs. The local registration has no verified external timestamp before training; full-vs-band loss scale differs. One teacher per architecture; pretraining and capacity are entangled. Unadjusted confidence intervals and Holm p values answer different questions.
5. ISIC lesion-identity independence is not established even though image IDs and bytes were checked. ACDC spacing/frames and Synapse anatomy naming constrain interpretation. Published comparator protocols and source distances differ.
6. E3 augmentation remains ongoing and outside cutoff. Experiment C needs an amended analysis frozen before any test metrics are read. Later E3 output cannot silently enter this freeze.
7. Several analyses and diagnostics in §11–15 are REPORTED rather than independently re-executed. Archived paths must not be confused with historical, missing or replaced checkpoint provenance. No test was rerun on a restricted held-out cohort for this audit.

## 19. Manuscript Revision Reference

| Section | Evidence / claim IDs to use |
|---|---|
| Abstract | C01, C04–C06: primary ACDC BF1 exact CI and unclear size; distinguish exploratory follow-up and unresolved band/teacher contributions. |
| Introduction and contributions | C01, C03–C06: frame tested comparisons and qualifications; no unsupported novelty or universal benefit. |
| Related Work | C12 and §14: literature protocols, teacher distinction, adapted comparator caveats. |
| Methods | §4–7: model graphs, gate, losses, morphology, teacher/caches, preprocessing, estimands and estimator. |
| Experimental Setup | §3–7 and §10: partitions, seeds, selection, pre-registration timeline, multiplicity and scope. |
| Results | §8–10; C01–C07: retain matched absolute means, 10-run negative finding, stage-2/3 and follow-up with exact CIs. |
| Discussion | §11, §13, §15: estimator sensitivity, possible mechanisms as hypotheses, negative and contradictory evidence. |
| Limitations | §18: held-out reconstruction, selected dataset, reused patients, teacher confounding, missing artifacts, statistical design. |
| Conclusion | C01–C06: studied 2D settings, magnitude uncertainty, no band-specific or teacher-specific causal conclusion. |
| Supplementary Material | §9, §11–15, §20: ablations, per-class and secondary results, full sensitivity, computational and provenance records. |

## 20. Freeze Manifest and Reproducibility Record

### Deterministic preserved evidence

`revision/evidence_freeze/primary-evidence-20261110.tar.gz` contains selected completed case-level metrics, per-run JSON configs/provenance/histories where present, and locked split manifests, in original relative paths. Selection explicitly excludes E3 and Experiment C. It contains 1,108 regular files; sorted archive paths, zero tar uid/gid/mtime and gzip mtime, file mode 0644. `revision/evidence_freeze/primary-evidence-20261110.sha256` lists SHA-256 of the uncompressed contents in lexicographic path order. Verified by reading every archive member and recomputing its digest; 1,108 matched, none missing or extra. The archive is a preservation artifact rather than another narrative summary.

For a clean extraction into a new, empty scratch directory, do not overwrite live results:

```bash
scratch=$(mktemp -d)
tar -xzf revision/evidence_freeze/primary-evidence-20261110.tar.gz -C "$scratch"
(cd "$scratch" && sha256sum -c /ABSOLUTE/PATH/TO/revision/evidence_freeze/primary-evidence-20261110.sha256)
```

Replace the placeholder with the actual clone path. Comparison algorithms are in `analysis/revision/expansion_statistics.py` and `analysis/followup/analyze.py`; the latter expects prediction artifacts absent from this lightweight archive, so direct re-running of that exact script still needs the original prediction files. Patient-level reconstruction can use the included case CSVs and the formula in §7; SHA matching establishes byte identity, not training integrity.

The read-only independent verifier can recompute the ACDC confirmation and
follow-up contrasts directly from the archive without extracting into live
directories: `.venv/bin/python revision/evidence_freeze/verify_snapshot.py`.
It checks every archived member against the manifest and recalculates all 25
contrasts and 20 Holm values. It requires NumPy and SciPy from the environment.

Independent calculations in this audit: (1) all 68 relevant student records: primary case CSV and prediction-file SHA against the two evaluation records; 25 stage-3/follow-up metric contrasts, 20 Holm values; max absolute discrepancy 2.23×10⁻¹⁶; (2) 30 ISIC all-seed/fresh-seed comparisons: 4.45×10⁻¹⁶; (3) 30 three-dataset transfer contrast cells: 3.56×10⁻¹⁵; (4) 165 matched three-run slice-weighted mean/SD cells: all ≤10⁻¹⁰, plus the three M7 rows; (5) 15 stage-2 contrasts: 2.23×10⁻¹⁶; (6) archive and file manifest: 1,108/1,108 hashes. These computations do not verify every REPORTED supplemental value.

### Important SHA-256 values

The digest of the source files below is computed over their actual bytes. The source commit plus this manifest identifies the frozen snapshot; this document does not contain its own post-commit hash.

| Relative source path | SHA-256 |
|---|---|
| revision/evidence_freeze/primary-evidence-20261110.tar.gz | 5fc0d861d8fc1e02446a4dbf97914ed63e5e0b7b6a2231f3c9ad0632903c7f5e |
| revision/evidence_freeze/primary-evidence-20261110.sha256 | 21a9d06818f6f63462a94c7088090ed314293193f32c8c288a236018868970ae |
| revision/evidence_freeze/verify_snapshot.py | 45118384a3232fe1ec61a885c666a4cd7a2524787eb0095d9366dda2133c044d |
| revision/CLAIM_BOUNDARY_v2.md | 61721adee51e79bc5a03ef9fb8599f867d56b7813c4c2bd84b5dd5ab7421f8f5 |
| revision/CLAIM_BOUNDARY_v3.md | 4c8f2befe8be13138cfe858765f057e5a094d6e8c75326e5eaab5b2b1e4f893f |
| preregistration/acdc_confirmation.md | 8ff4509d0f35cc83f9ab38f12bb9ab593b1a2f12847e34786bdf6db78db9c768 |
| revision/followup_registration.md | 30f771eb0c97c6fb9460c505989a1a0c97c99838553b748f9c565d1fa366a5c3 |
| revision/followup/AMENDMENT_1.md | 802799310708ce24520ca49d81cda16367b40bb9f3e4bc77d46b0e8990372256 |
| revision/acdc_confirm/results/results.json | bccc3a40ef61d822f7e036c423da180a8e4070b6ad96eed283a04573a1556131 |
| revision/followup/results/contrasts.csv | 4246542902c18c7ff8af4668dbdc395610a4c891708ac18f5406deff576d3d11 |
| revision/tables/families/expansion_interaction.csv | 786ef0d8a16b0919d9ebe57e3c77b39921244430e7e0f877174a4b7cfb7a01c8 |
| revision/tables/families/expansion_contrasts.csv | 4602ffa0957ce658cc38423e24f70949f99ef8daabf85b151bc6b7209f6e9984 |
| revision/tables/families/method_summary.csv | b7065cc09b9e4ff004aa17e5138864371ada731803da5c3d053ea529760880e2 |
| revision/kd_unified/contrasts.csv | 177aad43b1c3a683dc7702b4fee7c2e5c1f6d8e3d1ab4c7a13968a2bd58beab6 |
| results/acdc_confirm_eval/evaluation_record.json | 80365e94e71e72dc3bd271b94e0880e93c52fa62a6b7c05e05bb369430773ae9 |
| results/followup_eval/evaluation_record_e1e2.json | e246ffe800bf26e92950fd2a8237c68644dbc1ec9f127c299b2fd394162b1e28 |
| configs/revision/acdc_confirm/acdc_confirm_decoder_seed201.yaml | e07cdf08429da0544fa049af00e30745b5021512b6c6dced8e3fda91ee3e6ad8 |
| configs/revision/followup/acdc_followup_kd_full_seed201.yaml | cb181c3c4ae497e5d7b7682099c02edb736f1e5a9bc43df3929549489d57d5dc |
| configs/revision/followup/acdc_followup_kd_band_peer_seed201.yaml | 6c0cdf1c11ae1db30592b91650e4fc5897471ab32db05786993daa3136b1fced |
| scripts/kd_followup/launch.py | 487bf35694d01705f281ac775e0cdb220637a69879b62604a888bea0b6170736 |
| scripts/kd_mc/loss.py | 1b6d572b4f56ad21d8070043f888998b018b3894995f95ba4249862d52ebe309 |
| analysis/revision/expansion_statistics.py | 17099233fb6d2b67d769687109b3cc29a7b9a2f6e432006cad38e7e1d3c6b120 |
| analysis/followup/analyze.py | d1643c20ac1c8961b133d4808769edc66c13906aefb5f2ef0535840d369bbebf |

**Source state at cutoff:** tracked HEAD `5b2ff7a52edf906f16fdea52d7c6a9b4bad282d8`; the working tree was not clean (author changes in `revision/DEVIATIONS.md`, `revision/queue_plan.yaml`, figures, `revision/manuscript/generated/numbers.tex`, and live supervisor state; `PLAN.md` also contained the prior audit record). These files are not treated as source-commit bytes unless independently listed. The training output directory and `revision/followup/supervision/` are mutable; never use their later state to backdate evidence into this freeze.

**Persistent dependencies not fully captured:** `data/processed/`, `revision/restricted_local/`, `results/revision_runs/*/best.pt`, teacher softmax caches and `*.npz` prediction maps are not in the preserved archive. Evaluation records contain checkpoint/prediction SHA-256 values, and registration files contain source/config/split hashes, but many bytes are local or external. Some source releases, licenses, environment packages, and the lost original training pipeline may not be recoverable. The Git-ignored E3 queue outputs and the author's uncommitted Experiment C records were neither committed nor used. The local `.venv` is not a portable environment; `revision/env_freeze.txt` and `pyproject.toml` document part of it. An independent durable, access-controlled backup of licensed data/checkpoints remains needed for full inference/training reproducibility.

**Independent consistency audit:** comparator names, signs, sample units, official vs internal sets, seeds, nominal CIs versus Holm p, teacher capacities, date ordering, source hashes, negative findings and unsupported claims were checked against primary records and code as detailed above. Remaining limitations are explicit in §18. The source commit does not include this draft. No completed freeze commit or annotated freeze tag is established; this draft remains provisional. No claim that a tag by itself preserves ignored large dependencies is made.
