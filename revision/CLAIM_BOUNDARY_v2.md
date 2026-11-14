# Claim boundary v2: PeerJ CS revision (DRAFT for the author's approval)

**Status: DRAFT, revision 2, 2026-11-08. Not approved and not in force.**

**Changes in revision 2.** These follow the author's decisions of 2026-11-08:
- A3: the manuscript uses the literal §11 wording.
- A4: the HD95 exception, in the author's wording.
- New §0: the decoder-only baseline definition, with new forbidden claims F14 and F15.
- D5 and F12: the timestamp is now verified against the block header from two independent block
  explorers.
- §5: the author's decisions are recorded.

## How this document relates to the archived anchor

- **The current anchor stays authoritative.** `AGENTS.md` names
  `archive/manuscript_v2_bspc/manuscript_v2/ANCHOR_ABSTRACT.md` as the claim boundary, and says
  that "the anchor wins". The anchor can only be changed by a human commit with the `anchor:`
  prefix, and AGENTS.md only by an `agents:` commit.
- **This draft does not change the anchor.** The archived anchor is unchanged.
- **For this draft to govern the revision**, the author must approve it and then point `AGENTS.md`
  at it in an `agents:` commit. Until then, the archived anchor stays authoritative.
- **The anchor is outdated in places.** For example, it predates the withdrawal of the ISIC2018
  validation numbers and both KD studies (see §6).

**Sources.** Every number below is copied from a committed file, named in the "Source" column.
The values are as stored in those files, rounded to 4 decimals (2 for distances in pixels).

**Conventions.**
- Δ = first arm − second arm.
- For HD95 and ASSD the sign is reversed, so positive means a smaller distance.
- Intervals are 95%.
- "Decoder-only" is the auxiliary boundary supervision reference. It is defined exactly in §0.

---

## 0. The decoder-only baseline

### 0.1 What the confirmation's decoder-only arm contains

**Configs.** `configs/revision/acdc_confirm/acdc_confirm_decoder_seed{201..217}.yaml`, registered
with their sha256 in `revision/queue_acdc_confirm.yaml`.

**Training path.** `scripts/acdc_confirm/launch.py` calls the frozen
`scripts/kd_mc/train_mc.py` with λ_kd = 0 (pre-registration §9).

| Component | Setting in the decoder-only arm | Reference |
|---|---|---|
| Network | `bdslite_unet`: U-Net encoder, segmentation decoder, a 1-class boundary decoder D_b, a projection from segmentation features, and a boundary feature gate. Base 32 channels; 2,197,493 parameters. | config `model.*`; `src/bds_lite/models/bds_lite.py` (class `BDSLiteUNet`); pre-registration §8 |
| Segmentation loss | Cross-entropy 1.0 + Dice 1.0, **no class weights** (`foreground_class_weights: null`) | config `loss.ce_weight`, `loss.dice_weight`, `loss.foreground_class_weights`; `src/bds_lite/training/losses.py` (weights read at lines 393–395) |
| **Auxiliary boundary supervision** | **Yes.** D_b is trained with BCE + Dice on a union two-pixel boundary target, weight λ_b = 0.05. `detach_boundary_features: false`, so its gradient reaches the shared encoder. D_b runs only in training: `return_boundary` defaults to `self.training`, and the segmentation output never uses D_b. | config `loss.boundary_weight`, `model.boundary_classes`, `model.detach_boundary_features`; `bds_lite.py` `BDSLiteUNet.forward` |
| In-network boundary distillation (BDS-Lite) | **No.** `distill_weight` 0.0 | config `loss.distill_weight` |
| Gate | **Present but inert.** `boundary_gate: true`, `boundary_gate_scale: 0.0`. The gate returns seg_features × (1 + scale·(g − 0.5)), so at scale 0 it is the identity. It is driven by the segmentation-feature projection, not by D_b. | `src/bds_lite/models/bds_lite.py:53`; pre-registration §8 ("gate present with scale 0 (inert)") |
| Surface or distance loss | **No.** `surface_weight` 0.0, `distance_weight` 0.0 | config `loss.surface_weight`, `loss.distance_weight` |
| External-teacher KD | **No.** `kd.lambda_kd` 0.0. `KDCriterionMC` adds L_kd only when λ_kd ≠ 0. | `scripts/kd_mc/loss.py:68` |
| Training and selection | 150 epochs, AdamW (lr 1e-4, weight decay 1e-5), batch 16, cosine schedule, AMP, gradient clip 1.0, no augmentation. Checkpoint selection: first maximum of validation DSC. | config `trainer.*`; pre-registration §8 |
| Data | `data/processed/acdc`; splits `recovered/followup/splits/acdc/{train,val,test}.json` | config `dataset.*`, `revision.manifests` |

**Relation to the treatment arm.** D+KD-band uses the same settings and adds λ_kd = 1.0 L_kd: a
band-restricted soft cross-entropy to a frozen external teacher's softmax (pre-registration §8).

### 0.2 Relation to the "auxiliary boundary supervision" baseline in A1

**Method.** Model, loss and training settings were compared key by key between the confirmation
config and each run's `config.json` under `results/revision_runs/` (comparison run
2026-11-08). The run folder behind each A1 cell was resolved with
`analysis/revision/expansion_statistics.py::cell_path` and `isic_path`.

| A1 evidence | Decoder-only runs behind it | Difference from the confirmation's decoder-only arm | Category |
|---|---|---|---|
| A1.1–A1.4, **ACDC** cell (M2) | `acdc_m2_aux_boundary_only_unweighted_seed{1,2,3}` | **None** in `model.*`, `loss.*` or `trainer.*` (only `project.seed`). <br>• **Trainer:** `scripts/revision/train_run.py`, not the launcher and `train_mc.py`. The two paths are bitwise identical over 50 steps but not shown identical over 150 epochs (`revision/acdc_confirm/equivalence_check.json`; pre-registration §13). <br>• **Seeds:** 1–3, not 201–217. <br>• **Test patients:** the A1 numbers use the internal 20-patient split, not 101–150. <br>• **The same runs serve as** the exploratory KD reference (A2.2) and, on the official set, the decoder-only half of A5.1. | **Same model and losses.** Seeds, trainer script and test patients differ. |
| A1.1–A1.4, **Synapse** cell (M2) | `synapse_m2_aux_boundary_only_seed{1,2,3}` | No difference in `model.*`, `loss.*` or `trainer.*` keys. Different dataset. | Same recipe, **different dataset** |
| A1.1–A1.5, **ISIC2018** cell and the ten-run arm (M2) | `isic2018_m2_aux_boundary_only_seed{1..10}` | `trainer.batch_size` 8 (vs 16) and `trainer.lr` 3e-4 (vs 1e-4), dataset-specific. Binary, 3-channel data. Same runs as the exploratory KD ISIC2018 reference (A2.1). | **Different but comparable** (same losses and model; different dataset and two optimiser settings) |
| A1.6, historical re-evaluation (`a_decoder`) | Original-set runs trained with the lost pipeline of the earlier submission | Different training pipeline. The manuscript compares arms "only within a set" because same-recipe runs of the two sets differed (`revision/manuscript/main.tex` line 198). | **Not comparable** beyond the nominal recipe |

**What A1 adds on ACDC.** The A1 transfer contrast on ACDC is M3 − M2, where
`acdc_m3_no_sdf_unweighted` is the same config plus `distill_weight` 0.05 and
`boundary_gate_scale` 0.25. Its result is +0.0106 [−0.0052, +0.0263], "undetermined"
(`revision/tables/families/expansion_interaction.csv`: set=transfer, effect="distillation+gate
without surface", metric=boundary_f1, dataset=acdc).

### 0.3 What this means for setting A3 against A1 in one argument

**Allowed.** On ACDC, two different additions were each tested against the same decoder-only
configuration:
- BDS-Lite's in-network distillation + gate (A1; three runs per arm; internal test patients);
- external-teacher band KD (A3; 17 runs per arm; official patients 101–150).

They may be presented side by side, as separate results against the same baseline definition.
For example: "against the same decoder-only configuration, a gain was established for
external-teacher band KD on ACDC (A3) and not for in-network distillation (A1)."

**Not allowed.** Any comparison *between* the two additions:
- a head-to-head claim;
- a difference between the A3 and A1 effect sizes;
- "KD helps more than in-network distillation".

**Why not.**
- No pre-specified contrast between them exists.
- The two results differ in seeds and number of runs, trainer script, and test patients.
- The only direct D+KD-band − BDS-Lite contrast is exploratory and uses the full recipe (M6,
  with the surface loss), not M3: ACDC +0.0237 [−0.0002, +0.0475], not established
  (`revision/kd_unified/contrasts.csv`, dataset=acdc, comparison="KD-band - BDS-Lite",
  metric=boundary_f1).

See F14 and F15.

---

## 1. Allowed claims

### A1. BDS-Lite's in-network boundary feature distillation did not establish a gain over auxiliary boundary supervision (original study)

| # | Evidence | Number | Source |
|---|---|---|---|
| A1.1 | **Transfer path (distillation + gate), without the surface loss.** Three runs per arm, pooled over ISIC2018, ACDC and Synapse; Boundary F1. | +0.0092 [−0.0010, +0.0194], Cochran Q p = 0.57; "undetermined" | `revision/tables/families/expansion_interaction.csv`, set=transfer, effect="distillation+gate without surface", metric=boundary_f1, dataset="pooled (Q p=0.57)" |
| A1.2 | The same, with the surface loss | +0.0011 [−0.0104, +0.0125], Q p = 0.53; "undetermined" | same file, effect="distillation+gate with surface", boundary_f1, pooled |
| A1.3 | Pooled DSC, without / with the surface loss | +0.0010 [−0.0040, +0.0061] / −0.0023 [−0.0085, +0.0039]; both "no relevant effect" | same file, metric=dsc, pooled rows |
| A1.4 | Pooled HD95, without / with the surface loss | +0.59 [−0.36, +1.54] / −0.44 [−1.33, +0.44] px; both "no relevant effect" | same file, metric=hd95, pooled rows |
| A1.5 | **ISIC2018 retrained set**, 10 runs per arm: the full BDS-Lite recipe − decoder-only, Boundary F1 | −0.0321 [−0.0518, −0.0124], Holm p = 0.0187. Fresh seeds: −0.0405 [−0.0668, −0.0142], Holm p = 0.0314. Status "confirmed". | `revision/tables/families/expansion_contrasts.csv`, metric=boundary_f1, first=full, second=decoder |
| A1.6 | **Original (lost) pipeline**, post hoc re-evaluation, 3 seeds: full − decoder-only, Boundary F1, pooled | +0.0067 [−0.0082, +0.0216], Holm p = 1.0; ISIC2018 alone +0.0255 [−0.0782, +0.1293] | `revision/tables/families/historical_april.csv`, id=H5 |

**Wording limits.**
- **A1.5 must be scoped.** It may be stated only as "scored below the decoder alone on ISIC2018
  in the retrained set". The full recipe also contains the surface loss, so the deficit cannot be
  assigned to distillation.
- **A1.5 must not be generalised.** A1.6 shows that the deficit did not hold in the original
  pipeline.
- **Allowed summary:** "In the studied setting, BDS-Lite's in-network distillation did not
  establish a gain over the auxiliary boundary decoder."

### A2. Exploratory external-teacher boundary-band KD (D+KD-band − decoder-only): positive point estimates on all three datasets; none survived Holm correction

Primary contrast, test Boundary F1. The Holm adjustment is over the three datasets.

| # | Dataset (D+KD-band seeds; reference seeds) | Δ [95% CI] | p (unadjusted) | Holm p | Source |
|---|---|---|---|---|---|
| A2.1 | ISIC2018 (1–3; decoder-only 1–10) | +0.0376 [−0.0565, +0.1317] | 0.248 | 0.496 | `revision/kd_unified/contrasts.csv`, dataset=isic2018, comparison="KD-band - decoder-only", reference_seeds=1-10, metric=boundary_f1, primary=True |
| A2.2 | ACDC (1–3; 1–3) | +0.0239 [+0.0040, +0.0438] | 0.029 | 0.088 | same file, dataset=acdc, primary=True |
| A2.3 | Synapse (1–3; 1–3) | +0.0150 [−0.0423, +0.0724] | 0.500 | 0.500 | same file, dataset=synapse, primary=True |

**Wording limits.**
- **Call it exploratory.** It must be described as exploratory, with three D+KD-band runs per
  dataset.
- **Name the ISIC2018 reference.** The ISIC2018 reference has 10 decoder-only seeds, not
  seed-matched; this must be stated or the row footnoted.
- **No unadjusted significance claims.** The ACDC unadjusted p = 0.029 may be reported only
  next to its Holm p = 0.088.
- **Stay within D+KD-band vs decoder-only.** The comparisons against U-Net and BDS-Lite in the
  same file have no Holm adjustment and are not part of A2.

### A3. Pre-registered held-out confirmation on ACDC

**Setup.** 50 official ACDC test patients (101–150), not used in the exploratory study; 17
fresh seeds per arm (201–217). The estimator is the frozen unpaired, patient-clustered contrast.

| # | Quantity | Number | Source |
|---|---|---|---|
| A3.1 | Δ, test Boundary F1 (D+KD-band − decoder-only) | +0.0178 [+0.0098, +0.0258]; p = 7.8e-05 | `revision/acdc_confirm/results/results.json` → `primary.contrast` (delta 0.017820, low 0.009829, high 0.025812, p 7.84e-05) |
| A3.2 | Arm means of run means (SD between seeds) | D+KD-band 0.7975 (0.0069); decoder-only 0.7797 (0.0083) | same, `arm_a_mean` / `arm_b_mean` / `arm_a_sd` / `arm_b_sd` |
| A3.3 | Pre-registered decision | "success", size "of unclear size" | same, `primary.decision` |
| A3.4 | Achieved power (equals the planned power; no failed runs) | 0.803 at Δ = +0.015, 90% SD bound | same, `achieved_power.delta_0.015_sd_bound_90` (0.803245) |
| A3.5 | Seed-paired sensitivity (descriptive) | +0.0178 [+0.0106, +0.0250] | `revision/acdc_confirm/results/contrasts.csv`, analysis="replication seeds, primary set; seed-paired sensitivity", metric=boundary_f1 |

**Required wording.** This is pre-registration §11 ("Success"), verbatim, with the placeholders
filled:
- The template's `Δ [95% CI]` is filled with A3.1.
- The pre-registered phrase "of unclear size" is added, because the lower bound, +0.0098, is
  below +0.01.

> In a pre-registered replication with 17 fresh seeds per arm, evaluated on the 50 official
> ACDC test patients that played no part in the exploratory study, D+KD-band raised test
> Boundary F1 relative to decoder-only by +0.0178 [95% CI +0.0098, +0.0258], of unclear size.
> ACDC was chosen after the exploratory three-dataset results; the result supports the ACDC gain
> on held-out patients of this dataset in this setting. It does not show transfer to other
> datasets, and without a D+KD-full control it does not show that the gain is specific to
> boundaries.

- **The template text is the source.** It is in `preregistration/acdc_confirmation.md` §11
  (frozen, sha256 `8ff4509d…c768`).
- **Author decision 4:** the manuscript uses this literal form. `revision/acdc_confirm/STUDY_REPORT.md`
  stays as committed ("a gain of unclear size").
- **Do not round the interval to 3 decimals.** At 3 decimals the lower bound would read +0.010,
  which looks as if it meets the 0.01 smallest effect of interest.

### A4. Secondary results consistent in direction (descriptive; no multiplicity adjustment; cannot change A3)

Primary set, replication seeds:

**How secondary results are labelled.**
- **In the manuscript**, each secondary result is described only as "interval excludes zero" or
  "not established" (author decision 8).
- **Pre-registered relevance readings.** Pre-registration §7.5 defines a smallest effect of
  interest for every metric used here: 0.01 for DSC, IoU and Boundary F1; 2 px for HD95; 0.5 px
  for ASSD. Its relevance readings (e.g. "relevant gain", "gain, size unclear") are therefore
  pre-registered for every A4 row. They are kept below in a separate column for the record
  only, and **are not used in the manuscript**.

| # | Metric | Δ [95% CI] | Manuscript reading | §7.5 reading (record only) | Source |
|---|---|---|---|---|---|
| A4.1 | DSC | +0.0136 [+0.0061, +0.0211] | interval excludes zero | gain, size unclear | `revision/acdc_confirm/results/contrasts.csv`, analysis="replication seeds, primary set", metric=dsc |
| A4.2 | IoU | +0.0183 [+0.0105, +0.0262] | interval excludes zero | relevant gain | same, metric=iou |
| A4.3 | ASSD | +0.39 [+0.05, +0.72] px | interval excludes zero | gain, size unclear | same, metric=assd |
| A4.4 | HD95 | +1.22 [−0.09, +2.54] px | not established | undetermined | same, metric=hd95 |

Per structure (each cell is Δ [95% CI]; manuscript reading in parentheses):

| # | Structure | DSC | Boundary F1 | HD95 (px) | Source |
|---|---|---|---|---|---|
| A4.5 | RV | +0.0154 [+0.0003, +0.0305] (interval excludes zero) | +0.0263 [+0.0104, +0.0422] (interval excludes zero) | +0.29 [−1.23, +1.82] (not established) | `revision/acdc_confirm/results/per_class.csv`, structure=RV |
| A4.6 | Myocardium | +0.0211 [+0.0131, +0.0290] (interval excludes zero) | +0.0193 [+0.0116, +0.0269] (interval excludes zero) | +1.41 [−0.31, +3.13] (not established) | same, structure=myocardium |
| A4.7 | LV | +0.0068 [+0.0006, +0.0130] (interval excludes zero) | +0.0109 [+0.0030, +0.0187] (interval excludes zero) | +1.74 [+0.35, +3.14] (interval excludes zero) | same, structure=LV |

The §7.5 readings of the per-structure rows (record only) are in the `reading` column of
`per_class.csv`.

**Allowed wording (author decision 3):**
- Secondary and descriptive: the direction was consistent and established for DSC, IoU and ASSD.
- HD95 was not established.
- Per structure, the direction was established for all three structures only in DSC and
  Boundary F1.

**Rules for using it.**
- **"Established"** here means the 95% interval excludes zero, with no multiplicity adjustment.
- **HD95 must not be omitted** when DSC, IoU or ASSD are cited (A4.4; per structure, A4.5–A4.7).
- **No relevance labels for secondary results.** None of the A4 results is described in the
  manuscript as "relevant" or as above the smallest effect of interest (F16).

### A5. Other context (descriptive only; allowed only with these labels)

| # | Item | Number | Source | Required label |
|---|---|---|---|---|
| A5.1 | Exploratory 6 models (seeds 1–3) on the official set, Boundary F1 | +0.0154 [−0.0031, +0.0338], not established | `revision/acdc_confirm/results/contrasts.csv`, analysis="exploratory seeds 1-3, official test" | Descriptive context, **not** a confirmation (pre-registration §11) |
| A5.2 | Replication seeds on the internal 20-patient split, Boundary F1 | +0.0209 [+0.0088, +0.0330] | same, analysis="replication seeds, internal 20-patient split (seen)" | Already-seen data; a seed replication only |
| A5.3 | Seed correlation (paired), primary set, Boundary F1 | r = +0.580 | `revision/acdc_confirm/seed_correlation_supplement.json` | Descriptive; see Dev-3 (D2) |

---

## 2. Forbidden claims

| # | Forbidden | Why (evidence) |
|---|---|---|
| F1 | That the boundary-band restriction **causes** the gain, or that it is boundary-specific | There is no full-image KD (D+KD-full) control (pre-registration §8, §11, §13). |
| F2 | That the effect **generalizes** beyond ACDC, including to ISIC2018, Synapse, other centres, scanners or 3D | ISIC2018 and Synapse are exploratory and not established (A2.1, A2.3). The official test set comes from the same centre and protocol (§13). |
| F3 | That **teacher strength** (or any teacher property) explains when KD helps | One teacher, trained once; teacher capacity, pretraining and independence vary together (§13). |
| F4 | Any **causal explanation from dataset or structure properties** | The RV > myocardium > LV ordering of Boundary F1 gains (A4.5–A4.7) may appear only as a hypothesis in the Discussion, labelled as such. |
| F5 | Any use of the **pooled cross-dataset Boundary F1 interval** of the exploratory KD study (+0.0239 [+0.0113, +0.0366], `revision/kd_unified/summary.md`) as inferential evidence | Three datasets cannot support a between-dataset variance (pre-registration §1, §12.10). The interval may not appear as support for any claim. |
| F6 | Describing the ACDC selection as anything other than **post hoc**, made after the exploratory three-dataset results | Pre-registration §1, §13 |
| F7 | That the exploratory models' official-set result **confirms** the finding | Pre-registration §11 |
| F8 | That BDS-Lite's in-network distillation helps, or that "distillation" in general adds to the decoder, without naming the external-teacher, band-restricted, ACDC-only condition | A1 and A3 concern different methods. |
| F9 | That D+KD-band exceeds U-Net, BDS-Lite or GSL as a confirmed result | The confirmation tested only D+KD-band vs decoder-only. The other contrasts in `revision/kd_unified/contrasts.csv` have no Holm adjustment and three runs per arm. |
| F10 | That every D+KD-band run scored above every decoder-only run | True in the exploratory runs (lowest D+KD-band 0.8186 vs highest decoder-only 0.8095; `revision/kd_unified/per_seed.csv`, acdc, boundary_f1), but **not** in the replication (disclosure D4). |
| F11 | A gain "above the smallest effect of interest" | The lower bound +0.0098 is below +0.01 (A3.1). The seed-paired sensitivity bound (+0.0106) cannot change the pre-registered reading. |
| F12 | Any timestamp claim beyond D5. Allowed: "verified against the block header from two independent block explorers". Not allowed: that the Bitcoin timestamp shows the pre-registration predates the training launch. | The block header time (2026-11-07T22:23:50Z) is about 14 min **after** the launch (22:09:31Z). The proof bounds existence before the test evaluation, not before training (D5). |
| F13 | The anchor's existing must-not-say list (e.g. "novel", "state-of-the-art", "outperforms", "robust", "generalizes", "clinical") | It still applies; it is carried over unchanged from the archived anchor. |
| F14 | Equating the confirmation's decoder-only arm with the A1 decoder runs beyond what §0.2 establishes. That is, calling them "the same runs", the same test conditions or the same pipeline. Or equating it with the original-set (historical) decoder of A1.6. | §0.2: identical config only for the ACDC cell, with different seeds, trainer script and test patients. ISIC2018 and Synapse are different datasets. A1.6 is a different, lost pipeline. |
| F15 | Any head-to-head or difference-of-effects claim between external-teacher band KD (A3) and BDS-Lite's in-network distillation (A1). Examples: "KD outperforms / is better than in-network distillation"; "the KD gain exceeds the distillation effect by …". | §0.3: no pre-specified contrast; different seeds, N, trainer and test patients. The only direct exploratory contrast (vs M6) is not established. |
| F16 | Presenting any secondary metric as stronger evidence than the primary endpoint. Examples: calling IoU a "relevant gain" while the primary is "of unclear size"; leading with a secondary metric; citing a secondary interval in place of the primary. | Secondary analyses are descriptive and cannot change or replace the primary decision (pre-registration §4). In the manuscript they carry only "interval excludes zero" or "not established" (A4). |
| F17 | Using, or implying any knowledge of, Experiment C test results | None has been analysed. The original analysis cannot be run as registered, and an amended analysis must be frozen before any test metric is read (D7). |

---

## 3. Required disclosures

| # | Disclosure | Content and numbers | Source |
|---|---|---|---|
| D1 | **Post hoc selection of ACDC** | ACDC was chosen after the exploratory results (ACDC +0.0239, Holm 0.088; ISIC2018 Holm 0.496; Synapse Holm 0.500). The held-out set removes test-set reuse, but not this selection. | `revision/kd_unified/contrasts.csv`; `preregistration/acdc_confirmation.md` §1, §13 |
| D2 | **Amendment 1 and study deviations Dev-1–Dev-3** (labelled D1–D3 in the committed `DEVIATION_CHECKLIST.md`; renamed here to avoid a clash with disclosures D1–D8) | **Dev-1:** the pre-test integrity check could not pass because of a registration/provenance pairing defect. The registration lists 74 sources; provenance records 61 (the 13 study scripts are not recorded). A pre-test amendment adopted a stricter check before either test set was accessed (commit `885ce41`). **Dev-2:** a count misreported during the diagnosis, corrected. **Dev-3:** the seed correlation was omitted from the analysis outputs and recovered with the unchanged function. The hypotheses, endpoint, estimator and decision rules are unchanged. | `revision/acdc_confirm/amendment1/AMENDMENT_1.md`; `revision/acdc_confirm/DEVIATION_CHECKLIST.md` |
| D3 | **Preprocessing reconstruction and its measured impact** | **Gate on 1,536 development slices:** masks bit-exact; image difference up to 0.0148 z-units (tolerance 0.015). **Impact on the 10 validation patients** (no test data): run-level Boundary F1 changed by at most 0.00011, DSC by at most 0.00005. The validation arm contrast moved by −0.00011 (Boundary F1). **Not measured:** the effect on the official patients, whose original third-party preprocessing does not exist. | `revision/acdc_confirm/preprocessing_dev_gate.json` (`image_max_abs_diff` 0.014784); `revision/acdc_confirm/preprocessing_impact.json` (`max_abs_run_level_difference`); `preregistration/acdc_confirmation.md` §2 (contrast shift) |
| D4 | **Overlap between arms** | All 17 D+KD-band runs lie above the decoder-only mean, but the lowest D+KD-band run minus the highest decoder-only run is −0.0107. Per-seed ranges: D+KD-band 0.7842–0.8078, decoder-only 0.7645–0.7949. | `revision/acdc_confirm/results/results.json` → `consistency_boundary_f1`; `revision/acdc_confirm/results/per_seed.csv` |
| D5 | **Timestamp status: state only what has been verified** | See below this table. | `revision/acdc_confirm/PROTOCOL_LOG.md` (entries 2026-11-07T22:09:11Z, 2026-11-08T05:17:27Z, 2026-11-08T14:31:53Z); `revision/acdc_confirm/TIMESTAMP_MANIFEST.txt`; the `.ots` files (commit `cd5e3b6`) |
| D6 | **The confirmation's limitations** (pre-registered §13, as written) | Same centre and protocol; label check limited to values and adjacency; 2D slice evaluation averaged per patient; one teacher; trainer mix in the combined estimates. | `preregistration/acdc_confirmation.md` §13 |
| D7 | **Experiment C** (supplementary-level) | See below this table. | See below |
| D8 | **Cost of KD** | See below this table. | See below |

**D5, the timestamp status in detail.**

Status: **verified against the block header from two independent block explorers.** In detail:

- **The proofs.** Both OpenTimestamps proofs carry `BitcoinBlockHeaderAttestation(974994)`:
  `preregistration/acdc_confirmation.md.ots` (file sha256 `8ff4509d…c768`) and
  `revision/acdc_confirm/TIMESTAMP_MANIFEST.txt.ots` (`d8d69250…3388`). Both are committed in
  `cd5e3b6`.
- **The expected root.** `ots info` gives the expected Merkle root
  `337ef21b0e0a41ffa21c8119e4e257f56040b590b09ce4e0799396147127bce9` for both files.
- **The fetched headers.** On 2026-11-08T14:31:17Z the header of block 974994 was fetched from
  mempool.space and from blockstream.info. Only the height and the block hash were sent.
  - Both returned block hash `0000…5833dc5c` and the identical 80-byte header.
  - The double-SHA256 of that header equals the block hash, checked locally.
  - The header's Merkle root equals the expected root, in the same display byte order as
    `ots info`.
  - All three roots match.
- **What the proof establishes.** Both files existed **no later than the block header time,
  2026-11-07T22:23:50Z**, which is 15.6 h before the first test-set access
  (2026-11-08T13:58:47Z).
  - It does **not** show that the pre-registration predates the training launch (22:09:31Z).
  - That ordering rests only on the local freeze commit `345cacd` (never pushed; commit dates
    are self-reported) and on the calendar submission at 22:08:52Z, which was recorded locally
    (`de9a1bb`).
- **What was not done.** `ots verify` itself was not run, because no Bitcoin node is available.
  The check above reproduces what it checks, using two explorer copies of the header.

**D7, Experiment C, in detail (supplementary level).**

| Item | Content | Source |
|---|---|---|
| Why it was started | It was written after the frozen expansion results, which prompted its questions. **Primary:** does the training-only boundary decoder raise Boundary F1 relative to a matched U-Net on fresh seeds? **Secondary:** BDS-Lite vs decoder, and decoder and BDS-Lite vs GSL, on ACDC and Synapse. **Design:** 70 runs (ACDC and Synapse: 4 arms × seeds 4–10; ISIC2018: U-Net and decoder × seeds 11–17). All runs were to finish before any test metric was read. | `revision/CONFIRMATION_PROTOCOL.md` §1–§3 (frozen, commit `ebf00eb`) |
| Why it was stopped | **2026-11-07T11:52:53Z:** the author removed ISIC2018 seeds 11–17 "to reduce training volume". **2026-11-07T11:58:35Z:** the author stopped all remaining training (a compute cancellation; `revision/STOP_TRAINING`). Both records state that no Experiment C test metric was read for either decision. | `revision/CONFIRMATION_SCOPE_AMENDMENT_20261107.md` and `revision/DEVIATIONS.md` (**both are uncommitted author records**; see §5 item 10) |
| Which runs completed | **14 of 70 completed:** <br>• ACDC seed 4 and seed 5, all four arms (8); <br>• Synapse seed 4, all four arms (4); <br>• ISIC2018 seed 11, U-Net and decoder (2). <br>**1 incomplete:** `synapse_m0_unet_seed5` (last saved epoch 112 of 150). **55 never started.** | `revision/audit/experiment_c_blinding_check/experiment_c_status.json` (**uncommitted**; metric-free) |
| Were test results computed? | **Yes, automatically, for the 14 completed runs.** The frozen `train_run.py` scores the test set at the end of every run, writing `per_case_test.csv`, `preds_test.npz` and the `metrics.test` block of `final_metrics.json`. | Same record; pre-registration §9 |
| Were test results viewed? | **No evidence that any was viewed.** <br>• **File access times** (`relatime`): no `per_case_test.csv` or `preds_test.npz` has been read since it was written. `final_metrics.json` was read 6–10 s after writing in 9 of 14 runs, which matches the queue's automatic completion check. The 10th read was a key-listing check on 2026-11-08 that printed only key names. <br>• **Agent transcripts:** 32 Claude Code and Codex sessions, 1,823 tool calls. No tool output contains both an Experiment C run name and a test-metric token. <br>• **Derived files:** `auto_refresh.py` reads only seeds 1–3 or seed 1, and the audit ledger marks Experiment C as "BLIND" with no values. <br>• **Limitation:** `relatime` cannot rule out a read of `final_metrics.json` within 24 h of its first read, or a human read outside the logged agent sessions. | Same record; `revision/audit/experiment_c_blinding_check/scan_transcripts.py`; `revision/audit/project_audit_20261107/experiment_ledger.csv` |
| Required statements | Experiment C is incomplete. No Experiment C result is used anywhere. Its registered analysis cannot be executed as frozen. Any future use needs an amended analysis frozen before any test metric is read. | `revision/CONFIRMATION_SCOPE_AMENDMENT_20261107.md` ("Analysis boundary") |

**D8, cost of KD, in detail.**

| Item | Number | Source |
|---|---|---|
| Teacher architecture | `segmentation_models_pytorch` 0.6.0 `Unet("resnet34", encoder_weights="imagenet")`, one per dataset, trained once (seed 100) | `revision/KD_MULTICLASS_PROTOCOL.md` §3; pre-registration §8 |
| Teacher parameters (ACDC) | 24,436,804 | `revision/kd_multiclass/benchmark.json` (`role: teacher`, `dataset: acdc`) |
| Teacher training (ACDC) | 150 epochs, summed epoch time 0.243 h (median 5.80 s per epoch); selected epoch 13 | `revision/kd_unified/costs.csv` (run `acdc_kd_teacher_seed100`); pre-registration §8 |
| Teacher cache | 1,350 × 4 × 224 × 224 float16 softmax, about 0.54 GB. **The time to compute the cache is not recorded in any committed file and is not stated.** | Pre-registration §8; `revision/KD_MULTICLASS_PROTOCOL.md` §3 |
| Student training cost, confirmation | Summed epoch time over 17 runs: D+KD-band 7.63 h, decoder-only 7.52 h. The difference, 0.10 h (about 1.4%), is computed from these two values. | `revision/acdc_confirm/results/results.json` → `training_epoch_hours` |
| Student parameters | 2,197,493 in both arms | Pre-registration §8; `revision/kd_multiclass/benchmark.json` (`role: student`, `dataset: acdc`) |
| **Inference cost: unchanged by KD** | **By construction.** All 34 confirmation configs have one identical `model` block (checked 2026-11-08). KD enters only through the training loss (`scripts/kd_mc/loss.py:68`). The teacher is never part of inference, and the deployed graph equals that of decoder-only. **Additional check:** the `best.pt` files of both arms are the same size on disk (8,849,447 bytes); this is not in a committed file. **Not measured:** inference latency was not measured separately in this study. | `configs/revision/acdc_confirm/*.yaml`; `revision/KD_COMPARATOR_PROTOCOL.md` line 68; `revision/KD_MULTICLASS_PROTOCOL.md` §7 |

---

## 4. Draft answer to the title question (inside the allowed claims)

> *When does boundary distillation help lightweight medical image segmentation?* In our
> experiments, the answer depended on what was distilled and on the reference it was judged
> against. BDS-Lite's own in-network boundary feature distillation, compared with the auxiliary
> boundary decoder it builds on, did not establish a gain in any studied condition: pooled over
> three datasets with three runs per arm, distillation and the gate changed Boundary F1 by
> +0.0092 [95% CI −0.0010, +0.0194] without and +0.0011 [−0.0104, +0.0125] with the surface
> loss. On ISIC2018 in the retrained set, the full BDS-Lite recipe, which also includes a
> surface loss, scored below the decoder-only reference (Boundary F1 −0.0321 [95% CI −0.0518,
> −0.0124] over ten runs per arm, confirmed on fresh seeds). That deficit was not seen in a post
> hoc re-evaluation of the original training pipeline (+0.0067 [−0.0082, +0.0216], pooled over
> three datasets). Because the recipe combines several signals, the deficit is not attributed to
> distillation. Distillation from an external teacher restricted to a boundary band gave positive
> exploratory point estimates against the same decoder-only reference on ISIC2018, ACDC, and
> Synapse, none of which survived Holm correction. ACDC, selected after those results were seen,
> was then tested in a pre-registered replication with 17 fresh seeds per arm on the 50 official
> test patients not used in the exploratory study. Boundary F1 was higher by +0.0178 [95% CI
> +0.0098, +0.0258], of unclear size relative to the pre-registered smallest effect of interest
> of 0.01. The only condition in which we established a gain over the decoder-only reference is
> therefore external-teacher boundary-band distillation on ACDC. Our data do not identify which
> aspect of that condition is responsible, whether the band restriction, the teacher or the
> dataset, and they do not show that the gain transfers to other datasets.

---

## 5. Author decisions (2026-11-08) and open points

1. **Adoption (open).** The author reads this document end to end, then gives a one-time
   authorization for an `agents:` commit that points AGENTS.md's claim boundary here, keeping
   the archived anchor as historical. That commit has not been made.
2. **Decided: narrow four sentences in the manuscript.** The sentences in
   `revision/manuscript/main.tex` (lines 35, 56, 351, 400) will be narrowed to BDS-Lite's
   in-network distillation during section rewriting, and recorded in the revision plan. The
   manuscript does not yet describe the external-teacher KD studies.
3. **Decided: HD95 exception.** The wording is in A4.
4. **Decided: literal §11 wording in the manuscript** (A3). `STUDY_REPORT.md` stays as
   committed.
5. **Done: timestamp verified** against two block explorers. The proofs are committed in
   `cd5e3b6`, and D5 is updated.
6. **Done: baseline definition.** See §0, F14 and F15.
7. **Done: the A1.5 sentence** has been added to §4.
8. **Done: secondary labels.** Relevance labels are removed from manuscript wording (A4), and
   F16 is added.
9. **Done: renumbering.** The study deviations are now Dev-1–Dev-3, and F10 cites D4.
10. **Open: committing the Experiment C sources (D7).** Some D7 sources are not committed:
    - `revision/CONFIRMATION_SCOPE_AMENDMENT_20261107.md` and the Experiment C entries in
      `revision/DEVIATIONS.md` are the author's own uncommitted records. Agents do not stage
      them.
    - `revision/audit/experiment_c_blinding_check/` (the status record and the scanner) was
      written on 2026-11-08 and is also uncommitted.

    Under the rule that every number comes from a committed file, D7's run counts become
    citable only after these are committed.
11. **Done: KD cost** (D8). Two gaps are stated rather than estimated: the time to compute the
    teacher cache, and inference latency.
12. **Open, for information: auto-refresh is still running.** `analysis/revision/auto_refresh.py`
    (pid 1101238, running since Nov 2) is still active. It regenerates
    `revision/manuscript/generated/numbers.tex` and figures, and recompiles the manuscript,
    supplement and response letter whenever new runs complete. It last refreshed at
    2026-11-08T14:01:20Z, when the replication runs finished. The working-tree changes to
    `numbers.tex` and `revision/figures/*.pdf` may therefore come from this process, not from
    manual edits. Before section rewriting starts, the author may want to stop it.

## 6. Where the archived anchor is out of date (for information; the anchor is not modified)

- **Withdrawn numbers.** Its ISIC2018 numbers (validation Boundary F1 0.5597 vs 0.5520) belong
  to the earlier submission. The current manuscript says these "were measured largely on training
  images" and withdraws them.
- **Three seeds only.** Its design is three seeds per arm. The revision adds ten-run arms on
  ISIC2018 (A1.5) and the 17-seed confirmation (A3).
- **No KD studies.** It contains no external-teacher KD result and no held-out ACDC result.
