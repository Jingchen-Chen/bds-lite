# Held-out ACDC replication of D+KD-band vs decoder-only: study report

- **Pre-registration:** `preregistration/acdc_confirmation.md`, frozen at `345cacd` (tag `acdc-confirm-prereg-v3`).
- **Amendment 1** (pre-test integrity check): `amendment1/AMENDMENT_1.md`, committed in `885ce41`.
- **Deviations:** `DEVIATION_CHECKLIST.md`.
- **Results:** `results/{results.json, contrasts.csv, per_class.csv, per_seed.csv}` and `seed_correlation_supplement.json`.
- **Provenance:** every number below is read from those files, which `analyze.py` produced from the one evaluation of 2026-11-08 (`results/acdc_confirm_eval/evaluation_record.json`).

## Integrity amendment (stated first)

**The original integrity check failed on a code defect.** The pre-registered check
(`integrity.py`) failed for all 34 runs, and only because the registration lists 13 study
scripts that the frozen trainer never records in its provenance.

**Amendment 1 was adopted before either test set was touched.** It is documented, committed and
covered by regression tests. It replaces that one comparison with stricter checks:
- the 61 recorded source hashes must equal the registration;
- the 13 study files must equal the registration and the freeze commit;
- all 74 registered files must be unchanged on disk.

Every other original check still applies. The original failed output and a 609-file snapshot
were preserved first. The amended check passed before and after the evaluation.

**Nothing in the analysis changed:** the hypotheses, endpoint, estimator, metrics and decision
rules are as registered. See `DEVIATION_CHECKLIST.md`, D1–D3.

## Primary endpoint

Test Boundary F1, D+KD-band − decoder-only, on the 50 official ACDC test patients (101–150),
seeds 201–217, with the frozen unpaired patient-clustered estimator:

| Arm | n | Mean of run means | SD between seeds |
|---|---|---|---|
| D+KD-band | 17 | 0.7975 | 0.0069 |
| Decoder-only | 17 | 0.7797 | 0.0083 |

| Measure | Value |
|---|---|
| Δ | **+0.0178** |
| 95% CI | **[+0.0098, +0.0258]** |
| p (two-sided) | 7.8e-05 |
| Mann–Whitney p | 5.5e-06 |
| SE_run / SE_unit | 0.0026 / 0.0029 |
| Welch df | 30.9 |

**Decision (§11): Success.** The 95% lower bound is above 0.
- **Size: of unclear size.** The lower bound, +0.0098, is just below the SESOI of +0.01.
- **Pre-registered wording:**

  > In a pre-registered replication with 17 fresh seeds per arm, evaluated on the 50 official
  > ACDC test patients that played no part in the exploratory study, D+KD-band raised test
  > Boundary F1 relative to decoder-only by +0.0178 [95% CI +0.0098, +0.0258], a gain of unclear
  > size. ACDC was chosen after the exploratory three-dataset results; the result supports the
  > ACDC gain on held-out patients of this dataset in this setting. It does not show transfer to
  > other datasets, and without a D+KD-full control it does not show that the gain is specific
  > to boundaries.

**Achieved power.** It equals the planned power, since no run failed. Values are for assumption
A, fixed SE_unit, with 50 units:

| Effect | 90% SD bound | 80% SD bound |
|---|---|---|
| +0.015 | 0.803 | 0.930 |
| +0.024 | 0.997 | 0.9999 |

## Secondary analyses (descriptive; they cannot change the decision)

Throughout, positive values favour D+KD-band; distance signs are reversed.

**(a) The 6 exploratory models, seeds 1–3, on the official test set.** These checkpoints were
selected and reported before this study.

| Metric | Δ [95% CI] | Reading |
|---|---|---|
| Boundary F1 | +0.0154 [−0.0031, +0.0338] | not established |
| DSC | +0.0101 [−0.0038, +0.0240] | not established |
| IoU | +0.0140 [−0.0001, +0.0281] | not established |
| HD95 | +0.66 [−1.26, +2.58] px | not established |
| ASSD | +0.31 [−0.21, +0.83] px | not established |

Exploratory and replication seeds combined (20 + 20, official set): Boundary F1 +0.0175
[+0.0099, +0.0250].

**(b) Replication seeds on the internal 20-patient split.** This split was already seen in the
exploratory study, so this is a seed replication only.

| Metric | Δ [95% CI] | Reading |
|---|---|---|
| Boundary F1 | +0.0209 [+0.0088, +0.0330] | gain, size unclear |
| DSC | +0.0141 [+0.0024, +0.0259] | gain, size unclear |
| IoU | +0.0171 [+0.0037, +0.0306] | gain, size unclear |
| HD95 | +1.42 [−0.04, +2.88] px | not established |
| ASSD | +0.43 [−0.04, +0.90] px | not established |

**(c) Seed-paired sensitivity analysis** (17 pairs). The correlation is the cross-arm
correlation of per-seed run means.

| Set | Δ [95% CI] | Seed correlation r |
|---|---|---|
| Primary set | +0.0178 [+0.0106, +0.0250] | +0.580 |
| Internal split | +0.0209 [+0.0101, +0.0317] | +0.617 |

**(d) Other metrics on the primary set.**

| Metric | Δ [95% CI] | Reading |
|---|---|---|
| DSC | +0.0136 [+0.0061, +0.0211] | gain, size unclear |
| IoU | +0.0183 [+0.0105, +0.0262] | relevant gain |
| HD95 | +1.22 [−0.09, +2.54] px | not established |
| ASSD | +0.39 [+0.05, +0.72] px | gain, size unclear |

**(d) Per structure, primary set:**

| Structure | DSC | Boundary F1 | HD95 (px) |
|---|---|---|---|
| RV | +0.0154 [+0.0003, +0.0305] | +0.0263 [+0.0104, +0.0422] | +0.29 [−1.23, +1.82] (no relevant effect) |
| Myocardium | +0.0211 [+0.0131, +0.0290] | +0.0193 [+0.0116, +0.0269] | +1.41 [−0.31, +3.13] |
| LV | +0.0068 [+0.0006, +0.0130] | +0.0109 [+0.0030, +0.0187] | +1.74 [+0.35, +3.14] |

**Also reported:**
- **Seed consistency, primary Boundary F1.** All 17 D+KD-band runs lie above the decoder-only
  mean. The arms overlap: the lowest D+KD-band run minus the highest decoder-only run is −0.0107.
  The per-seed range is 0.7842–0.8078 for D+KD-band and 0.7645–0.7949 for decoder-only. The
  exploratory pattern, in which every D+KD-band run was above every decoder-only run, did not
  recur.
- **Validation contrast** (the selection set, not independent): Boundary F1 +0.0329
  [+0.0126, +0.0533].
- **Training time** (summed epoch time): 7.63 h for D+KD-band and 7.52 h for decoder-only. The
  wall clock was 15.56 h, including 54 thermal pauses.

## Limitations (pre-registered, §13)

- **ACDC was selected after the exploratory results were seen.** The held-out set removes
  test-set reuse, but not that selection.
- **Stage-1 preprocessing of the official set is a validated reconstruction**, not the original
  third-party code.
- **The official test set comes from the same centre and protocol as the training data.**
- **The label check covers label values and LV–myocardium adjacency only.** It does not validate
  the annotations themselves.
- **Evaluation is per 2D slice**, averaged per patient; there are no volume-level metrics.
- **There is one teacher and no D+KD-full control**, so boundary specificity is not assessed.
- **The combined estimates mix two trainer paths.** These are shown identical over 50 steps
  only.

## Open provenance item

**The OpenTimestamps proofs are not yet verified.** The proofs of the pre-registration and the
manifest both carry a Bitcoin block attestation at height 974994. They have **not yet been
verified** against a Bitcoin block header, because no node or chosen header source was available.
The timestamp must not be described as independently verified until that check is done.
