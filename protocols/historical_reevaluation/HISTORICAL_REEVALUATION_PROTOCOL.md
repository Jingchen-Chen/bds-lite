# Historical re-evaluation: frozen protocol

Frozen 2026-11-07, before any official-test metric of the arms below existed. Authorized by the author on 2026-11-07 after review of `revision/audit/project_audit_20261107/REPORT.md` (decisions 4 and 5). Inference only. No model is trained, and no training process is touched. Governed by the rules in `revision/NARRATIVE_CRITERIA_ADDENDUM.md`.

**Freezing.** The SHA-256 of this file and of both scripts below is recorded in `revision/historical_reevaluation/FREEZE.json` and in `revision/DEVIATIONS.md` before evaluation starts. This work is not committed to git at freeze time, because repository checks currently fail on pre-existing lint in unrelated files, and commits are made only on the author's request. The hashes are the freeze record.

## 1. Purpose
The audit raised three questions that existing validated evidence cannot answer:
1. Does the April grid's disagreement with the November ordering survive the current evaluator and the official ISIC2018 test?
2. Does the October development-set pattern (gate without distillation below the decoder; distillation restoring it) appear on independent images?
3. Is Boundary F1, the primary endpoint, sensitive to its matching tolerance for the frozen ten-run contrasts?

## 2. What was known when this was written
This is not a blind study. Known in advance: legacy per-case results of the April arms on their own test sets (`outputs/evaluations/*_test.csv`, ISIC2018 clean 390 images, ACDC, Synapse), with historical Boundary F1 and legacy distance units; October arms on the 519 validation images; all November results. Those numbers prompted the questions, which is why nothing here can confirm anything.

**Not known:** any official-test result of April arms other than U-Net and BDS-Lite, and any official-test result of October arms G, C, and D. Experiment C metrics are unread and are not used.

## 3. Cohorts
| Cohort | Checkpoints | Arm | Evaluated on |
|---|---|---|---|
| `a_decoder` | `outputs/runs/<ds>_bds_lite_boundary_only_seed{1,2,3}` | Decoder only | Official ISIC, clean ISIC 390, ACDC test, Synapse test |
| `a_distill` | `..._bds_lite_boundary_distill_...` | Decoder + distillation, no gate | Same |
| `a_gate` | `..._bds_lite_boundary_gate_...` | Decoder + gate, no distillation | Same |
| `a_gate_surface` | `..._bds_lite_gate_surface_...` | Decoder + gate + surface, no distillation | Same |
| `a_unet`, `a_full` | `..._unet_...`, `..._bds_lite_full_...` | U-Net, BDS-Lite | Clean ISIC 390 only; other sets already exist in `revision/extra_cohorts/famO_*` |
| `o_G`, `o_C` | `recovered/followup/results/pathway_controls_20261018/isic2018_{G,C}_seed{1,2,3}` | Gate only; gate + decoder | Official ISIC |
| `o_D` | `recovered/followup/results/boundary_specificity_20261018/isic2018_D_seed{1,2,3}` | Gate + feature matching, no boundary supervision | Official ISIC |

`a_gate_surface` is added to the audit's list of three April arms, because the with-surface transfer contrast in the audit cannot be checked without it. Clean ISIC 390 means the `test` list of `splits/isic2018_split.json`, disjoint from these models' training and selection images. Models: 12 + 6 + 9 = 27. Every arm and dataset is evaluated once. No checkpoint, arm, seed, or threshold is chosen from results.

## 4. Evaluator
`analysis/revision/historical_reevaluation.py`. Argmax predictions, CPU fp32, per-case metrics from the unchanged `recover_predictions.case_metrics`, the function used for all November cohorts. April state dictionaries load strictly into `BDSLiteUNet` with `projection_head` renamed and the gate present only where its tensors exist; `a_unet` loads into `UNet`; October models load through `build_model` from their stored config. Outputs go to `revision/historical_reevaluation/`, outside `revision/extra_cohorts/`, so that the live refresh job is not triggered.

## 5. Parity gate
`ABLATION_PROVENANCE_WARNINGS.md` requires graph semantics to be checked before reuse. An April model reaches the official test only after passing on all three sets it was originally evaluated on. Against its saved legacy per-case file, excluding both-empty rows:
- absolute difference of mean DSC at most 0.002, and
- at most 2% of rows differ by more than 0.01 in DSC.

An October model must meet the same two limits against its saved validation `per_case.csv` on the first 64 validation images. A model that fails is reported as failed and is not evaluated on the official test. Limits are not changed after seeing a failure.

## 6. Compute rules
- CPU only. `CUDA_VISIBLE_DEVICES` is cleared before torch is imported and the script aborts if CUDA is visible, because a foreign GPU process stops the training supervisor.
- Priority `nice 19`, two threads.
- Automatic pause while any GSL training run is active (`revision/plan.md` 0.3). Also pause at a GPU temperature of 82 C and resume at 72 C. Training normally runs at 80-84 C with a thermal stop at 85 C about every 90 seconds, and the CPU and GPU share cooling.
- Training throughput is compared before and after start, from `saved_progress_epoch` in the supervisor events. If it falls by more than 15%, this job is slowed or stopped. Throughput is the only quantity read; no Experiment C metric is.
- Stop on `revision/historical_reevaluation/STOP`. Completed units are skipped on restart.

## 7. Analysis (frozen in `analysis/revision/historical_statistics.py`)
Estimator: the frozen expansion contrast, imported unchanged from `analysis/revision/expansion_statistics.py`. Difference of arm means of run means, Welch standard error over runs combined with the standard error over test units, t interval. Primary metric Boundary F1; secondary DSC, IoU, HD95, ASSD. Labels are those of `EXPANSION_PROTOCOL.md` Section 9.

**Family H (April, seeds 1-3).** Per dataset (official ISIC, ACDC, Synapse), then pooled by inverse variance with Cochran's Q. Clean ISIC 390 is a sensitivity row and does not enter the pool. Holm over the five contrasts within each metric on the pooled p values.

| ID | Contrast | Question |
|---|---|---|
| H1 | Decoder vs U-Net | Does the decoder direction hold in this pipeline? |
| H2 | Decoder + distillation vs decoder | Transfer without gate or surface |
| H3 | Decoder + gate vs decoder | Gate without distillation |
| H4 | BDS-Lite vs decoder + gate + surface | Distillation within the full recipe |
| H5 | BDS-Lite vs decoder | Does the ISIC2018 deficit hold in this pipeline? |

**Family D (October, official ISIC test, seeds 1-3).** Holm over the eight contrasts within each metric.

| ID | Contrast | Question |
|---|---|---|
| D1 | B vs C | Distillation given decoder and gate |
| D2 | C vs G | Boundary supervision given gate |
| D3 | G vs U | Gate and projection alone |
| D4 | D vs G | Feature matching without boundary supervision |
| D5 | B vs D | Boundary supervision given matching |
| D6 | C vs decoder (M2) | Cost of a gate without distillation |
| D7 | B vs decoder (M2) | Does distillation restore it? |
| D8 | (B - C) - (D - G) | Interaction |

Sensitivity for D6 and D7: decoder arm with all ten seeds.

**Family T (tolerance, frozen predictions).** Boundary F1 at tolerances 1, 2, and 4 px for the six frozen contrasts of `EXPANSION_PROTOCOL.md`, ISIC2018 official test, U-Net, decoder, BDS-Lite, and GSL, seeds 1-10 and fresh seeds 4-10, Holm over the six contrasts within each tolerance. Tolerance 2 must reproduce the frozen values; a mismatch is a blocker. Experiment C runs are excluded by name and by seed.

## 8. Validation before any result is read
All expected outputs exist; sample counts and IDs match; recorded per-case hashes match; predictions are uint8 with the right shape; for each output folder, DSC is recomputed independently from predictions and references for 16 images; every model that reached the official test passed parity; every October and April row used the protocol hash above. `historical_statistics.py` refuses to run unless `validation.json` reports a pass.

## 9. Interpretation (fixed now)
All results are post hoc in origin and carry three seeds. **None can make a contrast Confirmed, change a Stage 1 status, change a tier of the addendum, or enter a registered family.** The April pipeline's trainer is lost, so its results are a cross-pipeline check and not a replication under identical conditions.

| Result | What the manuscript may say |
|---|---|
| H1 pooled estimate negative or zero | The decoder's direction did not hold across training pipelines. Statements that it pointed the same way everywhere are restricted to the retrained set |
| H1 pooled estimate positive | The direction held in both pipelines, with both estimates given. No claim of confirmation |
| H1 on official ISIC differs in sign from November | State it, as the same test images give opposite signs in two pipelines |
| H2 pooled | Reported beside the November transfer estimate with the difference between them stated (no gate here). No estimate pooled across pipelines is used as a claim |
| H5 on official ISIC not negative | The confirmed BDS-Lite deficit is stated as specific to the retrained pipeline |
| H5 on official ISIC negative | Stated as also seen in the April pipeline, with its interval |
| D6 estimate negative and D7 interval includes zero | "Consistent with" a cost of an unconstrained gate that distillation removes; development and test images agree. Post hoc, three seeds |
| Otherwise | The gate-rescue reading is dropped from the main text and reported in the Supplement as not seen on independent images |
| D4 and D5 | Reported as evidence on whether the boundary content of the target matters, in the same hedged terms |
| T: a frozen status would differ at tolerance 1 or 4 | State that the result depends on tolerance. The frozen status at 2 px stands |
| T: same direction at all tolerances | State that direction did not depend on the tolerance tested |

Placement: full tables in the Supplement, at most three sentences in the main Results, and one in Limitations. All experiments in this protocol are reported whatever they show, including parity failures.

## 10. Amendments
None.
