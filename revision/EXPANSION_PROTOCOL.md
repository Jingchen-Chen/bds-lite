# Expansion experiments: frozen protocol

Frozen 2026-11-05T22:49Z, before any expansion run started. Authorized by the author on 2026-11-05 after review of `revision/NARRATIVE_ASSESSMENT.md`. The git commit containing this file is the time stamp. Changes after that commit are amendments and must be recorded in section 11.

## 1. Purpose
1. Resolve the largest open question: do the decoder, BDS-Lite, and GSL change ISIC2018 Boundary F1 beyond retraining noise?
2. Fill the missing decoder + surface cell, so that the decoder x surface interaction can be estimated.

## 2. What was known when this was written
This is not a blind registration. Results for seeds 1-3 of every arm were known and shaped these questions: on ISIC2018 Boundary F1, decoder +0.016, BDS-Lite +0.004, GSL +0.020 relative to U-Net, all inside a run-level margin of about 0.038. Original-set models gave +0.031 for BDS-Lite. Nothing about seeds 4-10 or the new arm was known. For this reason, confirmation rests on fresh seeds only (section 8).

## 3. Runs (40)
| Group | Arm | Datasets | Seeds | Runs |
|---|---|---|---|---|
| Correction | U-Net + boundary decoder, without class weights | ACDC | 1-3 | 3 |
| Experiment B | U-Net + boundary decoder + surface loss (no distillation, no gate) | ISIC2018, ACDC, Synapse | 1-3 | 9 |
| Experiment A | U-Net; U-Net + boundary decoder; BDS-Lite; U-Net + GSL | ISIC2018 | 4-10 | 28 |

**Addition beyond the 37 authorized runs.** An audit of the effective configuration of all 33 arm-dataset cells found one outlier: the ACDC decoder-only arm was trained with foreground class weights (1.5, 1.0, 1.25). Ablation Amendment 1 re-registered the other ACDC ablations without weights and missed this arm. Every other retrained ACDC arm is unweighted, so the existing arm is not a fair corner of the 2x2. Its 3 runs are preserved, reported as a correction, and replaced in all ACDC analyses by the unweighted runs above.

## 4. Frozen configurations
- Builders: `analysis/revision/expansion_arms.py`. Registered configs: `configs/revision/*_seed*.yaml` listed in `revision/queue_expansion.yaml`, with SHA-256.
- Locked protocol, unchanged: 224x224 input, AdamW (weight decay 1e-5), cosine schedule over 150 epochs, gradient clipping 1.0, AMP, no online augmentation; ISIC2018 batch 8, learning rate 3e-4; ACDC and Synapse batch 16, learning rate 1e-4.
- Loss weights unchanged: boundary 0.05, distillation 0.05, surface 1, gate alpha 0.25, GSL lambda 1 with the step-5 schedule. Decoder + surface uses boundary 0.05, surface 1, distillation 0, alpha 0.
- Partitions unchanged: documented manifests (ISIC2018 2,075 training and 519 validation images). No class weights in any new run.
- Code: `scripts/revision/train_run.py` and `scripts/comparators/train_comparator.py` run unmodified. `scripts/expansion/train_expansion.py` only widens the permitted seed range; all other protocol checks still apply. Source hashes are registered per run.

## 5. Fairness rules
- All four ISIC2018 arms get the same seeds, schedule, partitions, selection rule, and evaluator.
- No hyperparameter is tuned for any arm. Nothing is changed in response to results.
- No arm gets extra runs, restarts, or seeds. Every completed run is reported.
- The other comparators (Boundary Loss, Attention U-Net, LB-UNet) stay at three seeds and are not part of Experiment A.

## 6. Model selection and the test set
- Checkpoint: maximum validation DSC, strict-greater tie rule, on the validation partition only.
- The official ISIC2018 test set is evaluated once per run, after selection, and is never used for selecting a model, checkpoint, epoch, hyperparameter, threshold, or arm.
- No test metric of any new run will be inspected until all 40 runs are complete and pass the checks in section 10. Monitoring during training reads only epoch progress, validation curves, temperature, and failures.

## 7. Reproduction gate (passed before freezing)
`revision/expansion_gate/gate.json`: seed 1 of all four ISIC2018 arms, run through the new entry point for 4 epochs, reproduces the archived validation DSC at every epoch with difference 0.0. Training stopped before any test evaluation. This shows new seeds come from the same pipeline as seeds 1-3, including the U-Net and BDS-Lite runs originally launched differently. Per-image metrics also match between evaluation paths (60 images, difference 0).

Execution order: correction runs, Experiment B, then Experiment A by seed (4 then 5, and so on; U-Net, decoder, BDS-Lite, GSL within a seed), so that arms stay balanced if training is interrupted. Thermal stops resume from the saved epoch through the existing supervisor. A technical failure is resumed or reported, never replaced by a different seed. Exclusion on the basis of results is forbidden.

## 8. Analysis (frozen in `analysis/revision/expansion_statistics.py`)
**Experiment A. Primary endpoint:** Boundary F1 on the official test set. **Secondary:** DSC, IoU, HD95, ASSD.

Six contrasts: decoder vs U-Net; BDS-Lite vs U-Net; BDS-Lite vs decoder; GSL vs U-Net; GSL vs BDS-Lite; GSL vs decoder.

The unit of replication is the training run. For each contrast: difference of arm means of run means, Welch standard error over runs combined with the standard error over test images, t interval with Welch df, and two-sided p. Mann-Whitney on run means is a sensitivity check. Seed numbers are not treated as pairs.

- **Confirmation (fresh seeds 4-10, 7 runs per arm).** Holm over the six contrasts within each metric. Seeds 1-3 are excluded because they prompted the questions.
- **Best estimate (all seeds 1-10, 10 runs per arm).** Reported as the effect size. Seeds were fixed in advance, not chosen by outcome.

**Experiment B. Primary endpoint:** interaction on Boundary F1, I = (decoder+surface - decoder) - (surface - U-Net), per dataset (seeds 1-3) and pooled by inverse variance with Cochran Q. **Secondary:** DSC, IoU, HD95, ASSD; the simple effects of each component; distillation+gate with and without the surface loss; ISIC2018 sensitivity with ten seeds in the U-Net and decoder cells. On ACDC the retrained U-Net is a low reference, so its interaction is read with that caveat.

**Component model.** The additive main-effects model is refitted with all eight cells and the ISIC2018 ten-seed arms. It remains exploratory. Registered Holm families from `prereg_revision.md` remain reported.

**Exploratory, not confirmatory.** The transfer probe (`analysis/revision/boundary_transfer_probe.py`) may be repeated on the ten-seed arms.

## 9. Interpretation criteria (fixed before results)
Effect labels use the registered SESOI (0.01 for DSC, IoU, Boundary F1; 2 px for HD95; 0.5 px for ASSD):

| Label | Rule on the 95% interval |
|---|---|
| relevant gain / relevant harm | wholly beyond +SESOI / -SESOI |
| gain / harm, size unclear | excludes zero, overlaps the threshold |
| no relevant effect | wholly inside +/-SESOI |
| undetermined | includes zero and extends past the threshold |

Status of each Experiment A contrast:

| Status | Rule | What the manuscript may say |
|---|---|---|
| **Confirmed** | Fresh-seed Holm p < 0.05, interval excludes zero, same sign as all seeds | The difference is real in this setting; its size follows the all-seed label |
| **Supported, not confirmed** | All-seed interval excludes zero, fresh seeds do not confirm | Suggestive only; not stated as an established effect |
| **Not established** | Neither of the above | No conclusion about a difference. Evidence of absence is claimed only if the label is "no relevant effect" |

Consequences for the paper, whichever way the results fall:

| Result | Consequence |
|---|---|
| Decoder vs U-Net confirmed gain | Report the decoder as raising Boundary F1 on ISIC2018, with its size. Not extended to other datasets |
| Decoder vs U-Net not established | Report as unresolved; drop any recommendation that relies on a decoder benefit |
| BDS-Lite vs U-Net confirmed gain | Report it, with cost. Attribution to distillation still requires the BDS-Lite vs decoder contrast |
| BDS-Lite vs decoder confirmed either way | Report that the extra components add to or subtract from the decoder |
| BDS-Lite vs decoder not established | State that added value over the decoder is unresolved, with the interval |
| GSL contrasts | Reported with equal prominence, in either direction |
| HD95 "no relevant effect" | Reported as evidence of no relevant HD95 change |
| Interaction interval excludes zero | Report antagonism or synergy on that dataset or pooled |
| Interaction inside +/-SESOI | Report additivity as supported |
| Interaction otherwise | Report as unresolved. The claim that the surface loss offsets the decoder must then be dropped or stated as a hypothesis |

Narrative 2 is retained for all outcomes, because its claim concerns what components do. The question-form title is retained provided the text does not imply a performance improvement that was not confirmed.

**Power, stated in advance.** With a run SD of 0.015 to 0.022, the 95% half-width on Boundary F1 is about 0.018 to 0.026 for 7 runs per arm and 0.014 to 0.021 for 10. Effects near 0.016 may therefore remain unconfirmed, and "no relevant effect" on Boundary F1 is unlikely to be reachable. With 3 seeds per cell the interaction half-width is about 0.05 or more per dataset, so "unresolved" is a likely outcome for Experiment B. These limits will be stated if they apply.

## 10. Validation before inspecting test results
For all 40 runs: 150 epochs completed; artifacts and hashes match; config and source hashes match this registration; selected epoch equals the first maximum of the validation curve; seeds are distinct; manifests match; per-image test metrics recomputed independently for a sample of runs; the first epochs of each arm differ across seeds. The check script and its output are retained.

## 11. Amendments
None.
