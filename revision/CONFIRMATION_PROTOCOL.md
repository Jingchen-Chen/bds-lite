# Experiment C: confirmation of the boundary decoder, frozen protocol

Status: frozen before training. Written after the expansion results were analysed, so the questions here are prompted by those results. That is why none of the seeds that prompted them are used for confirmation. Governed by `revision/NARRATIVE_CRITERIA_ADDENDUM.md`, which permits an experiment only if it could change a tier or resolve a named open question. Git history is the timestamp.

## 1. What prompted this
Frozen expansion results on Boundary F1, positive favours the first arm:

| Contrast | Data | Estimate [95% interval] | Status |
|---|---|---|---|
| Decoder vs U-Net | ISIC2018, seeds 1-10 | +0.021 [+0.001, +0.040] | Supported, not confirmed (fresh Holm p = 0.26) |
| BDS-Lite vs U-Net | ISIC2018, seeds 1-10 | -0.011 [-0.032, +0.010] | Not established |
| BDS-Lite vs decoder | ISIC2018, seeds 1-10 | -0.032 [-0.052, -0.012] | Confirmed against BDS-Lite |
| GSL vs decoder | ISIC2018, seeds 1-10 | -0.031 [-0.052, -0.009] | Confirmed against GSL |
| Decoder vs U-Net | Pooled, three datasets, seeds 1-3 | +0.016 [+0.005, +0.027] | Exploratory |

Tier 1 of the addendum is not met, so the paper cannot claim that BDS-Lite raises boundary accuracy on ISIC2018. The only positive candidate left is the training-only boundary decoder, and it is unconfirmed. Whether the ISIC2018 deficit of BDS-Lite relative to the decoder holds elsewhere is also unknown: ACDC and Synapse have three seeds per arm.

## 2. Questions
1. **Primary.** Does the training-only boundary decoder raise Boundary F1 relative to a matched U-Net, on seeds that played no part in suggesting it?
2. Does BDS-Lite add to or subtract from the decoder on ACDC and Synapse? This also tests fairly whether the ISIC2018 deficit is specific to that dataset.
3. How do the decoder and BDS-Lite compare with GSL on ACDC and Synapse?

## 3. Runs (70)
| Dataset | Arms | Seeds | Runs |
|---|---|---|---|
| ACDC | U-Net, decoder, BDS-Lite, GSL | 4-10 | 28 |
| Synapse | U-Net, decoder, BDS-Lite, GSL | 4-10 | 28 |
| ISIC2018 | U-Net, decoder | 11-17 | 14 |

The queue is ordered by seed, so arms stay balanced if it is interrupted. All 70 runs are completed before any test metric is read. There is no interim look and no early stopping on results.

## 4. Configurations and fairness
Every arm uses the existing recipe for its dataset; only the seed changes. No hyperparameter is tuned. Builders are in `analysis/revision/confirmation_arms.py`. On ACDC all four arms are unweighted, including the decoder, which uses the corrected arm. Selection is the first maximum of validation DSC, as before. The official test sets are used only for the final evaluation of each frozen run.

Because seeds 1-3 of some ACDC and Synapse arms came from recovered runs, a reproduction gate is passed first: seed 1 of each rebuilt arm must replay the archived validation trajectory exactly for at least 3 epochs (`scripts/confirmation/reproduction_gate.py`).

## 5. Analysis (frozen in `analysis/revision/confirmation_statistics.py`)
The estimator is the expansion contrast: difference of arm means of run means, Welch standard error over runs combined with the standard error over test units, t interval. Metric values are oriented so positive favours the first arm.

**Primary (one test).** Decoder vs U-Net on Boundary F1, per dataset on fresh seeds, pooled by inverse variance across the three datasets. Two-sided p from the normal approximation. Cochran Q is reported. Sensitivity: DerSimonian-Laird random effects, leave-one-dataset-out, and Mann-Whitney on run means per dataset.

**Secondary.** Thirteen fresh-seed contrasts, Holm within each metric: all six arm pairs on ACDC and on Synapse, plus decoder vs U-Net on ISIC2018. Metrics are Boundary F1, DSC, IoU, HD95 and ASSD. Pooled ACDC and Synapse estimates for BDS-Lite vs decoder, BDS-Lite vs U-Net and GSL vs decoder are reported without a test.

**Best estimates.** The same contrasts on all available seeds (ISIC2018 U-Net and decoder 1-17, other ISIC2018 arms 1-10; ACDC and Synapse 1-10), pooled across three datasets. These give effect sizes and are not confirmatory.

Validation before any test metric is read follows Section 10 of `EXPANSION_PROTOCOL.md`.

## 6. Interpretation (fixed now)
Labels are those of `EXPANSION_PROTOCOL.md` Section 9, with the same SESOI.

| Result | What the manuscript may say |
|---|---|
| Primary: p < 0.05, interval above zero | The decoder raises Boundary F1 across the studied datasets, with size from the label of the all-seed pooled estimate. This is the paper's positive finding |
| ... with Q p < 0.10, or a leave-one-out interval including zero | As above, and the effect must be described as uneven or as dependent on the named dataset |
| Primary: not significant | The decoder gain is reported as suggested on ISIC2018 and not confirmed; no recommendation to adopt it |
| Primary: p < 0.05, interval below zero | The decoder lowers Boundary F1 in fresh seeds; the earlier estimates are reported as not replicated |
| Per-dataset contrast: fresh Holm p < 0.05, interval excludes zero | Confirmed on that dataset only |
| BDS-Lite vs decoder not established on ACDC or Synapse | The ISIC2018 deficit is not extended to that dataset |
| BDS-Lite vs U-Net Confirmed gain on ACDC or Synapse | Reported for that dataset only. Tiers 1-3 of the addendum concern ISIC2018 and stay unmet |
| GSL contrasts | Reported with equal prominence, in either direction |

## 7. Power, stated in advance
From seeds 1-3, run SDs on Boundary F1 are about 0.008 (ACDC), 0.027 (Synapse) and 0.020 (ISIC2018). With 7 runs per arm and test-unit error, the pooled standard error is about 0.005, dominated by ACDC. A true effect of 0.016 would be detected with power near 0.9, and an effect of 0.01 with power near 0.5. Per-dataset intervals on Synapse and ISIC2018 will remain wide, about +/-0.03.

## 8. Considered and not run
- **Tuning BDS-Lite's loss weights or gate.** The distillation and gate effects are near zero with undetermined intervals, so tuning would add researcher choices to a stack that shows no measured benefit. A fair comparison would also need matched search budgets for every arm.
- **Tuning the decoder weight.** Same fairness cost. It should follow, not precede, confirmation that the decoder helps.
- **More seeds for the interaction.** With three seeds per cell the interaction is unresolved. Resolving it needs about 30 further M1 and M7 runs and would not change which components are recommended.

## 9. Amendments
None.
