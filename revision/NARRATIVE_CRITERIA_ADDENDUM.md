# Addendum to the expansion protocol: criteria for a stronger narrative

Written and committed on 2026-11-06 with 31 of 40 expansion runs complete and before any expansion test metric was inspected. Git history is the timestamp. `EXPANSION_PROTOCOL.md` is hash-checked while the queue runs, so it is not edited; Section 11 of that file will point here once the queue has finished.

## Why this exists
Protocol Section 9 retained Narrative 2 for every outcome. The author has since asked for the manuscript to be reassessed after the experiments, including whether the evidence supports a stronger, more positive narrative that restores parts of the original one. That changes a decision the protocol had fixed, so the rule for making it is fixed here, before the results are known. Everything else in the protocol stands: runs, analysis, labels, statuses, validation.

## Claim tiers
The tier is read from the frozen outputs. "Confirmed" and the effect labels have their Section 9 meanings. All ISIC2018 contrasts use Boundary F1 unless stated.

| Tier | Claim the manuscript may lead with | Required evidence |
|---|---|---|
| 0 | Narrative 2: what the components do and do not do | Default |
| 1 | BDS-Lite raises boundary accuracy on ISIC2018 relative to a matched U-Net | BDS-Lite vs U-Net Confirmed; all-seed interval excludes zero; DSC not "harm" or "relevant harm"; HD95 not "relevant harm" |
| 2 | The added parts contribute beyond the training-only decoder on ISIC2018 | Tier 1, and BDS-Lite vs decoder Confirmed in favour of BDS-Lite |
| 3 | Method-paper framing: BDS-Lite as a recommended way to train a compact segmenter | Tier 2, and BDS-Lite vs GSL not Confirmed against BDS-Lite on Boundary F1, and a same-direction result on at least one further dataset from a confirmatory analysis with at least 7 fresh seeds per arm |

If only the decoder contrast is Confirmed, the positive claim concerns the training-only boundary decoder. It is not credited to BDS-Lite as a whole.

## Elements that stay out at every tier
- The original +0.031 Boundary F1 as a lead result, because that model set used the ISIC2018 protocol now corrected.
- Removing or softening the ISIC2018 correction in the abstract.
- Any statement that BDS-Lite exceeds GSL, unless a Confirmed contrast shows it, and then only for the metric and dataset shown.
- Parameter count as evidence of effectiveness. Reported overheads stay as measured, including the inference time and memory increases.
- A claim on ACDC or Synapse from three seeds alone beyond the labels already reported.
- Clinical, 3D or hardware-deployment claims.

## Rules for any additional experiment
Additional experiments are allowed only if they could change a tier or resolve a named open question. Each is written down and committed before training, with the following rules.

1. **No test-driven tuning.** Every choice of hyperparameter, checkpoint or variant uses validation data. The official test set is evaluated once per frozen configuration.
2. **Equal treatment.** If a BDS-Lite setting is tuned, the comparison arms receive the same search budget on the same validation data.
3. **Fresh seeds.** A setting chosen after seeing any result is confirmed on seeds not used to choose it. Seeds used for selection do not count towards confirmation.
4. **Separate family.** New contrasts form a new Holm family. They do not enter or alter the registered families, and the registered results stay primary.
5. **Reporting.** Every experiment started is reported, including those that fail or are abandoned. Results that were influenced by the expansion outcomes are labelled as post hoc in origin.
6. **No replacement.** A tuned variant is reported beside the frozen BDS-Lite and does not replace it as "the method" without the author's decision.
