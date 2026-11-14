# Evidence addendum 2026-11-12 — review-round post hoc analyses

Versioned addendum to `revision/CLAIM_BOUNDARY_v4.md` and `revision/BDS_LITE_EVIDENCE_FREEZE.md`,
which require "a versioned evidence addendum and appropriate authorization" before a new analysis
enters the manuscript. Neither file is modified. Every registered protocol, estimator, endpoint,
decision and C01–C16 reading is unchanged.

## Authorization

The author's instruction of 2026-11-12 (two internal reviews of the revised manuscript plus a
brief): no new training; re-evaluation, statistical tests, error analysis and figures from existing
checkpoints, predictions and raw data are allowed; "特别注意统计方法、体积级评估的可比性、交互检验的
协方差处理". This supersedes the decision of 2026-11-11 not to compute the
augmentation-by-distillation interaction.

## Source

- Script: `analysis/revision/posthoc_review_20261112.py` (reads saved per-case outputs only; the
  official ACDC files are hash-checked through `analysis/followup/analyze.arm_runs`).
- Outputs: `revision/posthoc_review_20261112/{interaction,pooling_sensitivity,run_sd,
  missed_false_alarm}.csv`, `summary.json` (input and output SHA-256).
- Formatting: `analysis/revision/make_review_assets.py` (keys `RV.*`, `generated/numbers_review.tex`,
  `tab_s_review_*.tex`, `tab_errors_main.tex`, `tab_published_review.tex`).
- No model was trained or executed.

## Added evidence (C17–C20)

| ID | Supported statement and evidence | Statistical/practical reading and scope |
|---|---|---|
| C17 | Post hoc augmentation-by-distillation interaction on official ACDC patients 101–150, Boundary F1: I = (KD-band − D) − (KD-band_aug − D_aug) = +0.0215 [+0.0112, +0.0319], p = 1.4e-4 (registered-form four-arm SE; 17/17/10/10 runs). Crossed-model interval [+0.0124, +0.0306]; hierarchical bootstrap [+0.0111, +0.0322]; seeds 201–210 in all arms +0.0194 [+0.0075, +0.0314]. DSC +0.0207 [+0.0113, +0.0301]; IoU +0.0259 [+0.0161, +0.0357]; HD95 +1.05 px [−0.36, +2.47]; ASSD +0.32 px [−0.03, +0.68]. `interaction.csv` | Not registered; specified after both simple contrasts were known; patients previously examined; unequal run counts. The unit term uses the per-patient double difference, so shared patient effects are not counted twice. May be described as: the distillation contrast differed between the two regimes in this cohort. Does not establish that augmentation eliminates a distillation benefit (C13's E3 reading, undetermined, stands), and does not extend to other augmentations, datasets or teachers |
| C18 | Pooling-rule sensitivity of the stage-1 pooled Boundary F1 summaries (three runs per arm). Decoder − U: registered +0.0159 [+0.0053, +0.0265]; inverse variance with t interval [+0.0029, +0.0289]; equal weights +0.0156 [−0.0014, +0.0326]; without ACDC +0.0157 [−0.0058, +0.0373]; ACDC weight 76%. Transfer without surface: registered +0.0092 [−0.0010, +0.0194]; ACDC weight 84%. DerSimonian–Laird equals fixed effect (tau² = 0). `pooling_sensitivity.csv` | Sensitivity analysis only; the registered summaries and readings stand (C01). ACDC's weight follows from its smaller standard error, not from one arm's SD alone (the contrast pools the within-cell variance of both cells). Supports calling the pooled decoder gain exploratory and pooling-dependent; supports no new positive claim |
| C19 | Between-run SD of Boundary F1 in every arm with ≥10 runs: ISIC2018 0.0186–0.0257 (4 arms, 10 runs); official ACDC 0.0060–0.0086 (6 arms, 10 or 17 runs). `run_sd.csv` | Descriptive. The earlier text range "0.0069 to 0.0257" omitted two ACDC arms (0.0063, 0.0060) and mixed datasets; ranges are now given per dataset |
| C20 | Missed structures and false alarms per run (slice–class pairs). ACDC internal: U 5.7% missed / 28.6% false alarms of 85 empty pairs; D 3.5% / 42.0%. Synapse: about 10% / 10% in every arm; 73% of pairs have an empty reference. Official ACDC: D 4.6% / 41.3%; KD-band 4.2% / 44.3%; D_aug 3.5% / 30.7%; KD-band_aug 3.8% / 31.9%. `missed_false_alarm.csv` | Descriptive, post hoc, untested. Explains part of the gap between per-slice and volume-level scores; no causal claim |

Also used, from existing frozen artifacts (no new computation): volume-level DSC of the
matched-protocol U-Net, 0.842 (ACDC) and 0.683 (Synapse), `revision/tables/families/
volume_method_summary.csv` (keys `VOL.*.M0.dsc`), as descriptive context in main-text Table 8 under
C12 (no cross-paper ranking; empty-mask conventions differ from the published evaluators).

## Wording limits

- Use "post hoc" with every mention of C17, C18 and C20.
- C17: "differed between the regimes" / "larger without augmentation than with it"; never
  "augmentation removed/eliminated the effect".
- C18 does not downgrade a registered label; it qualifies how the pooled decoder gain is described
  (exploratory, pooling-dependent, not confirmed on fresh ISIC2018 seeds).
- Control contrasts C1 and C4 remain unresolved (C06): write "a gain remained without the band /
  with a compact teacher", not "did not depend on" or "does not need".

## Presentation change of 2026-11-13 (no new evidence)

At the author's request the main text, the response letter and the editor letter use three
decimals and one result vocabulary; the supplement keeps four decimals and the registered labels.
Mapping (also stated at the top of the supplement):

| Registered label (supplement) | Main-text word |
|---|---|
| Stage-3 registered success ("of unclear size") | confirmed |
| Fresh-seed "confirmed" | confirmed |
| "gain/harm of unclear size", "supported, not confirmed", Holm p < 0.05 | detected |
| "undetermined", "unresolved", "not established" | not detected (stated as not evidence of no effect) |

C04 limit kept: the rounded interval prints as 0.010 to 0.026, so the main text quotes the
unrounded lower bound 0.0098 and says the registered reading leaves open whether the gain exceeds
0.01. Rounding: `analysis/revision/make_main_round3.py` (`numbers_round3.tex`, `r3_*.tex`); ties
at the fourth decimal are resolved from the exact source values listed in the script.
The post hoc interaction (C17) was moved out of the Abstract; it stays in Results, S24 and the
response letter.
