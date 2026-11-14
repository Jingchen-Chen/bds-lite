# Frozen ACDC teachers on official patients 101-150 — result

Post hoc, descriptive, inference only (protocol `PROTOCOL.md`, committed `0071208` before any
new teacher inference). Patient-level aggregation as in the paper. No significance tests. Source:
`summary.json`.

## Sanity checks (all passed)

176 artifact hashes matched the one-time evaluation records. All 88 student runs, recomputed from
their saved predictions with the unchanged evaluator, equal their saved per-case files. Recomputed
group BF1: D 0.7796910 (paper 0.7797), D+KD-band 0.7975114 (0.7975), D_aug 0.8468818 (0.8469),
D+KD-band_aug 0.8431735 (0.8432); all six groups equal the frozen full-precision means.

## Teachers

| Teacher | Params | DSC | IoU | BF1 | HD95 (px) | ASSD (px) | BF1 RV | BF1 Myo | BF1 LV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| External (smp U-Net/ResNet-34, seed 100) | 24,436,804 | 0.8146 | 0.7426 | 0.8778 | 2.608 | 1.018 | 0.8216 | 0.9250 | 0.9155 |
| Compact (student architecture, D recipe, seed 100) | 2,197,493 | 0.7311 | 0.6536 | 0.7735 | 7.850 | 2.591 | 0.6671 | 0.8415 | 0.8513 |

Both teachers' new predictions are pixel-identical to their earlier one-time evaluations
(external: `results/acdc_confirm_eval_posthoc/`, compact: `results/followup_eval/`).

## Teacher vs student groups (Boundary F1)

Paired difference: teacher minus the group's per-patient mean over runs, averaged over 50
patients; 95% percentile bootstrap CI (10,000 patient resamples, seed 20261111).

| Group | Runs | Run mean (min–max) | Runs > external | External − group [95% CI] | Runs > compact | Compact − group [95% CI] |
|---|---:|---|---:|---|---:|---|
| D | 17 | 0.7797 (0.7645–0.7949) | 0 | +0.0981 [+0.0804, +0.1158] | 14 | −0.0062 [−0.0160, +0.0037] |
| D+KD-band | 17 | 0.7975 (0.7842–0.8078) | 0 | +0.0803 [+0.0636, +0.0967] | 17 | −0.0240 [−0.0356, −0.0127] |
| D+KD-full | 17 | 0.7903 (0.7775–0.8036) | 0 | +0.0875 [+0.0714, +0.1037] | 17 | −0.0168 [−0.0269, −0.0069] |
| D+KD-band_c | 17 | 0.7915 (0.7765–0.7990) | 0 | +0.0863 [+0.0703, +0.1020] | 17 | −0.0181 [−0.0295, −0.0070] |
| D_aug | 10 | 0.8469 (0.8376–0.8575) | 0 | +0.0309 [+0.0158, +0.0454] | 10 | −0.0734 [−0.0908, −0.0567] |
| D+KD-band_aug | 10 | 0.8432 (0.8353–0.8531) | 0 | +0.0346 [+0.0186, +0.0497] | 10 | −0.0697 [−0.0877, −0.0525] |

## Pre-specified interpretation

External teacher BF1 0.8778 ≥ D_aug mean 0.8469 → **Outcome B: "This explanation is not
supported."**
