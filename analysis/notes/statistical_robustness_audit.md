# Statistical Robustness Audit

## Seed-level limitation

There are three paired seeds per method. For a two-sided exact Wilcoxon signed-rank test with n=3, the smallest attainable p-value is 0.25. No seed-level comparison survives Holm correction. Consequently, seed tests can document direction consistency but cannot support a statement of statistical performance leadership.

## Why image-level tests need clustering

ISIC2018 validation images are independent image units under the locked split. ACDC slices repeat patients and Synapse slices repeat volumes. Treating every slice-seed row as independent inflates sample size; averaging seeds does not remove within-patient or within-volume dependence. The rescue analysis therefore averages predictions across seeds per slice and resamples/tests cluster means: 519 ISIC2018 images, 20 ACDC patients, and 12 Synapse cases.

## Cluster-level results

Oriented effects are positive when favorable.

| Dataset | Metric | Cluster mean | 95% cluster-bootstrap CI | Holm-adjusted signed-rank p |
|---|---|---:|---:|---:|
| ISIC2018 | Dice | -0.00026 | [-0.00460, 0.00370] | 0.00292 |
| ISIC2018 | Boundary F1 | +0.00769 | [0.00057, 0.01464] | 0.00292 |
| ISIC2018 | HD95 | +1.0409 px | [0.2476, 1.8092] | 0.00292 |
| ISIC2018 | ASSD | +0.2207 px | [-0.0119, 0.4383] | 0.000098 |
| ACDC | Dice | -0.00395 | [-0.01255, 0.00390] | >0.14 |
| ACDC | HD95 | -1.0175 | [-1.9705, -0.1999] | 0.160 |
| Synapse | Dice | +0.00197 | interval crosses 0 | 1.0 |
| Synapse | HD95 | -0.1528 | interval crosses 0 | 1.0 |
| Synapse | Boundary F1 | +0.00202 | interval crosses 0 | 1.0 |

The signed-rank test concerns paired location/rank behavior, while the bootstrap interval shown here concerns the cluster-mean effect. Their conclusions can differ, as seen for ISIC2018 Dice and ASSD. Both must be reported rather than selecting the more favorable statistic.

## Effect size and sensitivity

`cluster_level_statistics.csv` reports sign-based effect sizes, `(n_favorable - n_unfavorable) / n_nonzero`. `outlier_sensitivity.csv` reports full, 1%, and 5% trimmed means plus the share of absolute change contributed by the largest cases. The ISIC2018 HD95 result is directionally stable after trimming. ISIC2018 Dice changes sign after trimming. ACDC distance regressions shrink after trimming, while Synapse HD95 does not disappear. These patterns argue for distribution plots and case examples alongside means.

## Recommended inferential hierarchy

1. Three-seed means and standard deviations describe training variability.
2. Case-level seed-averaged deltas describe paired effects.
3. Cluster bootstrap intervals use the patient/volume as the resampling unit.
4. Cluster-level signed-rank tests, Holm-adjusted within dataset, are secondary.
5. Slice-level tests may appear only as sensitivity analyses labeled non-independent.
6. A mixed-effects model is optional if its random-effects structure is preregistered; with 12 Synapse cases it should not replace transparent cluster summaries.

## Safe manuscript wording

“Across three training seeds, no seed-level comparison was conclusive after Holm correction. We therefore report seed-averaged paired case differences and cluster bootstrap intervals, using patients for ACDC and volumes for Synapse. ISIC2018 shows a small favorable Boundary F1 and HD95 shift in the locked validation set; ACDC and Synapse do not show a consistent cross-metric advantage.”

Avoid “significantly outperforms.” For ISIC2018, a precise alternative is: “The paired signed-rank analysis detected a distributional shift after Holm correction, while the estimated mean effect remained small and some mean bootstrap intervals included zero.”
