# Table 2. Matched three-seed main results (journal main table)

Mean ± seed standard deviation over three seeds under the locked protocol
(seeds, split manifests, 224 x 224 inputs, batch sizes, AMP policy, 150-epoch
schedule, maximum-validation-DSC checkpoint rule). Test data were not used for
model or checkpoint selection. Higher DSC, IoU, and Boundary F1 are favorable;
lower HD95 and ASSD are favorable. **Descriptive only — no significance is
claimed from three seed means.** U-Net and BDS-Lite rows are from
`results/final_manuscript_audit_v2/recomputed_main_test_means.csv`; U-Net+GSL
rows are from the Phase 16 matched rerun
`results/submission_blocker_resolution_v1/gsl_memory_bounded/matched_gsl_summary.csv`
(aggregated in `matched_gsl_main_comparison.csv`). This matched table replaces
the obsolete separate-family GSL table.

| Dataset / split | Method | DSC | IoU | HD95 | ASSD | Boundary F1 |
|---|---|---:|---:|---:|---:|---:|
| ISIC2018 / val | U-Net | 0.8830 ± 0.0052 | 0.8091 ± 0.0054 | 18.5128 ± 0.7860 | 5.9192 ± 0.1467 | 0.5520 ± 0.0062 |
| ISIC2018 / val | BDS-Lite | 0.8828 ± 0.0066 | 0.8112 ± 0.0093 | 17.4435 ± 1.5792 | 5.6753 ± 0.4499 | 0.5597 ± 0.0124 |
| ISIC2018 / val | U-Net+GSL | 0.8662 ± 0.0013 | 0.7902 ± 0.0005 | 19.7954 ± 0.8841 | 6.6595 ± 0.0766 | 0.5399 ± 0.0075 |
| ACDC / test | U-Net | 0.7893 ± 0.0021 | 0.7113 ± 0.0015 | 6.8762 ± 0.5255 | 2.2080 ± 0.2768 | 0.8313 ± 0.0019 |
| ACDC / test | BDS-Lite | 0.7847 ± 0.0067 | 0.7073 ± 0.0067 | 7.6230 ± 0.7645 | 2.4874 ± 0.0215 | 0.8267 ± 0.0064 |
| ACDC / test | U-Net+GSL | 0.7856 ± 0.0065 | 0.7038 ± 0.0069 | 7.0143 ± 1.4387 | 2.0902 ± 0.3609 | 0.8277 ± 0.0041 |
| Synapse / test | U-Net | 0.8544 ± 0.0109 | 0.8360 ± 0.0109 | 10.1985 ± 0.2650 | 3.7515 ± 0.0844 | 0.8608 ± 0.0106 |
| Synapse / test | BDS-Lite | 0.8573 ± 0.0049 | 0.8385 ± 0.0048 | 10.3413 ± 0.5116 | 3.8280 ± 0.1152 | 0.8632 ± 0.0046 |
| Synapse / test | U-Net+GSL | 0.8640 ± 0.0088 | 0.8454 ± 0.0086 | 10.1285 ± 0.2532 | 3.8567 ± 0.2242 | 0.8701 ± 0.0088 |

Direction summary (descriptive): GSL is lower than both comparators on ISIC2018
DSC/Boundary F1 and higher on ISIC2018 HD95/ASSD; near the comparators on ACDC;
and higher on Synapse DSC/Boundary F1 and lower on Synapse HD95, with higher
ASSD. The matched comparison supports no superiority claim in either direction.
