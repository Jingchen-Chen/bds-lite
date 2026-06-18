# Table 3. Cluster-aware statistical summary (BDS-Lite vs U-Net)

Cluster-aware analysis with images as units for ISIC2018, patients for ACDC,
and cases for Synapse. Deltas are oriented so positive values are favorable for
BDS-Lite. CI columns are cluster bootstrap 95% intervals; p-values are paired
signed-rank with Holm correction. All values from
`results/midterm_q2_rescue/analysis/cluster_level_statistics.csv`. Seed-level
Wilcoxon tests at n=3 are separately reported and all Holm-corrected to 1.0
(`results/final_manuscript_audit_v2/seed_level_wilcoxon_summary.csv`).

| Dataset | Unit (n) | Metric | Oriented Δ | 95% CI | Holm p | Decision |
|---|---|---|---:|---|---:|---|
| ISIC2018 | image (519) | DSC | -0.0003 | [-0.0046, 0.0037] | 0.0029 | significant |
| ISIC2018 | image (519) | IoU | 0.0021 | [-0.0028, 0.0065] | 0.0009 | significant |
| ISIC2018 | image (517) | HD95 | 1.0409 | [0.2476, 1.8092] | 0.0029 | significant |
| ISIC2018 | image (517) | ASSD | 0.2207 | [-0.0119, 0.4383] | 0.0001 | significant |
| ISIC2018 | image (519) | Boundary F1 | 0.0077 | [0.0006, 0.0146] | 0.0029 | significant |
| ACDC | patient (20) | DSC | -0.0040 | [-0.0125, 0.0039] | 1.0 | not significant |
| ACDC | patient (20) | IoU | -0.0033 | [-0.0122, 0.0051] | 1.0 | not significant |
| ACDC | patient (20) | HD95 | -1.0175 | [-1.9705, -0.1999] | 0.1600 | not significant |
| ACDC | patient (20) | ASSD | -0.4036 | [-0.7932, -0.1105] | 0.1479 | not significant |
| ACDC | patient (20) | Boundary F1 | -0.0048 | [-0.0142, 0.0036] | 1.0 | not significant |
| Synapse | case (12) | DSC | 0.0020 | [-0.0141, 0.0166] | 1.0 | not significant |
| Synapse | case (12) | IoU | 0.0014 | [-0.0139, 0.0159] | 1.0 | not significant |
| Synapse | case (12) | HD95 | -0.1528 | [-0.7685, 0.4168] | 1.0 | not significant |
| Synapse | case (12) | ASSD | -0.0964 | [-0.3756, 0.1252] | 1.0 | not significant |
| Synapse | case (12) | Boundary F1 | 0.0020 | [-0.0160, 0.0189] | 1.0 | not significant |

Interpretation: small ISIC2018 cluster-level shifts reach Holm significance on
boundary and distance metrics (the DSC mean delta is near zero and its CI
crosses zero). No ACDC or Synapse metric reaches cluster-level significance,
and the ACDC mean distance directions are unfavorable. These cluster-level
results take precedence over slice-level p-values.
