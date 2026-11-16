# Stacked-volume findings — index-grid evaluation

Completed cohorts: 20; volumes: 520; volume-class rows: 2160. Class/frame and seed averaging uses patient/case units. Named anatomy needs review.

| Dataset | Endpoint | Paired n | Full−U mean | 95% bootstrap interval | Holm p |
|---|---|---:|---:|---|---:|
| acdc | dsc | 20 | 0.014232 | [0.002749, 0.029916] | 0.057701 |
| acdc | hd95 | 20 | 1.779490 | [0.661869, 3.017545] | 0.106339 |
| acdc | assd_pooled | 20 | 0.262131 | [0.045082, 0.510307] | 0.106339 |
| synapse | dsc | 12 | 0.013765 | [0.006717, 0.020477] | 0.020508 |
| synapse | hd95 | 11 | -1.901327 | [-7.571471, 3.683321] | 0.929688 |
| synapse | assd_pooled | 11 | -0.165651 | [-0.777175, 0.500035] | 0.929688 |

Both datasets have higher mean volume Dice for the full recipe. ACDC surface distances are worse on average; Synapse distance differences remain uncertain. Holm-adjusted volume-Dice evidence is stronger on Synapse than ACDC. Neither Dice interval lies entirely beyond the chosen engineering SESOI. These findings do not replace or erase different slice-level outcomes.

Synapse surface complete-case inference excludes one case; all-case diagonal-penalty sensitivity is retained in volume_primary_statistics.csv. Those penalties are not measured distances. No millimetre-based practical-equivalence statement is made.

Synapse uses original H5 labels and nearest-output zoom; processed inputs were bilinear resized rather than the official reference cubic input zoom. ACDC uses the observed processed-grid stacks, without verified native spacing or ED/ES assignment. No EF/mass results are reported. All Synapse per-class rows retain numeric IDs until anatomical names are verified.

All per-volume values and source/output hashes are under volume_results/. KDTree distances matched explicit EDT distances in synthetic3D checks. The full raw reference-empty convention and robust alternative are both saved. Four real-volume EDT spot-checks passed; complete manuscript integration remains pending.
