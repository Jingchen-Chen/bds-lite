# Table 5. Failure cases and limitation summary

Illustrative failure cases were selected by deterministic metric criteria
before visual review; they do not estimate prevalence. Limitations define the
scope of the paper.

| Item | Category | Observation | Evidence | Source |
|---|---|---|---|---|
| `ISIC_0016060` | failure case | unfavorable lesion case | DSC −0.223, Boundary F1 −0.218, large distance degradation | corrected draft Sec 5.5 |
| `patient052_frame01_slice_5` | failure case | favorable DSC with worse HD95/ASSD | overlap–distance conflict | corrected draft Sec 5.5 |
| `case0004_slice134` | failure case | marked Synapse distance failure | qualitative panel | corrected draft Sec 5.5 |
| ACDC overall | limitation | no stable advantage; unfavorable mean distance | cluster Holm p 1.0; HD95/ASSD oriented deltas negative | `cluster_level_statistics.csv` |
| Synapse distance | limitation | overlap/boundary up while HD95/ASSD worsen | HD95 10.1985→10.3413; ASSD 3.7515→3.8280 | `recomputed_main_test_means.csv` |
| GSL comparison | limitation | matched three-seed means mixed in both directions | Table 2 matched rows | `matched_gsl_main_comparison.csv` |
| Seed count | limitation | three seeds only; no Holm seed-level superiority | all seed-level Holm p 1.0 | `seed_level_wilcoxon_summary.csv` |
| Cluster sizes | limitation | ACDC 20 / Synapse 12; subgroup analyses exploratory | n_clusters | `cluster_level_statistics.csv` |
| Efficiency scope | limitation | no time-to-convergence or cross-hardware claim | single-environment profile | `training_cost_final_decision.md` |
| Generalization | limitation | public 2D benchmarks only; no clinical/3D conclusion | scope statement | candidate manuscript Sec 7 |
