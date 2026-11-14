# Matched-checkpoint gate identity findings

The intervention replaces the inference gate with unit modulation on the same trained models. It is not training-time M5. Architecture/scalar settings, exact state/alias identity and operator logits were validated for every evaluated cohort. Statistical analysis is a revision-stage exploratory amendment registered after prediction generation began, before comparative effect inspection.

| Dataset | Metric | Paired units | Identity−full mean | 95% bootstrap interval | Holm p |
|---|---|---:|---:|---|---:|
| isic2018 | dsc | 1000 | -0.000008 | [-0.000176, 0.000172] | 5.13e-16 |
| isic2018 | iou | 1000 | -0.000251 | [-0.000419, -0.000078] | 3.22e-17 |
| isic2018 | boundary_f1 | 1000 | -0.001410 | [-0.001605, -0.001209] | 5.21e-55 |
| isic2018 | hd95 | 999 | 0.093631 | [0.028168, 0.165542] | 7.14e-12 |
| isic2018 | assd | 999 | 0.028934 | [0.013878, 0.043198] | 6.39e-23 |
| acdc | dsc | 20 | 0.000404 | [0.000255, 0.000555] | 0.000524521 |
| acdc | iou | 20 | 0.000392 | [0.000230, 0.000556] | 0.00128937 |
| acdc | boundary_f1 | 20 | 0.000438 | [0.000230, 0.000658] | 0.00429726 |
| acdc | hd95 | 20 | 0.047222 | [0.010175, 0.093402] | 0.265453 |
| acdc | assd | 20 | 0.003753 | [-0.000041, 0.007983] | 0.265453 |
| synapse | dsc | 12 | -0.000087 | [-0.000216, 0.000052] | 1 |
| synapse | iou | 12 | -0.000088 | [-0.000217, 0.000050] | 1 |
| synapse | boundary_f1 | 12 | -0.000094 | [-0.000218, 0.000039] | 1 |
| synapse | hd95 | 12 | 0.001470 | [-0.002720, 0.006332] | 1 |
| synapse | assd | 12 | 0.001710 | [-0.002721, 0.009112] | 0.646973 |

Pixel maps are not generally identical, and several rank-test differences are measurable. Every primary endpoint interval lies inside the already specified engineering SESOI band in these tested models. This supports a qualified practical-equivalence description conditional on trained seeds, not exact invariance, clinical equivalence, general removability or zero deployment overhead.

The mean-effect bootstrap and Wilcoxon rank test assess different features of a potentially asymmetric difference distribution; an interval containing zero can coexist with a small rank-test p-value. Do not alter either output to force agreement. Shared finite-cell eligibility and all-cell diagonal-penalty sensitivities remain explicit.

Sources: tables/gate_identity_effects.csv, gate_identity_by_seed.csv, gate_identity_per_unit_seed.csv and gate_identity_changed_pixels.csv. Provenance: validation/gate_identity_lineage.csv and per-run prediction-recovery files. All changed masks remain saved. This evidence does not complete the training M5 ablation.
