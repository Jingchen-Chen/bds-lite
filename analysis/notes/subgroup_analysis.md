# Subgroup and Applicability Analysis

## Method

Ground-truth masks were used to derive target area, area ratio, boundary length, boundary complexity (`boundary length / sqrt(area)`), compactness, and the number of foreground classes. Continuous properties were split into dataset-specific thirds. Metrics were averaged over three seeds per sample. Bootstrap intervals resampled independent clusters: images for ISIC2018, patients for ACDC, and cases for Synapse.

Tables:

- `subgroup_summary.csv`
- `property_delta_correlations.csv`
- `per_class_summary.csv`
- `per_class_case_metrics.csv`

Figure: `../figures/per_class_oriented_deltas.png`.

## ISIC2018

The target-area result is the clearest exploratory applicability signal:

- Low-area lesions: mean Dice delta -0.0121 and Boundary F1 delta -0.0040.
- Medium-area lesions: Boundary F1 delta +0.0160 with a cluster-bootstrap interval above zero.
- High-area lesions: Dice +0.0063 and Boundary F1 +0.0111; their bootstrap intervals are favorable.

Higher boundary complexity is associated with larger Boundary F1 deltas: Spearman rho 0.161 (p=0.00023). The high-complexity third has mean Boundary F1 delta +0.0190, but its distance deltas are near zero. The evidence therefore supports a narrow hypothesis that boundary supervision may help local boundary agreement on medium/large or more complex skin lesions; it does not support a general complex-shape distance claim.

## ACDC

No subgroup is stable enough to define a clear applicability condition. Some larger-target groups have small favorable overlap means, while distance results remain unfavorable in several groups. Only 20 independent patients are available, intervals are wide, and slice-level correlations cannot be treated as patient-level evidence.

Class-level three-seed slice means show heterogeneous behavior:

| Class | Dice delta | Boundary F1 delta | Oriented HD95 delta | Reading |
|---|---:|---:|---:|---|
| RV | -0.0012 | -0.0026 | -0.8126 | Mostly unfavorable |
| Myocardium | -0.0051 | -0.0041 | -1.2016 | Clearest weak class |
| LV | +0.0052 | +0.0054 | -0.1873 | Overlap/boundary gain with distance cost |

This pattern is better used as a failure mechanism than as a favorable claim.

## Synapse

The multi-organ results vary by organ:

- Liver has the most coherent favorable means: Dice +0.0292, Boundary F1 +0.0325, oriented HD95 +1.97, and oriented ASSD +0.74.
- Pancreas has small favorable means across all four reported directions.
- Gallbladder and right kidney are unfavorable across overlap, boundary, and distance means.
- Stomach has favorable Dice/Boundary F1 but HD95 -1.50 and ASSD -0.28, directly illustrating metric conflict.

The high-complexity third has a favorable Dice mean (+0.0169) but unfavorable distance means. Only 12 independent cases underpin Synapse cluster intervals. These findings are hypothesis-generating and require confirmation with case-level organ summaries or a preregistered repeat.

## Single-organ versus multi-organ interpretation

ISIC2018 provides the clearest boundary/distance alignment. ACDC and Synapse introduce class imbalance, empty per-slice classes, heterogeneous object scales, and multiple structures. The artifacts are consistent with greater instability in multi-class segmentation, but the datasets differ in more than class count; the analysis cannot isolate class count as the cause.

## Main-paper use

- Present ISIC2018 target-size and boundary-complexity results as exploratory applicability evidence.
- Present ACDC myocardium and Synapse organ heterogeneity as failure analysis.
- Use liver, gallbladder, kidney, and stomach to explain why macro averages hide materially different class behavior.

## Limitations

Subgroups were derived after observing the completed experiments, multiple comparisons were not powered as confirmatory tests, and empty-class exclusions change row counts. No subgroup should be presented as a validated decision rule. A future experiment must preregister property definitions and use patient/volume-level organ summaries.
