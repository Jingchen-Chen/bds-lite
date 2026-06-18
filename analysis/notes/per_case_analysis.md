# Per-Case Improvement Analysis

## Method

For each dataset, predictions were paired by sample and seed, metric values were
averaged across the three available seeds per case, and deltas were computed as
`BDS-Lite - U-Net`. For HD95 and ASSD, the analysis also reports an oriented
delta (`U-Net - BDS-Lite`) so that positive values always indicate a favorable
direction. Empty-class rows that do not have finite distance values were not
invented or imputed.

Primary tables:

- `per_case_metrics.csv`
- `per_case_distribution_summary.csv`
- `top_case_changes.csv`
- `tradeoff_pattern_counts.csv`
- `outlier_sensitivity.csv`
- `boundary_distance_correlations.csv`

Figures:

- `../figures/per_case_delta_distributions.png`
- `../figures/boundary_distance_relationships.png`

## Distribution results

| Dataset | Metric | Mean oriented delta | Favorable cases | Interpretable result |
|---|---|---:|---:|---|
| ISIC2018 | Dice | -0.00026 | 56.1% | Mean essentially unchanged |
| ISIC2018 | Boundary F1 | +0.00769 | 56.3% | Small, broadly distributed gain |
| ISIC2018 | HD95 | +1.0409 px | 56.9% | Favorable mean and majority |
| ISIC2018 | ASSD | +0.2207 px | 57.8% | Favorable mean and majority |
| ACDC | Dice | -0.00211 | 51.9% | Tail magnitude offsets small wins |
| ACDC | Boundary F1 | -0.00241 | 50.8% | No broad benefit |
| ACDC | HD95 | -0.8093 | 52.4% | Unfavorable mean despite a slight favorable majority |
| ACDC | ASSD | -0.3411 | 50.4% | Unfavorable tails dominate |
| Synapse | Dice | +0.00311 | 53.1% | Small overlap gain |
| Synapse | Boundary F1 | +0.00314 | 42.1% | Mean driven by magnitude/ties, not a majority |
| Synapse | HD95 | -0.1003 | 44.8% | Distance regression |
| Synapse | ASSD | -0.0462 | 48.2% | Small distance regression |

Usable row counts vary because empty foreground/classes make some macro
distance metrics undefined: 519 ISIC2018 images, 351--366 ACDC slices, and
820--1,039 Synapse slices depending on metric.

## Are effects concentrated in a few samples?

ISIC2018's HD95 result remains favorable after trimming the largest absolute
deltas: +1.04 px untrimmed and +0.95 px after 5% trimming. Its Dice mean changes
from -0.00026 to +0.00269 after 5% trimming, showing that the overlap mean is
sensitive to negative outliers.

ACDC distance regressions attenuate after trimming (HD95 -0.81 to -0.34; ASSD
-0.34 to -0.12), so a minority of larger failures matters. Synapse HD95 becomes
more unfavorable after trimming (-0.10 to -0.21), which argues against a claim
that the Synapse distance result is caused only by a few bad slices.

## Cross-metric relationships

Spearman correlations between Boundary F1 delta and favorable distance delta
are moderate on ISIC2018 (HD95 0.459; ASSD 0.640) and ACDC (0.560; 0.619), but
weak on Synapse (0.239; 0.252). On Synapse, Boundary F1 tracks Dice much more
closely (0.919). This supports a metric-conflict discussion: local boundary
agreement can rise while rare distant errors worsen HD95 or ASSD.

## Explicit trade-offs

- ISIC2018: 64/517 cases (12.4%) have nearly unchanged Dice
  (`|delta| <= 0.005`) and Boundary F1 gain above 0.01.
- ACDC: the corresponding count is 12/351 (3.4%).
- Synapse: the corresponding count is 16/820 (2.0%).
- Synapse has about 21% of usable slices where Dice or Boundary F1 is favorable
  while HD95 worsens.
- ISIC2018 still has 10.8% of usable images where Dice is favorable but HD95 is
  unfavorable.

## Main-paper use

The ISIC2018 distributions support a limited statement: in this locked
validation setting, favorable boundary and distance shifts occur across more
than half the images and are not solely an average created by one extreme case.
The cross-metric plots and trade-off counts support the paper's central question
about when boundary-oriented training aligns with distance metrics.

## Limitations

ACDC and Synapse do not show a stable case-wise advantage. The Synapse distance
regression is not explained away by a small number of outliers. Per-slice rows
are not independent patient/volume observations; inferential claims must use
the cluster analysis in `statistical_robustness_audit.md`.
