# Final Caption Set (conservative)

Captions are deliberately conservative: no significance claims from three seed
means, no winner highlighting, no universal-superiority or hardware claims.

## Figure 1

**BDS-Lite training and inference graphs.** The compact U-Net encoder-decoder
is the retained segmentation path. During training, an auxiliary boundary
decoder supplies boundary supervision and features used for distillation and
bounded gating. The auxiliary boundary decoder is removed at inference. The
diagram describes the studied implementation and does not imply a
hardware-specific property.

## Figure 2 (Phase 16 matched replacement)

**Matched three-seed comparison under the locked protocol.** Bars show seed
means and error bars show seed standard deviations for U-Net, BDS-Lite, and
U-Net+GSL on ISIC2018 validation, ACDC test, and Synapse test (test where
applicable). Higher DSC and Boundary F1 and lower HD95 and ASSD are favorable.
All methods use the locked split family and maximum-validation-DSC checkpoint
rule; test data were not used for model or checkpoint selection. The metric
directions are mixed across datasets, and the plot does not imply significance
from three seed means or any universal superiority.

## Figure 3

**Distributions of per-case BDS-Lite versus U-Net changes.** Distance changes
are oriented so positive values indicate favorable movement for BDS-Lite.
ISIC2018 uses image-level units. ACDC and Synapse slice distributions are
descriptive because slices from the same patient or case are not independent;
cluster-aware analyses provide the inferential summaries.

## Figure 4

**Boundary F1 and surface-distance changes can disagree.** Each point is a
seed-averaged image or slice in the available analysis, with distance changes
oriented so positive values are favorable for BDS-Lite. Boundary improvement
does not always imply distance improvement. Correlations are descriptive and
do not establish causality or slice independence for ACDC and Synapse.

## Figure 5

**Exploratory ACDC and Synapse class-level heterogeneity.** Cells show
seed-averaged oriented BDS-Lite minus U-Net changes; positive values are
favorable after metric orientation. Color scales are metric-specific and
should not be compared across panels. These slice-derived summaries are
exploratory and do not establish organ-specific efficacy.

## Figure 6

**Automatically selected illustrative cases.** The panels include an ISIC2018
favorable example, an ISIC2018 unfavorable example, an ACDC example in which
Dice and distance changes conflict, and a Synapse distance failure. Selection
used predeclared metric criteria before visual review. The examples are
illustrative and do not estimate prevalence or establish causality.
