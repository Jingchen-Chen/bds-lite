# Failure-Case Analysis

## Selection protocol

Cases were selected automatically from three-seed per-case deltas using
predefined categories: clearly favorable, clearly unfavorable, similar Dice
with a Boundary F1 difference, Synapse distance failure, and typical ACDC
negative cases. The complete selection and values are in
`failure_case_manifest.csv`. This avoids manually choosing visually attractive
examples, but the panels remain descriptive rather than inferential.

Each panel contains the input image, ground truth, U-Net prediction, BDS-Lite
prediction, false-positive/false-negative maps, and boundary overlays. GSL
per-image prediction arrays were not found, so GSL could not be included in
these panels.

## ISIC2018

Favorable cases such as `ISIC_0013140` combine a large distance reduction
(HD95 about 40.6 px and ASSD about 8.1 px in the favorable direction) with a
Boundary F1 increase of about 0.139. These panels show boundary cleanup without
requiring a large Dice change.

The clearest negative case, `ISIC_0016060`, has Dice delta about -0.223,
Boundary F1 delta about -0.218, HD95 worsening by about 56 px, and ASSD worsening
by about 26.5 px. This case demonstrates that the method can alter topology or
retain distant false-positive regions; the favorable mean does not protect
every lesion.

## ACDC

`patient052_frame01_slice_5` is the most useful counterexample: Dice is higher
by about 0.076 while HD95 worsens by about 26.1 and ASSD by about 17.3. A small
remote error or contour fragment can be nearly invisible to overlap metrics but
dominate distance metrics. Typical ACDC-negative panels show modest contour
shifts distributed across cardiac structures rather than a uniform failure
mode.

## Synapse

`case0004_slice134` has Dice delta about -0.078 and distance deterioration of
about 37.8 in HD95 and 36.6 in ASSD. `case0001_slice145` and
`case0003_slice187` provide additional distance-failure examples. The panels
support the interpretation that multi-organ predictions can gain local overlap
on some structures while introducing a remote component or missing a small
structure, which is strongly penalized by distance metrics.

## What the panels explain

1. Boundary supervision can sharpen a local contour while leaving global
   overlap almost unchanged.
2. Boundary F1 and HD95/ASSD are not interchangeable; distant errors can make
   them disagree.
3. Small or low-contrast structures and empty-class transitions are vulnerable
   in multi-class slices.
4. Aggregate means hide both large individual failures and organ-specific
   effects.

## Paper use and limitations

Use one favorable and one unfavorable ISIC panel, the ACDC Dice-distance
trade-off, and one Synapse remote-error panel in the main paper. Place the full
24-panel set under `../figures/failure_cases/` in supplementary material.
Panels must be captioned as automatically selected illustrations. They do not
establish prevalence, causal mechanism, or general applicability.
