# Gate identity descriptive/inferential analysis amendment

This specification is committed after gate prediction generation began, but before
inspection of comparative gate effects. Old manuscript gate-collapse claims were
known. It is a transparent revision-stage exploratory amendment, not an original
preregistration. The intervention/checkpoint/cohort protocol preceded prediction
generation and remains unchanged. No test-based model or threshold selection.

Use official ISIC test images, ACDC test patients and Synapse test cases as units;
report validation cohorts descriptively. Pair original full and identity-gate
predictions within each trained seed. For each primary test endpoint, require the
same image-class cells finite in both conditions and all three seeds, then average
cells within patient/case and seeds within unit. ISIC has one class cell per image.
Record finite/missing cells, missed structures, false alarms and changed pixels.

FamilyI: DSC,IoU,BoundaryF1,HD95,ASSD; two-sided paired Wilcoxon and Holm across
these five within each dataset. Bootstrap10000 paired independent units with
RNG20261102. Report sensitivity penalties separately on all cells. Empty policies,
preprocessing and two-dimensional processed-grid units are unchanged.

Reuse the revision-stage engineering SESOI from prereg_revision.md; do not claim
clinical equivalence or that non-significance proves no effect. Results remain
conditional on these trained seeds. No inference-overhead or training-time-M5
claim follows from gate-collapse accuracy alone. This comparison evaluates a
counterfactual of the same trained model, not a new training experiment.
