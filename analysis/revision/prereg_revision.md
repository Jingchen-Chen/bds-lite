# Prospective revision analysis plan — 2026-11-02

This is a revision-stage registration, not an original preregistration. Historical manuscript findings, October ISIC development U/S/B/F results and closed ACDC routes were known before this document. The overlap consistency audit and training-mask descriptors were already generated. No new official-test model effects, confidence intervals, p-values or practical-relevance decisions have been inspected. Downloading official archives is acquisition, not model selection or evaluation. Git commit records the prospective boundary.

## Frozen rules

New training must follow revision/plan.md section 0.2 and pass WP0 reproduction before the training queue. Three trained seeds 1/2/3. Maximum validation DSC checkpoint; no official test used for checkpoint, method, hyperparameter, threshold or stopping selection. Existing models are not selected because of new official-test performance. Read checkpoint embedded config and use strict manifest membership. Preserve every failed, empty and adverse case.

Prospective ISIC inference initially reuses fixed completed U/S/F seed1/2/3 checkpoints under recovered/followup/results/followup_20261018_v2/isic2018_{U,S,F}_seed{1,2,3}/best.pt. These estimate original U-Net, SDF-only and full-recipe behavior; the experimental arms were fixed in October. Checkpoint SHA256s are frozen in revision/audit/checkpoints.csv at this commit. No selection among arms follows test evaluation. Other required comparators remain pending; their absence does not waive them. Confirm normalization and exact state loading before evaluation. Use argmax class maps, no tuned postprocessing.

Before official inference: verify archive CRC and SHA256 receipts; exact image/mask IDs and counts; compare IDs with training/development; inspect available lesion/patient metadata. If patient identities are incomplete, report official held-out image set, not patient-independent external validation. A known training/validation identity overlap precludes patient-independence claims for development results.

## Units and families

ISIC: image. ACDC: patient. Synapse: case. Average each unit's metrics across trained seeds before paired inference; never regard slices or three seed-image pairs as independent patients. Preserve seed-level descriptive differences and missing-distance denominators.

- P: full BDS vs U-Net; Holm across five metrics within each dataset.
- A: each ablation vs full BDS, corrected across all ablation×metric tests within dataset. Addition-path contrasts vs U-Net form a separately labeled family, never select favorable correction scope after results.
- C: each comparator vs U-Net; Holm across comparator×metric tests within dataset.
- Omnibus: Friedman across complete matched methods at dataset×metric; Holm-adjusted paired Wilcoxon post-hoc. Incomplete families are provisional; report raw and family-defined adjusted values without claiming final family closure.

Two-sided paired Wilcoxon (zeros discarded, SciPy auto algorithm; record SciPy version), paired mean differences and percentile bootstrap 95% intervals. Resample independent units with replacement, 10,000 draws, RNG seed 20261102. If sample size or missingness prevents a test, report missing and reason; do not substitute fabricated p-values. Development-set inferential summaries remain exploratory.

## Practical relevance

Revision-stage thresholds chosen after historical findings were known, before independent-test effects: DSC 0.01; IoU 0.01; Boundary F1 0.01; HD95 2 processed pixels, ASSD 0.5 processed pixels. Physical analyses, only after geometry verification: HD95 1 mm; no new physical ASSD threshold is asserted here. These are engineering SESOI thresholds, not validated clinical safety boundaries.

Orient differences so positive favors BDS/comparator (negate distance differences). A 95% interval entirely above +SESOI is practically superior; entirely below −SESOI inferior; entirely inside [−SESOI,+SESOI] practically equivalent; otherwise inconclusive. Statistical significance alone does not imply practical benefit. Report raw-unit effect and uncertainty alongside labels.

## Empty masks and missingness

Use the released DSC/IoU epsilon convention and Boundary F1 tolerance2. Primary surface means retain valid-pair NaNs and explicit denominators. Also report all-case sensitivity distances: one-empty=image diagonal, both-empty=0; this is a failure penalty, not measured surface distance. No method may benefit silently from dropping complete misses. Apply the same empty policy and eligibility rules to paired contrasts; report all-case penalized contrasts as sensitivity, not replacement primary evidence.

## Figures, geometry and claims

Deterministic qualitative selection: largest positive, median, largest negative seed-averaged delta HD95, lexicographic ID ties; empty cases shown separately. Do not pick examples by visual attractiveness. No AI-generated scientific images.

Stacked/native metrics require verified sample-to-volume mapping, original dimensions and evaluator equivalence; physical metrics additionally require verified spacing/units/affine. No mm, EF or myocardial mass from failed geometry. Clinical/narrative/licensing/TierC approvals remain human responsibilities. Negative findings stay in manuscript. Final results are generated from CSV/JSON, never manually invented in LaTeX.
