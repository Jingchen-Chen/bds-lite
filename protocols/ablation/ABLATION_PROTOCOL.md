# WP2/WP4 queue-A protocol: config-only ablations and Attention U-Net (frozen before training)

Written 2026-11-02 before any queue-A run started. No queue-A result exists at
registration time. This document is hashed into every queue-A config.

## Adapter and locked protocol

All runs use the unchanged, already-validated adapter `scripts/revision/train_run.py`
(see TRAINING_ADAPTER_PROTOCOL.md). That adapter reproduced the historical ISIC
full-model seed-1 trajectory exactly (selected epoch 82, identical validation DSC),
so new arms are trained under the same pipeline as the existing main M0/M6 families.
Locked: split-seed-2026 manifests, training seeds 1/2/3, 224x224, AdamW wd 1e-5,
cosine 150 epochs, clip 1.0, AMP, no augmentation, ISIC batch 8 / lr 3e-4,
ACDC/Synapse batch 16 / lr 1e-4, checkpoint = maximum validation DSC (strict >).
ISIC test = official 1,000-image Task-1 test, never used for any selection.
Gradient probes are disabled (they are a WP5 seed-1 diagnostic only).

## Arm definitions (all other settings identical to the M6 full recipe)

| Arm | model | gate scale alpha | lambda_b | lambda_distill | lambda_sdf |
|---|---|---:|---:|---:|---:|
| M6 full (existing p1b) | bdslite_unet | 0.25 | 0.05 | 0.05 | 1 |
| M1 SDF only | unet | n/a | 0 | 0 | 1 |
| M2 aux boundary only | bdslite_unet | 0 | 0.05 | 0 | 0 |
| M3 minus SDF | bdslite_unet | 0.25 | 0.05 | 0.05 | 0 |
| M4 minus distillation | bdslite_unet | 0.25 | 0.05 | 0 | 1 |
| M5 minus gate (alpha=0) | bdslite_unet | 0 | 0.05 | 0.05 | 1 |
| M0w (ACDC only) U-Net + class weights | unet | n/a | 0 | 0 | 0 |
| C3 Attention U-Net | attention_unet (c=32) | n/a | 0 | 0 | 0 |

- alpha=0 keeps the gate module, so parameter initialisation (RNG consumption) is
  identical to M6; the modulation factor 1 + 0*(g-0.5) is exactly 1. The gate
  then receives zero gradient. This is the training-time M5, distinct from the
  existing inference-only gate-identity intervention (WP1i).
- M2 uses alpha=0 and lambda_distill=0, so the projection P receives no gradient
  and is unused; the deployed function equals a U-Net trained with an auxiliary
  boundary decoder only.
- M4: P is trained only through the gate.

### ACDC foreground class weighting (disclosed confound)

The existing ACDC M6 (p1b full) runs used foreground class weights
[1.5, 1.0, 1.25] in the soft-Dice and SDF terms, whereas the ACDC M0 U-Net used
none. ISIC and Synapse used no class weights for any arm. Ablation arms are
"M6 minus a component", so ACDC M1-M5 keep the M6 class weights. The extra ACDC
arm M0w (U-Net + the same class weights, nothing else) separates the class-weight
effect from the boundary/SDF components. C3 is a U-Net-level comparator and uses
no class weights (like M0).

## Reuse of existing matched arms (not retrained)

- ISIC M1 = followup arm S (unet, SDF 1) and ISIC M3 = followup arm B
  (bdslite, SDF 0), seeds 1/2/3, recovered/followup/results/followup_20261018_v2.
  The same followup runner reproduced U and F bit-identically to the main p1b
  checkpoints (revision/validation/isic_main_aliases.json), so S/B are matched.
- M0 and M6 = existing p1b families.

## Analysis (registered before results)

Family A (each ablation vs M6) and baseline family (each arm vs M0), Holm across
the five metrics within each dataset; units, seed-averaging, bootstrap and SESOI
exactly as in analysis/revision/prereg_revision.md. Addition pathway
M0 -> M1 -> M6; leave-one-out M6 vs M3/M4/M5. Results are reported whether they
support or contradict the theoretical predictions (SDF -> distant errors / HD95,
ASSD; boundary BCE+Dice -> boundary band / Boundary F1). Attention U-Net is
reported in Family C (vs M0).
