# Independent review of provisional evidence sections 11–15

Scope: source and saved-output inspection, no model execution, no experimental
result edits, and no E3 or Experiment C outcome access. Findings below concern
the provisional document read on 2026-11-10. The root agent owns the final rewrite.

## Corrections needed

1. **Verified reporting error: cache timing is available as a post hoc measurement.**
   Section 12 says cache generation time is unavailable. The original end-to-end
   duration was not recorded, but
   `results/acdc_confirm_eval_posthoc/teacher_cache_timing.json` records a later
   median **27.277531358995475 seconds** across three passes of the 1,350 training
   images (batch 16). It includes teacher forward, softmax, float16 copy and
   per-image diagnostics; excludes model loading, validation, saving and the
   20-image comparison. `analysis/revision/make_restructure_assets.py:131–135`
   binds this to the manuscript as 27 seconds / three repetitions. Preserve
   original-duration absence and exact benchmark scope. Do not call it the
   original complete cache-generation duration.

2. **Verified implementation qualification: identical KD-versus-D graph is not
   identical unpruned eager execution to plain U-Net.**
   `src/bds_lite/models/bds_lite.py:168–176` computes the segmentation projection,
   gate and output head even when gate scale is zero. The gate's own forward
   (`:53–54`) still computes its convolution and sigmoid. D and its external-KD
   variants have identical model blocks and therefore identical inference
   architecture relative to one another. At scale zero the gate is algebraically
   identity, and unused projection/gate computation can be pruned. The unchanged
   eager forward does not itself perform that pruning. State graph equivalence
   to plain U-Net only for an explicitly pruned/exported graph; do not claim
   measured zero overhead for the original eager D/KD execution. This does not
   change any metric comparison or the KD-versus-D inference-cost identity.

3. **Literature-source attribution needs precision.**
   Section 14's unavailable “LB-UNet/Swin-related” ECCV-workshop chapter should
   name the Swin-Unet chapter (doi:10.1007/978-3-031-25066-8_9), not conflate it
   with LB-UNet's MICCAI 2024 publication. Existing
   `revision/validation/published_metric_protocol_audit.json` records upstream
   code hashes and protocol differences; that does not independently certify
   exact historical published-run provenance. Literature values remain reported
   contextual evidence unless their original documents are separately verified.

4. **Avoid conflating original checkpoints with withdrawn contaminated scores.**
   Section 11's near-duplicate “submitted historical F−U” result refers to original
   checkpoints re-evaluated on the official 1,000-image test set, not the invalid
   519-image development evaluation. Use that distinction explicitly.

## Values checked against supporting saved tables

- Gate-scale and distillation-weight means in section 11 match
  `revision/tables/families/sensitivity_means.csv`. Replication counts differ:
  ISIC alpha sweep three runs; Synapse alpha and ISIC lambda sweeps one run.
- Near-duplicate numbers match
  `revision/tables/families/isic_duplicate_sensitivity.csv`: original-checkpoint
  official-test Boundary F1 difference +0.030665 [0.025676, 0.035611], and +0.023966
  [0.018476, 0.029275] after the strictest exclusion (833 images); retrained family
  +0.003608 [0.000068, 0.007207], then +0.002517 [-0.001317, 0.006457]. These are
  conditional fixed-model intervals, not training-recipe uncertainty.
- Gate-identity ACDC/Synapse effects match
  `revision/tables/gate_identity_effects.csv`; “not exactly invariant” and
  conditional scope remain appropriate.
- Volume DSC rows in section 15 match `revision/tables/volume_primary_statistics.csv`:
  ACDC +0.014232 [0.002749, 0.029916], Holm p=0.057701; Synapse +0.013765
  [0.006717, 0.020477], Holm p=0.020508. These concern fixed trained models, image
  stacks and index grids; the Synapse surface analysis drops one case.
- Section 13's diagnostic cosine means/conflict fractions reproduce directly from
  saved raw seed-1 gradient records in the final statistics verifier. Its AP
  means reproduce as aggregation of the saved probe table. Original feature/pixel
  inference outputs were not reconstructed in this audit.
- Section 12 profile arithmetic reproduces from the saved resource table:
  +8.3268% batch-1 / +14.9883% batch-16 FP32 GPU latency, 1,072 extra deployed
  parameters, and 104 to 115 kernels. Profile timing was not independently rerun.

## Evidence status

The identified corrections are reporting/implementation-scope qualifications,
not discrepancies in primary statistical calculations. The main fixed-effect,
ISIC and ACDC/E1/E2 calculations passed 4,095 independent scalar checks with no
mismatch (new final statistics receipt). Source-table checks above must not be
promoted to independent reconstruction from prediction/feature outputs. Remaining
raw-data provenance, feature-mechanism, published-literature and clean-environment
gaps are explicit limitations. Whether E3 has analyzable registered outcomes is
the separate E3 review's responsibility; no opinion on its results is supplied here.
