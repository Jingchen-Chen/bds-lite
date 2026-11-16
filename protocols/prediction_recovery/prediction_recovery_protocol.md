# Prospective fixed legacy-checkpoint prediction recovery

Registered after U/S/F official outcomes were known, before generating or inspecting
any new official results for the checkpoint families below. This timing is
transparent: no claim that these comparisons were originally preregistered.

All surviving plain-U-Net GSL P2 checkpoints, seeds1/2/3 across ISIC/ACDC/Synapse,
are included by identity, never chosen by favorable test outcomes. Their hashes
and cohort list are frozen in prediction_recovery_queue.json. They are the
**legacy_gsl** family, not missing Phase16 models. Recovered Phase16 scalar
results must never be attached to these weights. The exact historical GSL training
source/numeric distance-transform equivalence remains unverified.

The existing Attention U-Net checkpoints, training seed2026, are also included as
**attention_reference**. One seed per dataset is not a matched three-seed
comparator, and cannot be used as three independent samples or substituted for
seeds1/2/3. These references are reported separately.

Inference uses embedded model config, strict state loading, released unaugmented
NCHW contiguous float32 inputs, argmax labels, no tuned postprocessing, original
seed_everything settings, batch8 ISIC/batch16 other tasks. Read processed val/test
IDs from frozen manifests; refuse any missing/extra prediction IDs. Official ISIC
uses the same checked archives and preprocessing as the completed U/S/F evaluation.
No model, threshold, hyperparameter or checkpoint is selected using test labels.

Save per-image/per-class five metrics, empty-mask indicators, diagonal-penalized
surface sensitivity and predictions. For official ISIC, also original-resolution
Jaccard and thresholded Jaccard. Keep valid counts. Physical units remain
unverified: all processed-grid distances use index units; no mm or clinical claim.

Prediction recovery is inference only, no optimizer/gradient/training. One process
holds revision/.gpu.lock and refuses an already occupied GPU, without terminating
unrelated processes. Skip only outputs with matching completion/provenance hashes;
refuse to overwrite unknown incomplete results. Training queue remains separate.

Numerical output may be used only after validation and with its true family label.
Main three-seed matched-comparator claims and Phase16 prediction requirements
remain unfulfilled unless their actual provenance requirements are met.
