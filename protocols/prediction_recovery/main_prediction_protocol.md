# Fixed main-family ACDC/Synapse prediction recovery

Frozen before generating these prediction files or their new volume metrics.
The historical scalar outcomes were known from earlier manuscripts. No new
claim of original preregistration is made. Original revision statistical plan
analysis/revision/prereg_revision.md still controls new comparisons.

Use all existing p1b U-Net and bdslite_full_current seed1/2/3 checkpoints for both
ACDC and Synapse, validation and test. Identities and hashes are fixed in
main_prediction_queue.json. Do not select models/seeds by the known or new test
outcomes. Strict embedded model config/state loading; no training, no TTA or
postprocessing; contiguous NCHW FP32 inference, original seed settings, batch16.

Use the exact frozen image-ID manifests. Primary processed labels/images are
read unchanged. Keep classwise metrics, missingness, diagonal-penalized surface
sensitivity and complete uint8 predictions. Save checkpoint/config provenance.
These main-family outputs remain distinct from legacy_gsl, Phase16 and
single-seed Attention references. Exact manifest and metric validation is
required before using regenerated numbers in manuscript.

One GPU lock/process, no unrelated job interruption. No scientific metric is
selected for favorable direction. Further stacked/native evaluation follows
volume_evaluation_protocol.md; physical geometry remains unverified.
