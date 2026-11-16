# Inference parity investigation

Initial GPU comparison of first eight fixed validation images for U seed1 differed by 59 pixels. Raw RGB preprocessing matched stored NPZ images exactly. Repeating with original seed_everything/cudnn deterministic settings alone retained 59 differences. No official model metrics were generated at this stage.

Inspection identified a further input-layout difference: np.stack of CHW-transposed PIL arrays may retain channels-last-compatible strides, while NpzSegmentationDataset explicitly calls np.ascontiguousarray. The new inference adapter now enforces contiguous NCHW before GPU transfer. Result is recorded separately in official_inference_parity.json if and only if all nine checkpoints reproduce all eight masks exactly. This is a checkpoint-inference check, not a training replication pass.
