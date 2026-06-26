# Configuration of record

This release ships the **fully resolved configuration snapshot** for every training run, so each reported number traces to an exact hyperparameter set.

## Where the configs live

- `configs/run_resolved/<dataset>_<model>_seed<n>.yaml` — 18 snapshots (`{isic2018,acdc,synapse}` × `{unet,bds_lite_full}` × seeds `{1,2,3}`). These are the expanded `config_resolved.yaml` written by the trainer at run time and are the **authoritative** record of what ran.
- `configs/gsl/phase16_matched_gsl_<dataset>_seed<n>.yaml` — 9 matched-protocol GSL configs (the `U-Net+GSL` comparator).
- `configs/base.yaml`, `configs/datasets/*.yaml` — the common and per-dataset building blocks.

Clean top-level "recipe" YAMLs were **not** reconstructed; the resolved snapshots are authoritative and sufficient.

## Two fields in the snapshots need interpretation

The resolved snapshots were written by the code/layout in place at run time. Two fields do not map directly onto the cleaned public code and are explained here rather than rewritten (so the record stays pristine):

1. **`model.name: bds_lite`.** In the cleaned public code the model dispatch key is `bdslite_unet`. The public `build_model` (`src/bds_lite/training/builders.py`) accepts **both** `bds_lite` and `bdslite_unet` and builds the identical `BDSLiteUNet` (verified: 2,198,003 total parameters; 1,927,042 for the plain `unet`). So the snapshots run unchanged.

2. **`dataset.split_file: splits/isic2018_split.json`** (and the analogous ACDC / Synapse entries). This field is **vestigial**: the data loader (`NpzSegmentationDataset`) does **not** read it. It selects samples by globbing `data/processed/<dataset>/<split>/*.npz`. The authoritative partition is the set of seed-2026 manifests in `splits/` (ISIC train 2075 / val 519; ACDC 1350 / 186 / 366; Synapse 1738 / 473 / 1568), which match the processed-directory counts exactly. The specific path named in the snapshot points at a **superseded seed-1 split that is not shipped**; because the loader ignores it, training and evaluation used the seed-2026 partition regardless. See [`naming.md`](naming.md).

## Key hyperparameters (from the snapshots)

For `bds_lite_full`: 150 epochs, batch size 8, AdamW lr 3e-4, weight decay 1e-5, cosine schedule, AMP, grad clip 1.0; bounded gate scale α = 0.25; loss weights `boundary 0.05`, `distill 0.05`, `surface 1.0`, `dice 1.0`, `ce 1.0`. The auxiliary boundary decoder is disabled at inference. The `unet` runs share the trainer settings without the boundary/gate/distill/surface terms. The GSL configs use `use_gsl: true`, `gsl_alpha_schedule: step5`, otherwise matching the U-Net protocol.

## GSL config path remapping

The shipped GSL configs were copied from an internal phase-named directory; their `output_dir`, `split_file`, and wrapper paths were remapped to the public layout (`results/gsl`, `scripts/gsl/`). The GSL locked split manifests (`results/gsl/splits/<dataset>_locked_seed2026.json`) were verified to contain **exactly** the seed-2026 partition (their `source_manifests` record the `splits/<dataset>/*.json` paths with matching sha256 and counts).
