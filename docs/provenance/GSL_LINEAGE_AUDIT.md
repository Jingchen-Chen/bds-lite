# GSL lineage audit — 2026-11-02

The nine `results/checkpoints/{dataset}_p2_unet_gsl_seed{1,2,3}/best.pt` checkpoints are loadable legacy P2 models. They are **not** interchangeable with the later Phase16 SciPy-wrapper models whose logs/CSVs are preserved in `recovered/followup/results/gsl/`.

Comparison of all 150 validation epochs shows nonzero trajectory differences in all nine pairs. Best epochs differ in eight pairs (ISIC seed2 shares epoch36 but its trajectory differs). Even where configuration optimizer/seed/batch/epoch settings match, differing training trajectories preclude claiming that the surviving legacy checkpoint generated a Phase16 result. The torch.cdist/SciPy distance-map implementations, runtime numeric behavior and historical provenance need investigation; no single cause is asserted from log differences alone.

Phase16 checkpoint files are absent from the inventoried sources at their documented paths. Phase16 per-case metric CSVs and complete logs may be preserved as historical evidence, but missing prediction arrays cannot be reconstructed from them. Do not attach the old checkpoint to those rows. Do not recreate a checkpoint or result.

Attention U-Net checkpoints exist for ISIC/ACDC/Synapse, but embedded training seed is 2026, not seeds1/2/3. These are useful historical reference models, not completed three-seed matched comparators. Source/config existence does not complete C3.

Next steps: infer surviving legacy GSL checkpoints into a separately labeled family; preserve exact checkpoint/config/split lineage. Inspect archived ZIP listings and any additional source locations for actual Phase16 checkpoints before authorizing their conditional retraining. Recovered predictions from legacy GSL must not be mixed with Phase16 aggregates. All negative and mixed findings remain reportable.
