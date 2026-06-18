# Zenodo deposit (large binaries)

The trained checkpoints and per-case prediction arrays are too large for git and are
archived on Zenodo. `ZENODO_MANIFEST.csv` (committed at the repo root) lists every
archived file with its `bundle`, `relative_path` (where to place it in the repo),
`bytes`, and `sha256`, so the deposit is verifiable against this repository.

## Bundles — one Zenodo record, one version DOI

| Bundle (tarball) | Contents | Files | Size |
|---|---|---|---|
| `checkpoints_unet_bdslite` | `outputs/runs/<dataset>_{unet,bds_lite_full}_seed{1,2,3}/best.pt` | 18 | ~447 MB |
| `checkpoints_gsl` | `results/gsl/checkpoints/phase16_matched_gsl_*/best.pt` | 9 | ~209 MB |
| `predictions` | `outputs/evaluations/predictions/<dataset>_{unet,bds_lite_full}_seed{1,2,3}_<split>/*.npy` | 14,718 | ~740 MB |

- **Checkpoints are best-only** (the max-validation-DSC checkpoint, which backs every
  reported number). `last`/per-epoch checkpoints are intentionally excluded.
- **Predictions** cover the in-scope comparison (U-Net vs BDS-Lite) on the evaluated
  splits: ISIC2018 val (519), ACDC test (366), Synapse test (1568), three seeds.
- All three bundles go into a **single Zenodo record / single DOI**. Reference the
  minted **version DOI** (not just the concept DOI) in `README.md` and `CITATION.cff`.

## Verifying a download

1. Extract each tarball so files land at the `relative_path` from the manifest.
2. `sha256sum -c` against `ZENODO_MANIFEST.csv` (per-file hashes).

## Per-tarball hashes

The manifest commits the **per-file** sha256 (sufficient to verify every extracted
file). The **per-tarball** sha256 can only be computed once the tarballs are built;
record them here (and in the Zenodo record description) at deposit time:

```
# filled in at deposit:
# checkpoints_unet_bdslite.tar.gz  sha256: <...>
# checkpoints_gsl.tar.gz           sha256: <...>
# predictions.tar.gz               sha256: <...>
```

To reproduce the cluster analysis, place the `predictions` bundle under
`outputs/evaluations/predictions/` and run `analysis/generate_rescue_analysis.py`
(see `analysis/README.md`).
