# Third-Party Notices

This repository's original code is released under the MIT License (see `LICENSE`).
It additionally incorporates the third-party component below.

## Generalized Surface Loss — `src/bds_lite/losses/gsl.py`

`src/bds_lite/losses/gsl.py` is **adapted and modified** from the MIST framework's
`GenSurfLoss` implementation:

- **Upstream:** MIST (Medical Imaging Segmentation Toolkit),
  <https://github.com/mist-medical/MIST>, file
  `mist/loss_functions/losses/generalized_surface.py`.
- **License:** Apache License 2.0. Copyright [2024] [Adrian Celaya].
  The full upstream license text is vendored verbatim at
  [`THIRD_PARTY_LICENSES/Apache-2.0-MIST.txt`](THIRD_PARTY_LICENSES/Apache-2.0-MIST.txt).
- **Modifications (Apache-2.0 §4(b)):** ported from MIST's 3D `GenSurfLoss` to a
  2D multi-class module; the surface term is written in MIST's algebraic form and
  is bit-for-bit equal to the originally-run expression on one-hot targets
  (verified); the module additionally keeps explicit per-class weights with a
  weighted sum-then-divide aggregation and uses the paper's squared-Dice +
  cross-entropy region term; a step/linear/cosine alpha schedule is added.

### Citations (both required per the journal's third-party-code policy)

- A. Celaya, B. Riviere, D. Fuentes. "A Generalized Surface Loss for Reducing the
  Hausdorff Distance in Medical Imaging Segmentation." arXiv:2302.03868 (2024).
- A. Celaya et al. "MIST: A simple, configurable, and reproducible pipeline for 3D
  medical imaging segmentation." arXiv:2407.21343 (2024).

## Datasets

ISIC 2018, ACDC, and Synapse multi-organ images/labels are **not redistributed**
here (third-party licenses). Download instructions, DOIs, and citations are in
[`docs/data_access.md`](docs/data_access.md). Only locked split manifests
(`splits/`) and dataset configs are included.

## Model architectures

The U-Net backbone (`src/bds_lite/models/unet.py`) is an original reimplementation
of the architecture of Ronneberger, Fischer, and Brox (MICCAI 2015) and is covered
by this repository's MIT license; the paper is cited in the manuscript.
