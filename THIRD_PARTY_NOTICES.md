# Third-party notices

The original code and text of this repository are released under the MIT License (`LICENSE`).
The items below are third-party material, or are derived from it, and keep their own terms.

## Images

### Figure 3 of the manuscript: ACDC cine-MRI slices (CC BY-NC-SA 4.0)

- **Where:** `paper/figures/acdc_qualitative_combined.pdf` (Figure 3; the separate official-test panel is also retained as `paper/figures/acdc_qualitative_main.pdf`), and the manuscript PDF that
  embeds it, `paper/main.pdf`.
- **Content:** six ACDC test slices with reference masks and predictions. Panel 3a shows three
  internal test slices with U, D and BDS-Lite (study 1); panel 3b shows three official test slices
  from patients 101–150 with D and D+KD-band (replication). Each panel includes a change map
  against D. The slices were resampled to the 224 x 224 grid, cropped, and drawn with a scale bar.
- **Source:** Automated Cardiac Diagnosis Challenge (ACDC),
  <https://www.creatis.insa-lyon.fr/Challenge/acdc/>.
- **Licence:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
  (CC BY-NC-SA 4.0), <https://creativecommons.org/licenses/by-nc-sa/4.0/>, as stated on the ACDC
  distribution portal. Use is restricted to non-commercial scientific research. These images, and
  the two PDF files above insofar as they contain them, are **not** covered by this repository's MIT
  License; they are shared under CC BY-NC-SA 4.0.
- **Required citation:** O. Bernard, A. Lalande, C. Zotti, F. Cervenansky, et al. "Deep learning
  techniques for automatic MRI cardiac multi-structures segmentation and diagnosis: is the problem
  solved?" *IEEE Transactions on Medical Imaging* 37(11):2514–2525, 2018.
  doi:10.1109/TMI.2018.2837502.

### ISIC 2018 Task 1 reference masks in the contour figures (CC0)

- **Where:** `paper/figures/contours_isic2018*.{pdf,png}`, and the PDF that embeds them
  (`paper/supplement.pdf`).
- **Content:** reference-mask boundaries of ISIC 2018 Task 1 test images and the agreement,
  false-positive and false-negative pixels of this study's predictions, on a white background. No
  pixel of any dermoscopic photograph is reproduced.
- **Licence:** ISIC 2018 Task 1 ground truth is marked CC0 on the ISIC data page
  (<https://challenge.isic-archive.com/data/>); ISIC terms of use, section 4.3: annotations are CC0.
- **Attribution requested by ISIC:** N. Codella et al., "Skin lesion analysis toward melanoma
  detection 2018: a challenge hosted by the International Skin Imaging Collaboration (ISIC)",
  arXiv:1902.03368 (2019); P. Tschandl, C. Rosendahl, H. Kittler, "The HAM10000 dataset, a large
  collection of multi-source dermatoscopic images of common pigmented skin lesions", *Scientific
  Data* 5:180161 (2018). HAM10000 Dataset (c) by ViDIR Group, Department of Dermatology, Medical
  University of Vienna; MSK Dataset (c) Anonymous.

No other dataset image or annotation is included. In particular, the repository contains no ISIC
2018 photograph and no Synapse image or annotation.

## Code

### Generalized Surface Loss — `src/bds_lite/losses/gsl.py` (Apache-2.0)

- **Upstream:** the Generalized Surface Loss of Celaya et al., reference implementation
  <https://github.com/aecelaya/gen-surf-loss> (commit `2115cca`, cited in the module), also
  distributed as `GenSurfLoss` in MIST, <https://github.com/mist-medical/MIST>. Both are under the
  Apache License 2.0.
- **Licence text:** `THIRD_PARTY_LICENSES/Apache-2.0-MIST.txt` (Copyright [2024] [Adrian Celaya]).
- **Modifications (Apache-2.0 section 4(b)):** ported from a 3D loss to a 2D multi-class module;
  explicit per-class weights with a weighted sum-then-divide aggregation; squared-Dice +
  cross-entropy region term; step, linear and cosine alpha schedules.
- **Citations:** A. Celaya, B. Riviere, D. Fuentes, "A generalized surface loss for reducing the
  Hausdorff distance in medical imaging segmentation", arXiv:2302.03868 (2024); A. Celaya et al.,
  "MIST: a simple, configurable, and reproducible pipeline for 3D medical imaging segmentation",
  arXiv:2407.21343 (2024).

### Boundary Loss components — `scripts/comparators/components.py` (MIT)

- **Upstream:** <https://github.com/LIVIAETS/boundary-loss> at commit
  `171c32d88a4ce59af8be46fb88b96d3637b9515b` (`SurfaceLoss`, `one_hot2dist`, and the
  `gdl_surface_steal` weight schedule).
- **Licence text:** `THIRD_PARTY_LICENSES/MIT-boundary-loss.txt` (Copyright (c) 2023 Hoel
  Kervadec), the upstream `LICENSE` file at that commit.
- **Modifications:** the region term is this study's CE + soft Dice instead of generalized Dice, and
  all foreground classes are used.
- **Citation:** H. Kervadec et al., "Boundary loss for highly unbalanced segmentation", *Medical
  Image Analysis* 67:101851 (2021).

### EGE-UNet reimplementation — `src/bds_lite/models/ege_unet.py` (no upstream code; MIT)

- **What it is:** a reimplementation of the EGE-UNet design described by Ruan et al. (MICCAI 2023):
  six encoder/decoder stages, depthwise-separable shallow blocks, grouped multi-axis attention
  (GHPA) and group aggregation bridges (GAB). It contains no code from the official
  implementation and is covered by this repository's MIT License.
- **Check (2026-11-14):** compared with the official `models/egeunet.py` of
  <https://github.com/JCruan519/EGE-UNet> at its latest commit `f52ba30` (the repository has seven
  commits, 2023-07-17 to 2023-07-20). Of the module's 103 code lines of 20 or more characters
  (comments and imports excluded), only the class header `class EGEUNet(nn.Module):` also occurs
  upstream. The internals also differ: this GHPA uses
  per-channel softmax gating with BatchNorm, whereas the official block uses interpolated learned
  maps with depthwise convolutions and LayerNorm, and the bridges differ too. The module is
  therefore not the official architecture.
- **Upstream licence, for reference:** the official repository is under the Apache License 2.0,
  unchanged since its initial commit `806ec7e` (2023-07-17).
- **Correction to the module docstring:** it calls the file a port of the public implementation
  "commit f87e3df, MIT license". No commit `f87e3df` exists upstream, the upstream licence is
  Apache-2.0, and the code is a reimplementation, not a port. The docstring is left unchanged:
  the file's SHA-256 is registered in the run records, and source files under `src/` are not
  edited in this revision.
- **Use in the study:** runs of this model belong to two sets in the supplement's run ledgers: the
  April 2026 "family O reproducibility grid", of which only other arms are reported, and the
  "EGE-UNet compact baseline" (nine runs, dropped before submission). No result of this model is
  reported in the manuscript or supplement. The EGE-UNet value in the published-results table is
  the number reported by Ruan et al.
- **Citation:** J. Ruan, M. Xie, J. Gao, T. Liu, Y. Fu, "EGE-UNet: an efficient group enhanced UNet
  for skin lesion segmentation", MICCAI 2023.

### Not redistributed (fetched or installed at run time)

- **LB-UNet**, <https://github.com/xuxuxuxuxuxjh/LB-UNet>, commit `913ee68`: no licence is stated
  upstream. `scripts/comparators/fetch_lbunet.py` downloads it at run time into `third_party/`
  (git-ignored); the wrapper in `scripts/comparators/` contains no upstream code.
- **segmentation_models_pytorch** 0.6.0 (MIT), installed from PyPI; used by the external teacher.
- **ResNet-34 ImageNet encoder weights**, Hugging Face `smp-hub/resnet34.imagenet`, revision
  `7a57b34f`: downloaded at run time and hash-checked; terms as published by the provider.

### Identifiers taken from other projects

- `docs/provenance/sources/transunet/test_vol.txt`: the Synapse test-case list released with TransUNet
  (<https://github.com/Beckschen/TransUNet>, Apache-2.0); J. Chen et al., "TransUNet: transformers
  make strong encoders for medical image segmentation", arXiv:2102.04306 (2021).

## Datasets (not redistributed)

- **ISIC 2018 Task 1**, <https://challenge.isic-archive.com/data/>: images under each contributor's
  licence, annotations CC0. Cite Codella et al. (2019) and Tschandl et al. (2018).
- **ACDC**, <https://www.creatis.insa-lyon.fr/Challenge/acdc/>: CC BY-NC-SA 4.0, non-commercial
  scientific research only. Cite Bernard et al. (2018).
- **Synapse multi-organ CT (BTCV)**, <https://www.synapse.org/Synapse:syn3193805>,
  doi:10.7303/syn3193805: provider's access terms. Cite B. Landman et al., MICCAI 2015 Multi-Atlas
  Labeling Beyond the Cranial Vault workshop and challenge (2015).

Only split manifests (case identifiers) of these datasets are included (`splits/`, and the per-run
configurations in `configs/`).
