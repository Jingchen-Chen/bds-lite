# Compute environment

All experiments were run on a single workstation. The numbers in the manuscript were produced under the following environment.

## Hardware

- GPU: NVIDIA GeForce RTX 5060 Laptop GPU
- CUDA runtime: 13.0
- OS: Linux 7.0.0 (x86_64)

## Software

- Python 3.14.4
- PyTorch 2.11.0 (`torch==2.11.0+cu130`), TorchVision 0.26.0
- NumPy 2.4.4, SciPy 1.17.1, scikit-image 0.26.0
- pandas 3.0.2, Pillow 12.2.0, PyYAML 6.0.3, einops 0.8.2, tqdm 4.67.3
- matplotlib 3.10.9, h5py 3.16.0

Exact pins are in `requirements.txt` (pip) and `environment.yml` (conda).

## Determinism notes

- Seeds: model/training seeds are 1, 2, and 3 (three independent runs per configuration). The data split seed is **2026** (see `splits/`).
- The cluster-aware statistical analysis (`analysis/generate_rescue_analysis.py`) uses a module-global NumPy generator seeded with `np.random.default_rng(20260606)`. Because that single generator is consumed in script order, the bootstrap confidence-interval endpoints reproduce **exactly only when the full analysis script is run end-to-end** (not when functions are called out of order). All non-bootstrap quantities (point estimates, sign-based effect sizes, Holm-adjusted p-values, leave-one-cluster-out ranges) are fully deterministic. See `docs/statistics.md`.
- GPU kernels (e.g. distance transforms, interpolation) may introduce last-bit nondeterminism across hardware; the manuscript's conclusions are reported at the level of cluster-aware statistics, not bit-exact reproduction.

## Approximate training cost

Per the resource profile (manuscript Table 4; `results/profiling/`), one training step (batch 8, 224×224, AMP, no-update profile) takes ~40 ms (U-Net), ~57 ms (BDS-Lite), ~52 ms (U-Net+GSL) on the GPU above. Each configuration is trained for 150 epochs; the auxiliary boundary branch is removed at inference, leaving a deploy graph within ~0.06% of the U-Net parameter count.
