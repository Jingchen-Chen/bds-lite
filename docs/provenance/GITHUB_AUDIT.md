# Audit against the public GitHub release (2026-11-02)

Baseline: https://github.com/Jingchen-Chen/bds-lite (HEAD 3104cb3; initial release 1b0c4f6,
which also contained ZENODO_MANIFEST.csv and docs/zenodo.md, later deleted). Every statement
below is backed by a command that can be re-run; evidence files are listed.
Supersedes the "Decision" section of FAMILY_PROVENANCE.md.

## 1. What the submitted Table 2/3 are built from (traced)

| Item | Source | Status |
|---|---|---|
| U-Net / BDS-Lite rows, all datasets | `outputs/evaluations/<ds>_{unet,bds_lite_full}_seed{1,2,3}_test.json` (GitHub `results/main_test_means.csv`) from checkpoints `outputs/runs/<ds>_<model>_seed<n>/best.pt` ("family O") | traced |
| Table 3 / Fig. 2 | `results/midterm_q2_rescue/analysis/{per_case_metrics,cluster_level_statistics}.csv` from per-case prediction arrays of the same checkpoints | traced |
| U-Net+GSL rows | Phase16 matched GSL (`configs/gsl/phase16_*`, SciPy-EDT wrapper `scripts/gsl/gsl_memory_bounded.py`), aggregates in `results/matched_gsl_*.csv` | traced; checkpoints not present locally |
| Gate-collapse paragraph | p1b follow-up family (`results/next_stage/p1b_gatefree_multiseed`) | traced |

**The local family-O checkpoints are the submitted models.** Re-running the six ISIC checkpoints on
the 519 locked-validation images reproduces the submitted per-seed DSC (e.g. U-Net seed 1
0.875820 vs 0.875817 in `main_test_means.csv`) and the submitted per-image values in
`per_case_metrics.csv` (max |dDSC| 4.3e-4 U-Net, 1.1e-3 BDS-Lite over 519 images; CPU vs GPU
numerics). Their file SHA-256 equals the archived release manifest (U2).

**Core code is identical.** `src/bds_lite/{models/bds_lite.py, training/losses.py,
training/runner.py, data/datasets.py, data/boundary.py, evaluation/metrics.py}` are byte-identical
between the GitHub release and the local tree. Differences are confined to the model registry
(`builders.py`, extra models) and comments in `gsl.py`. Family O was launched with an older trainer
CLI (`environment.json`: `train.py --config ... --seed N --output-dir outputs/runs --deterministic`,
commit 5914d35 not in the public history).

## 2. Classification of findings

### Confirmed scientific problem (affects submitted numbers)

**C1. ISIC2018 U-Net/BDS-Lite were trained on a different partition than the one they were
evaluated on.** Two independent lines of evidence:
- Optimiser step counts in the family-O ISIC checkpoints: 226.9 Adam steps/epoch (best.pt and
  last.pt, all six runs) = ceil(1,815/8); the locked partition would give ceil(2,075/8) = 260.
  ACDC (85 = ceil(1,350/16)) and Synapse (109 = ceil(1,738/16)) match the locked partitions.
- The best validation DSC in each ISIC training log equals, to six decimals, the stale
  389-image `_val.json` evaluation (e.g. 0.849585 vs 0.849583), i.e. checkpoint selection used the
  389-image validation set of the seed-1 split (`splits/isic2018_split.json`, 1,815/389/390).
The reported "519-image validation" numbers (n = 519 in Tables 2-3, Fig. 2) come from the locked
seed-2026 validation set, of which **358 images (69%) were in these models' training set**, 84 in
their selection set and 77 unseen. The ISIC U-Net and BDS-Lite rows of Tables 2-3 and the ISIC
claims built on them are therefore not held-out estimates (training-set contamination affects both
arms, so the paired contrast is not a clean held-out comparison either). GitHub
`docs/config_of_record.md` ("split_file is vestigial ... training used the seed-2026 partition")
is incorrect for ISIC2018 and correct for ACDC/Synapse.

*Remedy without retraining:* the official ISIC2018 Task-1 test set (1,000 images) is disjoint from
the whole 2,594-image development pool, hence independent of both partitions. The submitted
family-O ISIC checkpoints are being evaluated on it (`revision/extra_cohorts/famO_*`).

### Previously unresolved concerns (status after follow-up investigation)

- **U1 update (2026-11-03): RESOLVED by exact reproduction.** The queue_b GSL retrain (revision adapter,
  locked manifests, src/bds_lite GSL loss) reproduces the Phase16 `metrics.jsonl` exactly: validation DSC
  identical at every one of 150 epochs (max |d| = 0.0) and identical selected epoch, for ISIC2018 seeds 1-2
  (seed 3 identical through all epochs trained so far); training-loss means differ at ~1e-9 (summation order).
  Hence Phase16 GSL ISIC was trained on the locked 2,075-image partition (no contamination) and its reported
  ISIC value is the best-validation value on the 519 selection images. The retrained checkpoints are
  functionally the Phase16 models (file hashes not comparable: Phase16 best.pt 23.2 MB includes optimiser
  state, revision best.pt 7.7 MB). Evidence: `analysis/revision/gsl_trajectory_compare.py` →
  `revision/tables/families/gsl_trajectory_comparison.csv`; supplement S12. 2026-11-04: extended to 7/7 completed runs (ISIC2018 x3, ACDC x3, Synapse seed 1), all
  exact; the retrained runs reproduce the submitted U-Net+GSL Table-2 values with the original aggregation
  (max |d| 8e-5 overlap, 0.03 px distances; `gsl_table2_reproduction.csv`). 2026-11-04: all 9/9 runs complete and exact.
  Original text kept below.
- **U1. Phase16 GSL ISIC partition — likely documented partition; checkpoints unavailable.** The nine
  Phase16 checkpoints (hashes in the release manifest) are not on this machine (0 of 159 GSL-related
  checkpoints match; the Windows partition holds none). Timeline evidence: Phase16 ran 2026-06-09/10;
  the p1b family, trained 2026-05-07 by the same glob-based loader, provably used the 2,075-image locked
  partition (the revision adapter reproduces p1b exactly from the locked manifest); family O predates
  it (April grid). Phase16's config points to the locked manifest and its reported ISIC value equals
  its own best-validation value (0.8674). Hence Phase16 GSL ISIC was most likely trained on 2,075 and
  selected/reported on the 519 locked validation images (selection-set reporting, but no training
  contamination). Consequence: the submitted ISIC statement "U-Net+GSL weaker" compared clean GSL values
  with contaminated U-Net/BDS-Lite values and is withdrawn with the ISIC rows. The queued GSL retrain
  (locked partition) provides official-test GSL results. Ask the authors whether a Zenodo deposit with
  `checkpoints_gsl` exists; if so, evaluate it on the official test as well.
- **U2. Checkpoint hash mismatch — RESOLVED (my error).** All 18 submitted checkpoints are byte-identical
  to the archived release manifest (`revision/sources/github_release/ZENODO_MANIFEST_1b0c4f6.csv`); my
  earlier comparison failed because the manifest uses CRLF line endings. Independent confirmation:
  `analysis/revision/verify_checkpoint_identity.py` (file hashes; regenerated per-sample prediction arrays
  vs manifest prediction hashes, CPU vs original GPU).

### Release-only defects (do not affect the original experiments)

- **R1. Seed collision:** release `scripts/train.py` writes to `<output_dir>/runs/<experiment.name>`
  and the resolved configs' `experiment.name` has no seed suffix, so running seeds 1-3 from the
  release overwrites outputs. The original runs used the older CLI with `--seed` and per-seed
  directories (distinct optimiser histories confirm three separate runs).
- **R2.** `scripts/gsl/run_phase16_gsl.sh` sets `ROOT` three levels above the script (outside the repo).
- **R3.** `ZENODO_MANIFEST.csv` and `docs/zenodo.md` were deleted while README/RELEASE_INVENTORY
  still reference them; DOI placeholder.
- **R4.** Documentation error C1 above (`config_of_record.md`, `naming.md` describe the stale split
  as unused for training).
- **R5.** Local processed ISIC train folder contains a byte-identical duplicate
  `ISIC_0015312(1).npz` (glob-based loaders would see 2,076 files); not present in manifests; the
  revision adapter loads by manifest.

### Retracted claims from my earlier revision work (2026-11-02)

- **"ACDC BDS-Lite used class weights" was wrong for the submitted results.** Family O and the GitHub
  configs use `foreground_class_weights: null`. The weights [1.5, 1.0, 1.25] belong only to the
  later p1b follow-up family. Queued ACDC ablations were re-registered without weights
  (ABLATION_PROTOCOL_AMENDMENT_1.md).
- **"The submitted family is non-executable/incompatible and should be replaced by p1b" is
  withdrawn.** The submitted checkpoints exist and are evaluable; the core code equals the release.
  Family O remains the authoritative submitted family; the only confirmed problem is the ISIC
  partition mismatch (C1), which is corrected by evaluating the same models on independent data.
- The earlier statement that the family-O ISIC split difference was "only seen in a stale CSV" is
  replaced by the step-count and selection-log evidence above.
- The empty-class convention (both-empty = 1 in the submitted means) is a documented convention,
  not an error; it is now stated explicitly (F9) and a both-empty-excluded analysis is added.

### Genuine scientific limitations (unchanged)

Small cluster counts (ACDC 20, Synapse 12), three seeds, one compact backbone, no augmentation,
pixel-unit distances, unverified physical geometry for clinical indices.

## 3. Role of the p1b family

p1b (`results/checkpoints/*_p1b_*`, trained May 2026 with the release-equivalent trainer on the
locked partitions; ACDC BDS-Lite with class weights) diverges from family O from epoch 1 (e.g. ACDC
U-Net seed-1 epoch-1 validation DSC 0.4266 vs 0.4220): it is an independent realisation of the same
protocol, not a correction of family O. The revision adapter reproduces p1b exactly, so p1b is the
**pipeline-matched reference for the new ablations and comparators** (all trained by the adapter on
the locked partitions). It also provides an independent replication of the main comparison
(ISIC/Synapse identical recipe; ACDC differs by class weights).

## 4. Consequences for the revision (proposal; author approval required)

1. Main comparison (Tables 2-3): keep family O. ACDC and Synapse results stand as submitted
   (re-analysed with the documented conventions). ISIC: report family-O results on the official
   test set and withdraw the 519-image numbers as contaminated, with a transparent explanation.
2. Ablations and comparators: compare against pipeline-matched references (p1b U-Net/BDS-Lite for
   ISIC and Synapse; p1b U-Net and three new adapter-trained unweighted BDS-Lite runs for ACDC).
3. p1b reported as an independent replication family.

## 5. Verification of the official ISIC2018 test evaluation (approved replacement)

`analysis/revision/verify_official_isic.py` → `revision/validation/official_isic_verification.json`:
archive SHA-256 and member counts match the acquisition receipt; 1,000 images with matching masks;
0 ID overlap with either development partition; preprocessing parity exact (test loader vs
re-implementation, 50 images; released converter vs stored development arrays, 499 available raw
JPEGs); predictions reproduced through a separate code path (1 differing pixel in 6x64 images);
DSC/IoU recomputed exactly and HD95/ASSD to 7e-6 px by an independent EDT implementation; family-P
means and Wilcoxon p-values reproduced exactly by an independent implementation (bootstrap CIs
within Monte-Carlo error).
Repeated lesions: content matching finds test images that repeat development lesions under another
ID (31 at r > 0.99, up to 167 at r > 0.90; no clean threshold). Bracketed exclusion
(`isic_duplicate_sensitivity.py`) leaves every ISIC conclusion unchanged (supplement S11).
