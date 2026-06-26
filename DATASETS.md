# Datasets

The three benchmarks used in this study are **not redistributed** in this repository: they are governed by third-party licenses and data-use terms. This file summarizes how to obtain them and how the pipeline expects them on disk. The deeper provenance and the exact rebuild commands live in [`docs/data_access.md`](docs/data_access.md); only the locked split manifests (`splits/`) and dataset configs (`configs/datasets/`) are shipped.

## The three benchmarks

| Dataset | Task | Fg. classes | Access | Registration |
|---|---|---|---|---|
| **ISIC 2018** (Task 1, lesion segmentation) | binary skin-lesion (2D) | 1 | ISIC Archive — <https://challenge.isic-archive.com/data/> | account required to download |
| **ACDC** (Automated Cardiac Diagnosis Challenge) | cardiac MRI (2D slices) | 3 (RV, myocardium, LV) | Creatis / MICCAI 2017 — <https://www.creatis.insa-lyon.fr/Challenge/acdc/> | registration + agreement |
| **Synapse** (Multi-Atlas Labeling Beyond the Cranial Vault, abdomen) | abdominal multi-organ CT (2D slices) | 8 organs | Synapse `syn3193805` — <https://www.synapse.org/#!Synapse:syn3193805> | Synapse account + data-use agreement |

Citations: ISIC2018 (Codella et al. 2019; Tschandl et al. 2018), ACDC (Bernard et al. 2018), Synapse (Landman et al. 2015). See the manuscript and `references_zotero.bib`.

## Splits and preprocessing (used in this study)

- **Split seed: 2026.** Subject-disjoint, locked manifests in `splits/` (`train/val/test.json`); each carries a `sha256` and a `count`. Integrity is guarded by `tests/unit/test_split_integrity.py`.
- **Counts:** ISIC2018 train/val = 2,075 / 519 (**no test split** — see [`docs/naming.md`](docs/naming.md)); ACDC train/val/test = 1,350 / 186 / 366 slices (70/10/20 patients); Synapse train/val/test = 1,738 / 473 / 1,568 slices (14/4/12 cases).
- **Processing:** all matched experiments use processed **224×224** arrays; images use continuous interpolation, masks nearest-neighbor. ISIC2018 images scaled to [0,1]; ACDC uses the TransUNet H5 z-score path; Synapse preserves provided image values. See the manuscript Methods and `src/bds_lite/data/README.md` for the exact per-source rules.

## Expected on-disk layout

Raw downloads are converted into the processed `.npz` layout the loader reads:

```text
data/
├── raw/                         # your downloads (any layout the converters accept)
│   ├── ISIC2018/
│   ├── ACDC/
│   └── Synapse/
└── processed/                   # produced by scripts/convert_*.py (git-ignored)
    ├── isic2018/{train,val}/*.npz
    ├── acdc/{train,val,test}/*.npz
    └── synapse/{train,val,test}/*.npz
```

`data/` is git-ignored and never committed.

## Rebuild commands

```bash
python scripts/convert_isic2018.py            # -> data/processed/isic2018/{train,val}
python scripts/convert_acdc.py                # -> data/processed/acdc/{train,val,test}  (needs nibabel)
python scripts/convert_synapse.py             # -> data/processed/synapse/{train,val,test}
python scripts/prepare_boundary_targets.py    # adds boundary + SDF training targets
```

## Pointing configs at your data

The dataset configs in `configs/datasets/*.yaml` carry the processed-data root. Edit the path field (or override on the command line) to point at your local `data/processed/<dataset>` directory; use **relative paths** so the repo stays portable. The committed split manifests document the partition those directories must match for exact reproduction.

## Licensing

Each dataset is subject to its own license and data-use agreement; obtain and comply with those terms from the sources above. This study is a secondary analysis of public, de-identified benchmarks; no patient-identifiable data is stored in this repository.
