# Data access and provenance

The ISIC 2018, ACDC, and Synapse images and labels are **not redistributed** in this
repository because they are governed by third-party licenses and data-use terms. This
document explains how to obtain them and rebuild the processed tensors that the
pipeline consumes. Only the locked split manifests (`splits/`) and dataset configs are
shipped here.

> Bibliographic details (years, DOIs, accession IDs) should be verified against the
> primary sources before submission; see `manuscript/human_tasks_before_submission.md`.

## Datasets

### ISIC 2018 — Task 1 (skin lesion boundary segmentation)
- Source: ISIC 2018 Challenge, Task 1 (lesion segmentation), ISIC Archive.
  <https://challenge.isic-archive.com/data/>
- Key references: Codella et al., "Skin Lesion Analysis Toward Melanoma Detection
  2018: A Challenge Hosted by the ISIC" (arXiv:1902.03368); Tschandl et al., "The
  HAM10000 dataset" (Scientific Data, 2018).
- Classes used here: background, lesion (binary).

### ACDC — Automated Cardiac Diagnosis Challenge
- Source: ACDC, Creatis, MICCAI 2017 challenge.
  <https://www.creatis.insa-lyon.fr/Challenge/acdc/>
- Key reference: Bernard et al., "Deep Learning Techniques for Automatic MRI Cardiac
  Multi-Structures Segmentation and Diagnosis: Is the Problem Solved?" (IEEE TMI,
  2018).
- Classes used here: background, right ventricle, myocardium, left ventricle.

### Synapse — Multi-Atlas Labeling Beyond the Cranial Vault (abdomen)
- Source: MICCAI 2015 Multi-Atlas Labeling Beyond the Cranial Vault, Synapse
  (accession syn3193805). <https://www.synapse.org/#!Synapse:syn3193805>
- Key reference: Landman et al., "MICCAI Multi-Atlas Labeling Beyond the Cranial
  Vault — Workshop and Challenge" (2015).
- Classes used here: background + 8 abdominal organs (aorta, gallbladder, spleen,
  left kidney, right kidney, liver, stomach, pancreas).

## Rebuilding the processed tensors

After downloading each raw dataset, convert it to the processed `.npz` layout the
pipeline expects:

```bash
python scripts/convert_isic2018.py   # -> data/processed/isic2018/{train,val}/*.npz
python scripts/convert_acdc.py       # -> data/processed/acdc/{train,val,test}/*.npz   (needs nibabel)
python scripts/convert_synapse.py    # -> data/processed/synapse/{train,val,test}/*.npz
python scripts/prepare_boundary_targets.py   # adds boundary/SDF targets used in training
```

`scripts/convert_acdc.py` requires `nibabel` (optional extra; see `requirements.txt`).
The converters partition data according to the **seed-2026** manifests in `splits/`
(subject-disjoint; each manifest carries a `sha256` and `count`). ISIC2018 has no test
split — see [`naming.md`](naming.md) for the `_test`→validation caveat.

## Ethics / licensing

Use of each dataset is subject to its own license and data-use agreement; obtain and
comply with those terms from the sources above. No patient-identifiable data is stored
in this repository.
