# Table 1. Datasets and locked protocol

All values are from the locked split manifests under `data/splits/`. Numbers
were source-confirmed in `results/submission_blocker_resolution_v1/corrected_number_audit.csv`
(Table 1 / Section 4.1 rows). Inputs are processed at 224 x 224 with z-score
normalization in the common configuration.

| Dataset | Task | Train | Validation | Test | Unit | Source |
|---|---|---:|---:|---:|---|---|
| ISIC2018 | binary skin lesion (2D) | 2,075 | 519 | none in manifest family | images | `data/splits/isic2018/{train,val}.json` |
| ACDC | cardiac MRI (2D slices) | 1,350 | 186 | 366 | slices (70/10/20 patients) | `data/splits/acdc/{train,val,test}.json` |
| Synapse | abdominal multi-organ CT (2D slices) | 1,738 | 473 | 1,568 | slices (14/4/12 cases) | `data/splits/synapse/{train,val,test}.json` |

Notes: ISIC2018 has no test partition in the locked manifest family; legacy
aggregate filenames retain a `_test` suffix but the 519-image partition is
validation. ACDC/Synapse slice counts correspond to the patient/case splits
shown. Split seed 2026 with subject-disjoint checks where subject identifiers
exist.
