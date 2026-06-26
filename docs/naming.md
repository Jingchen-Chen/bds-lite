# Split naming and the ISIC2018 `_test` / `_val` caveat

This document records a **known, deliberate naming caveat** in the ISIC2018 artifacts. It is surfaced here, not silently renamed, so that a reviewer can trace every reported number to the correct underlying split.

## ISIC2018 has no held-out test split

The locked seed-2026 partition for ISIC2018 is **train = 2075, val = 519** (see `splits/isic2018/`). There is **no** ISIC2018 test split. Everything the manuscript and the aggregate tables label as ISIC2018 "test" is in fact this **519-image validation set**. ACDC and Synapse, by contrast, have genuine held-out test splits (ACDC test = 366, Synapse test = 1568) and carry no such caveat.

## The naming inversion (read before tracing ISIC artifacts)

The same 519 ISIC validation cases are referred to under **two opposite suffix conventions** in the source artifacts:

| Artifact family | Suffix for the 519 real-validation set | Stale / unused artifact |
|---|---|---|
| Aggregate evaluation JSON (`outputs/evaluations/isic2018_*_*.json`) | `_test.json` (`split: test`, `num_samples: 519`) — **this is what the main tables use** | `_val.json` (`num_samples: 389`) |
| Per-case prediction directories (`outputs/evaluations/predictions/isic2018_*_*`) | `_val/` (519 `.npy`) — **this is what the cluster analysis uses** | `_test/` (390 `.npy`) |

So:

- `results/main_test_means.csv` for ISIC2018 derives from `isic2018_*_test.json` = the 519-image validation set (the `_test` label is the misnomer).
- `analysis/generate_rescue_analysis.py` sets `split = "val"` for ISIC2018 and reads the `isic2018_*_val/` prediction directories = the same 519 validation cases.

Both therefore evaluate the **identical 519 validation images**, despite the opposite suffixes.

## Stale artifacts (present in the working tree, not shipped, not used)

The `_val.json` aggregates (389 cases) and the `_test/` prediction directories (390 cases) are leftovers from a **superseded seed-1 ISIC split** (1815 / 389 / 390). They do **not** feed any reported number. The stale top-level seed-1 split file itself is **not** included in this release; only the authoritative seed-2026 manifests in `splits/` are shipped. See also `docs/config_of_record.md` for why the `dataset.split_file` field inside the config-of-record snapshots is vestigial.

## Why not just rename?

Renaming would alter artifact filenames that are referenced verbatim by the `source_artifact` columns of the shipped tables and by the analysis script. Per the reproducibility goal of this repository, the artifacts are preserved as-produced and the caveat is documented instead.
