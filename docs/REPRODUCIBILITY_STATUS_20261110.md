# Reproduction status at the scientific editorial revision

Status: incomplete for portable training and inference; selected statistical
reanalysis is supported by the existing case-level outputs. This document is
documentation of unresolved dependencies, not an evidence freeze or lockfile.
The editorial cutoff is 2026-11-10 13:27 UTC. E3 is ongoing and excluded.

## What is available and what is not established

| Layer | Verified local evidence | Remaining limitation / required action |
|---|---|---|
| Statistical reanalysis | Completed-run case CSVs, registered estimator source, saved contrasts and independently recalculated numerical receipts | Selected verification does not certify every historical analysis; retain exact aggregation, missingness and multiplicity definitions |
| Inference | Local evaluation records name prediction and checkpoint SHA-256 values | The selected evidence archive omits checkpoint/prediction bytes; a hash cannot reconstruct missing bytes. An independent authorized backup and retrieval instructions are still needed |
| Training inputs | Locked manifests and processed arrays exist locally | Raw data releases, access rights and preprocessing dependencies must remain obtainable. Original third-party ACDC conversion and the earlier trainer are not fully recoverable |
| Software | `pyproject.toml`, `revision/env_freeze.txt`, local environment | Minimum requirements and a partial inventory do not reconstruct the actual environment. The overlay described below depends on other directories |
| Teacher | Source pins SMP 0.6.0 and pretrained-weight identifiers/hashes in `scripts/kd/common.py` and `scripts/kd/teacher.py` | Teacher dependencies, transitive versions and offline pretrained-weight/cache acquisition must be documented and tested in a clean environment; this has not been performed |
| Public release | Manuscript and release sources are prepared locally | A Zenodo DOI and deposited bundle have not been established. Deposition, content verification and author release decisions remain outstanding |
| Evidence inventory | Existing selected archive/checksum files remain unchanged | `revision/BDS_LITE_EVIDENCE_FREEZE.md` is provisional until all experiments finish and the author authorizes finalization. No new freeze/tag is made here |

## Actual environment dependency

The local file
`.venv/lib/python3.14/site-packages/_bds_frozen_overlay.pth` adds:

```text
/home/jc/bds-followup/.venv-kd/lib/python3.14/site-packages
/home/jc/bds-followup/.venv/lib/python3.14/site-packages
/home/jc/bds-followup/data/tools/acdc_audit_20261018/site
```

Thus `.venv` is not self-contained. Merely installing this repository with
`pip install -e .` does not establish reproduction of the environment used by
the completed experiments. No package installation or training-environment
change was made during this revision, to avoid affecting ongoing training.

## Scientific conventions needed for reanalysis

- Checkpoint selection uses the first maximum validation DSC, averaged across
  foreground classes per slice/image and then slices/images, including
  both-empty classes as 1. Reported analyses exclude both-empty pairs and
  recipe-level ACDC/Synapse contrasts average slices within patient/case.
  Whether another selection rule changes checkpoint rankings is unknown.
- HD95/ASSD omit undefined one-sided-empty pairs in the primary analysis.
  Diagonal-penalty sensitivities are alternative estimands, not measured
  distances for missed structures. Distances are resized-grid pixels.
- ACDC reconstruction is validated on development/validation data. Its
  effect on official patients is unknown; the roughly -0.02 official/internal
  score shift combines different cohorts and preprocessing pipelines.
- Image-ID separation does not prove ISIC lesion/patient independence.
  Synapse numeric-label consistency does not establish anatomical names.
- The registered primary estimator and its original decisions remain in
  force. Its residual duplication is model-dependent variance inflation;
  post hoc corrected/paired/bootstrap intervals do not replace the primary.
- Follow-up reuses examined patients; C1/C4 remain unresolved after Holm
  correction. One teacher per configuration and changed band geometry prevent
  isolated capacity/pretraining attribution. Full-image KD has a different
  effective loss scale at the same coefficient.

## Actions still needed

Before claiming portable reproduction, inventory the exact data and artifact
dependencies, provide lawful independent storage/retrieval, supply a complete
resolved environment with pretrained-weight acquisition instructions, and
validate it in a clean environment under separately authorized work. Before
claiming public archival availability, deposit and inspect the actual release.
None of those outstanding steps is represented as completed by this revision.
