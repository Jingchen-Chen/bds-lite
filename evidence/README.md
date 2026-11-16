# Evidence and supporting records

Supplementary S16 describes the frozen evidence archives. Their original filenames, bytes and
member-checksum manifests are preserved:

| Archive | Scope |
| --- | --- |
| `primary-evidence-20261110.tar.gz` | Per-run outputs, configurations, split manifests and registered analysis inputs |
| `final-supplement-20261110.tar.gz` | Final supplementary results, including the completed augmentation comparison |
| `addendum-20261115.tar.gz` | Post hoc gate-identity evaluation of the unweighted ACDC reference (S19) |
| `supporting-records.tar.gz` | Additional case-level outputs, protocols and provenance previously distributed as loose files |

[The evidence freeze](BDS_LITE_EVIDENCE_FREEZE.md) and the teacher-evaluation and review-round
addenda describe their scope. The supporting archive is a packaging change: it preserves the
existing records without recomputing or replacing a result. Files already present byte-for-byte
in an original archive are referenced there instead of copied into the supporting archive.

## Verify or restore

```bash
# Standard library only: check all packaged supporting records, without extraction.
python evidence/restore.py --check

# Restore all records at the paths expected by the frozen study scripts.
python evidence/restore.py

# Or restore only a selected group.
python evidence/restore.py --group protocols
python evidence/restore.py --group results
python evidence/restore.py --group provenance
```

[archived_paths.csv](archived_paths.csv) gives the destination, source archive, member name and
SHA-256 of every restored record. Existing files must match; different local files are never
overwritten. Restored paths are git-ignored. `--destination DIR` restores into a separate directory.
The adjacent `.sha256` files list member hashes; `supporting-records.archive.sha256` records the
whole supporting-archive digest (run `sha256sum -c supporting-records.archive.sha256` from `evidence/`).

The two independent numerical checks need NumPy/SciPy and no extraction:

```bash
python analysis/revision/verify_archived_statistics.py
python evidence/verify_snapshot.py
```

## Study index

Protocol and result paths below become available after restoration. The analysis code and locked
configurations remain directly available in the checkout.

| Paper | Protocol directory | Analysis / results |
| --- | --- | --- |
| Study 1; S6–S11, S13 | `protocols/ablation/`, `protocols/comparators/`, `protocols/sensitivity/` | `analysis/revision/`; `results/tables/families/` |
| Study 2; S13 | `protocols/expansion/`, `protocols/confirmation/` | `analysis/revision/expansion_statistics.py`, `confirmation_statistics.py` |
| Study 3; S17 | `protocols/kd_isic2018/`, `protocols/kd_acdc_synapse/` | `analysis/kd/`, `analysis/kd3/`; `results/kd_unified/` |
| Study 4; S18 | `protocols/acdc_confirm/` | `scripts/acdc_confirm/analyze.py`; `results/acdc_confirm/` |
| Study 5; S21–S22 | `protocols/followup/` | `analysis/followup/`; `results/followup/` |
| Historical correction; S4–S5, S15 | `protocols/historical_reevaluation/`, `protocols/prediction_recovery/` | `docs/provenance/`; `results/historical_reevaluation/`, `results/official_isic/` |
| Volume scoring and gate intervention; S9, S19 | `protocols/volume/`, `protocols/gate_identity/` | `results/volume/`, `results/prediction_recovery/`, `results/gate_identity_unweighted_acdc/` |
| Teacher and other post hoc analyses; S20, S23–S24 | Scope documented in the supplement and evidence addenda | `analysis/teacher_eval/`; `results/teacher_eval/`, `results/posthoc_review_20261112/`, `results/posthoc_restructure/` |

The restored `protocols/README.md` gives the complete protocol/queue index. Timestamp proofs,
withdrawn-run records, deviations and historical evaluation corrections remain preserved.
[evidence/path_map.csv](path_map.csv) maps development paths to this release. Development commit
identifiers refer to the private development repository and cannot be resolved in this snapshot.

Dataset images, complete checkpoints and teacher caches are not included. See
[the reproduction guide](../docs/reproducing.md) for extraction of per-run data, timestamp checks,
training requirements and paper builds.
