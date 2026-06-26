# Traceability: tracing any reported number to its evidence

This repository is built so that every number in the paper traces to a config, a script, and a machine-readable artifact. Four resources cover that trail:

1. **Claim → evidence.** `manuscript/tables/table6_claim_evidence_final.csv` maps each manuscript claim (and each explicitly *not-claimed* statement) to its supporting evidence and source artifact.

2. **Paper item → artifact.** The "Reproduce map" table in [`../README.md`](../README.md) maps Tables 1–6 and Figures 1–6 to the shipped artifact of record.

3. **Original path → public location.** The shipped tables and result CSVs keep their `source_artifact` columns **verbatim** (the exact path in the author's working tree where each number was computed, preserving the provenance trail and the verified file hashes). [`provenance_crosswalk.md`](provenance_crosswalk.md) maps every such original path to its public-repo location, and lists what was deliberately not shipped.

4. **Methods and caveats.** [`statistics.md`](statistics.md) (cluster-aware methods + bootstrap-RNG caveat), [`config_of_record.md`](config_of_record.md) (resolved snapshots; the vestigial `split_file`; the `bds_lite` alias), [`naming.md`](naming.md) (the ISIC2018 `_test` = validation caveat and the prediction/aggregate suffix inversion), and [`zenodo.md`](zenodo.md) (large binaries) document how the numbers were produced and the known caveats.

## Verified reproductions

- The cluster-aware analysis reproduces its artifacts of record **bit-for-bit** (identical sha256), including the bootstrap CIs when the script is run end-to-end; see [`../analysis/README.md`](../analysis/README.md).
- The GSL comparator's surface term is **bit-identical** to its Apache-2.0 upstream (MIST); see [`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).
- Split manifests are seed-2026, subject-disjoint, sha256-stamped; guarded by `tests/unit/test_split_integrity.py`.
