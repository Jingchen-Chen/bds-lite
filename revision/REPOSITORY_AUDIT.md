# Repository audit — 2026-11-02

Authority: `/home/jc/Downloads/plan.md`, preserved in `revision/plan.md`. Active development root: `/home/jc/bds-lite`. Historical root PLAN.md targets BSPC and is superseded for this task; it and AGENTS.md remain unchanged. No missing anchor was reconstructed.

## Located sources

| Absolute path | Observed content | Authority decision |
|---|---|---|
| /home/jc/bds-lite | 13,125 research files, 34,141,451,575 bytes before migration; source/tests, data, 236 best.pt paths under results/outputs; no .git | Primary working repository. Historical checkpoints/aggregates require per-family validation. |
| /home/jc/bds-lite-new | 575 files, 66,095,796 bytes; no src/data/.git | 574 corresponding files SHA256-identical to primary, one additional path. Incomplete snapshot, not a self-contained clean GitHub release. Exact original purpose cannot be proved from names or dates. |
| /home/jc/bds-followup | 32,698 research files, 16,661,978,246 bytes excluding environments/caches | Most complete follow-up source/protocol/prediction records. Separate exploratory research family; not automatically revision evidence. |
| /home/jc/bds-latex | 24 files, 29,207,943 bytes; main.tex/main.pdf/bibliography/template/submission ZIP | Candidate submitted PeerJ source. Original reviewer PDF and editorial decision email not found here. |

Directory search covered /home, /mnt, /media and /tmp to depth 4–5. No separately named fourth historical bds-lite found. The primary already contains historical experiments and archives; whether it is the previously preserved copy is unresolved. Do not claim all historical copies were located. No mounted external disks found under /mnt or /media.

## Git

Only followup has .git. HEAD d55a171, branch exp/boundary-attribution-20261017; main/origin/main 3104cb3; tag research-base-20261017. Remote https://github.com/Jingchen-Chen/bds-lite.git. Full reachable history preserved in audit/followup-history.bundle, log in followup-git-log.txt. Numerous untracked monitoring records and phase-baseline outputs were present and migrated too. No source branch, remote, index or history was changed; no network push.

## Component authority

Retain primary model/training implementation pending checkpoint parity checks. Preserve follow-up scripts/source/configs/tests in recovered/followup; do not replace primary with an architecture selected because of favorable validation metrics. Preserve lite-new alternatives in recovered/lite-new. Preserve submitted manuscript in recovered/peerj-submission; older BSPC manuscript snapshots remain immutable under archive/.

Scientific evidence is ranked: mask/checkpoint + actual resolved config + exact split + predictions + per-case metrics + training log, then aggregates, then prose reports. SHA256 proves byte identity only. Reports describing completed work are not completion evidence by themselves. Existing summaries conflict about data/checkpoint availability; each describes a past checkout state.

## Verified checks

- Every inventoried research file was SHA256 hashed before migration; full names, sizes, links and comparisons in audit/inventory.csv and corresponding_files.csv. Environments/build/cache directories explicitly excluded and recorded as such; not a forensic inventory of every cache file.
- Migration: 31,176 new content copies verified against source inventory SHA256; 1,750 internal relative symlinks avoid duplicate bytes. Complete mapping: audit/migration.csv.
- Primary tests: 59 passed. Frozen interpreter /home/jc/bds-followup/.venv/bin/python, Python 3.14.4, torch 2.11.0+cu130, CUDA runtime 13.0; driver 595.99.02 reports capability CUDA 13.2. No package upgraded.
- All 12 followup U/S/B/F runs: exact 519 prediction IDs, uint8 224x224 predictions, selected checkpoint and last checkpoint present; all 6,228 DSC/IoU rows recomputed with zero difference. This does not certify HD95/BF1, training reproducibility or official-test generalization.
- ISIC primary train has extra ISIC_0015312(1).npz outside the 2,075-ID manifest. It is preserved. Val and ACDC/Synapse split membership checks pass; see validation/split_compatibility.csv.
- Seed-1 primary p1b U and full BDS checkpoint tensors equal the corresponding recovered October U/F tensors. Thus these cannot be counted as independent additional trained seeds or independent replication merely from directory names.
- No final_metrics.json found in legacy research runs; legacy completion must not be inferred from the new queue's marker convention.

## Open validation and risks

Original ACDC physical metadata ambiguity is documented in recovered/followup/research/boundary_attribution_20261017/ACDC_GEOMETRY_AMENDMENT_20261018.md: header spacing versus affine mismatch and unknown units. Do not report verified millimetres, EF or mass from it. Primary NPZs contain image/mask/sample_id, without original geometry; native recovery exists only for a development subset. Synapse raw H5 exists in primary, contradicting older followup statements that raw data were unavailable; physical spacing remains unverified.

ISIC identity leakage findings and all closed-route STOP decisions are preserved. Validation-based screens do not provide independent-test significance. Stage results by exact protocol family; never pool historical 390/389-image pseudo-tests with 519 validation or official 1,000 test.

Full-tree Ruff initially failed in frozen downloaded literature/alternate snapshots. Those bytes were preserved. ruff.toml explicitly excludes recovered/ from active-code linting, keeping src/scripts/analysis/tests and active results scripts checked. Exact failures and active checks retained in audit/. This is a source-snapshot scope distinction, not a relaxation of experimental validation.

## Post-migration validation update

All 32,926 migrated paths rehashed successfully. Of 46,422 original research files, 46,420 remain byte-identical; two live monitor JSONs changed during audit (timestamps Nov2 17:14/17:20), with old and new versions preserved. See audit/live_monitor_drift.json; no scientific data/checkpoint/source drift found. Original monitors were not interrupted.

The May/October U seed1 pair has separately dated 150-epoch logs, validation trajectories equal within floating precision, identical selected tensors and passing DSC/HD95 tolerances. This is verified reuse of an existing completed rerun, not additional seed evidence or fresh November training. See validation/reused_reproduction.json.

Official U/S/F validation/test evaluation and independent output validation are now complete; see OFFICIAL_ISIC_FINDINGS.md. Legacy GSL versus Phase16 trajectories diverge in all nine pairs; surviving legacy checkpoints cannot regenerate Phase16 predictions. Attention checkpoints use training seed2026. See GSL_LINEAGE_AUDIT.md.
