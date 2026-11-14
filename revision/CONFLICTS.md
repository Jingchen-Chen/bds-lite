# Conflicts and decisions

1. **Roadmap/venue:** root PLAN.md is historical BSPC; current user explicitly designates Downloads/plan.md PeerJ roadmap authoritative. Preserve both; execute revision/plan.md. Do not enforce the old plan's training prohibition against newly authorized revision experiments.
2. **Missing anchor:** manuscript_v2/ANCHOR_ABSTRACT.md absent at primary path; do not invent it. Archived manuscripts are provenance, not a replacement human-approved claim boundary.
3. **Missing fourth historical root:** not found in searched locations. Main root may already be restored historical material, but evidence is insufficient to assert identity. Continue with discovered artifacts; no automatic global completeness claim.
4. **Source conflicts:** 15 followup files at corresponding primary paths differ. Full hashes in audit/corresponding_files.csv. Preserve followup source under recovered/followup, active primary untouched. Model equivalence must precede promotion.
5. **ISIC extra train file:** ISIC_0015312(1).npz is outside manifest; preserve and exclude via strict manifest loader, never delete or silently increase training population.
6. **Split metadata:** legacy root split seed field is 1, followup manifest seed is 2026. Actual memberships checked separately. Matching counts or labels do not prove preprocessing identity.
7. **Checkpoint duplication:** primary p1b seed-1 U/F tensor states equal recovered October U/F. Do not count path duplicates as fresh reproducibility evidence.
8. **GSL reports versus artifacts:** historical Phase 16 says nine matched runs completed. The current primary gsl_memory_bounded folder lacks promised checkpoint subfolders. Recoverable GSL CSVs/configs under recovered/followup/results/gsl require lineage validation; existing primary legacy GSL checkpoints also exist. Do not retrain until mapping is resolved.
9. **Physical geometry:** ACDC recovered development pixdim/affine ambiguity unresolved; primary NPZ geometry absent. Index-grid analyses may proceed with explicit units, physical/clinical analyses blocked pending trustworthy geometry. Synapse H5 availability is now confirmed; spacing provenance remains open.
10. **Raw snapshot lint:** downloaded third-party source and preserved alternatives fail full-tree Ruff; originals not reformatted. Explicit ruff.toml excludes recovered snapshots, active code checks remain required.
11. **Git:** primary had no .git; followup history preserved as bundle. Any new primary Git history is a fresh consolidation history, not the missing original historical checkout.
12. **Publication:** no result licenses for ACDC/Synapse figures approved. Prepare label-only fallback until human license approval. No journal contact, submission, push or external message authorized.

13. **Live operational drift:** all destination hashes pass, but two followup monitor JSONs changed during verification. Old/new bytes preserved; no scientific-source drift detected. See audit/live_monitor_drift.json.
14. **GSL family identity:** all150-epoch trajectory comparisons diverge across all nine pairs; Phase16 checkpoints remain missing. Do not substitute surviving P2 checkpoints.
15. **Attention seeds:** historical dataset checkpoints use seed2026 and do not close required seeds1/2/3 comparisons.
16. **Inference layout:** exact raw preprocessing alone was insufficient for saved-mask parity until NCHW contiguity matched the released loader. Failure/correction preserved; official outputs use the corrected path.
