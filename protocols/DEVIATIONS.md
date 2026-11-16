# Deviations and protocol compatibility

No new revision training run has been executed. The locked experimental protocol has not been changed.

- Historical root PLAN.md targets BSPC and conflicts with current user-authorized PeerJ roadmap. Preserve it; execute revision/plan.md, as instructed explicitly by user.
- Original primary Git metadata was absent. Followup Git history is preserved separately; any initialized primary history will record consolidation, not pretend to recover missing ancestors.
- Followup config declares zscore, but the released Dataset applies no zscore: stored ISIC RGB is bilinear-resized /255 CHW. Official preprocessing matches this actual behavior; exact raw-to-stored parity was verified. Config text alone must not determine preprocessing.
- Followup gradient measurements at epochs 0/25/75/150 do not satisfy every-50-step revision sampling; retain as exploratory historical observations.
- Followup ACDC context/phase-reduction protocols use different sampling/reduction/augmentation; do not pool into no-augmentation matched revision runs.
- Primary ISIC train includes an extra copy-like filename outside manifest; exact manifest filtering is required for future runs. Nothing deleted or retrospectively relabeled.
- Current historical runs use summary.json/status.json/per-image .npy rather than new final_metrics.json/preds_split.npz; preserve their schema and separately audit completeness. Never add fake completion markers.
- Physical geometry ambiguity blocks verified mm/clinical results. Index-grid distances must remain labeled as such.
- Initial full-tree lint included immutable recovered literature/snapshots and failed. ruff.toml explicitly scopes active lint to active code, preserving recovered bytes. Failure logs retained; no model/test assertions weakened.

- Two live operational monitor JSONs changed externally during migration verification; both pre-migration snapshots and observed later versions preserved. All destination hashes match and all other inventoried original files remain unchanged.
- Historical CSV exports containing literal escaped CRLF are decoded in memory for verification; source bytes remain preserved.
- NCHW contiguous input is necessary for exact saved-mask parity: transposed NumPy stack layout produced 59 different pixels in an eight-image check; contiguous layout yielded zero across all nine fixed models. Initial failure and corrected evidence preserved.
- Git whitespace checks explicitly exempt byte-preserved raw audit logs, prior report backups and captured source HTML via .gitattributes. Active code and generated scientific tables retain whitespace checks. The final formatter found two whitespace-only lines in the new qualitative script after import fixing; these were removed and checks rerun. No data value changed.
- The volume surface-eligibility clarification was first committed as an appended paragraph to the calculation protocol, then moved into volume_statistics_protocol.md to keep the calculation document immutable. All20 completed volume cohorts retain the original computation-protocol SHA256 (5a80b1e949f2d5a8ebba5f0822312304bb2164618ac763b19ed5730ac3c6370b); no stored numerical result used the transient document hash. No volume calculation or result changed.

- Gate inference was registered before prediction generation. Its statistical amendment was committed after prediction generation began and before comparative effect inspection; findings are explicitly exploratory rather than original preregistration. No training-time M5 completion is inferred.
- New report-generator lint found long lines and Python3.10-incompatible f-string escaping; corrected before producing its findings. One shell invocation used unavailable `python`; no files changed in that invocation, rerun used the recorded environment. Intermediate protocol-only commits had unrelated pending script style warnings; final active-code checks apply to the complete results commit. No pre-commit hook was bypassed.

- New diagnostic configs explicitly describe current active model semantics and omit stale unused legacy switches; ACDC class weights, all loss weights, preprocessing and frozen manifests are retained. The first engineering GPU check exposed a constant-zero SDF with no autograd graph on an empty Synapse batch. Probe now records zero norm and undefined cosine, with a regression test; original failure log retained. These engineering updates are not scientific experiments or changes to the locked optimization protocol.
- New resume is at epoch boundaries, retaining the released runner's per-epoch GradScaler reset. No mid-epoch continuation is claimed. Save epoch1/every10 and on requested stop; crashes can replay unsaved epochs. Exact CPU resume and all three full-batch GPU next-update neutrality checks passed before queue registration.

- Reporting audit caught letter-suffixed WP tasks omitted by build_status.py's overview grouping, which made empty WP1/WP5 groups appear complete via all([]). Corrected generated overview and reviewer mappings with analysis/revision/refresh_status.py; neither WP is complete. The old reporting generator is preserved in the running training-source snapshot, with replacement entrypoint documented. No scientific status was relaxed.
- Real queue acceptance intentionally requested an epoch-boundary stop at ISIC epoch2. Exit130 is expected, with original queue traceback/log retained; complete optimizer/scheduler/best/RNG/shuffle state was saved. The same registered run resumed at epoch3, and logged epochs are contiguous without duplicates. No new seed or completed experiment credit assigned.

- ISIC diagnostic reached the unchanged85C guard and saved epoch24 before exit130. Added a separate operational supervisor (training source/config hashes unchanged): cool to<=70C, then use the same queue/resume command. Only a verified recent thermal stop for the current run or the queue's cool-launch guard is eligible; unknown failures/occupied GPUs/explicit STOP are not retried. Attempts/cooldown are bounded and all logs preserved. No optimization setting changed.
- A new diagnostic scalar check found up to9.62e-6 discrepancy between dot products from separate component backward passes and a combined-loss backward pass in the first10 epochs. The initial strict check failed and is retained. Tensor inputs areFP32 with ambient cuDNN allow_tf32=true (CUDA matmul allow_tf32=false); exact component/combined accumulation equivalence is not certified. The report now records this residual explicitly as pending independent numerical validation, not a passing check. No recorded gradient value or training source was altered; mechanism claims remain blocked pending validation/completed trajectories.

- Surface CDF validation matches all saved valid HD95 cells within2e-5px (observed maximum7.63e-6 from float32 quantile rounding), with identical missingness and10 independent KDTree checks. Initial connected-line render interpolated discrete ECDF jumps; replaced by a separate right-continuous rendering script, preserving original PDF/PNG. No distance values changed.
- New reporting companions correct the legacy WP grouping and snapshot mutable runtime logs. Frozen training-source scripts are not edited during runs. Mutable supervisor files remain local and timestamped copies enter audit/runtime_snapshots; final reports explicitly distinguish observed epochs from completed experiments.

- 2026-11-05 follow-up to the gradient residual item above: `analysis/revision/gradient_residual_audit.py` quantified the residual over all probes of the three completed WP5 runs (acceptance fixed in the script before the summary was inspected). Maximum absolute residual 1.3e-4 (ISIC, later epochs; the earlier 9.62e-6 covered only the first 10 epochs), median relative residual <= 7.4e-5, maximum implied cosine error 6.1e-4 versus a smallest between-dataset gap of 0.139: immaterial for the reported cosines and P3. Exact separate/combined equivalence is still not claimed; the original failed strict check is retained.

- 2026-11-05 manuscript reconstruction: added post hoc, after results were known, a run-level margin (pooled within-arm variance of run means, excluding LB-UNet) and a minimum detectable effect on the original models. Reported the registered all-case distance sensitivity for the first time. Not part of prereg_revision.md except the all-case analysis; labelled in Methods. No model was trained or re-evaluated. analysis/revision/figure_dataset_examples.py and figure_qualitative.py changed labels only; their manifests are unchanged.

- 2026-11-06 expansion queue pause (operational, no training change): `isic2018_m6_full_seed8` stopped at the normal thermal guard (exit 130) after saving epoch 115 at 19:48:07 UTC. An unrelated Codex job (`train_autoencoder.py`, coursework, outside this repository) had started on the same GPU at about 19:47:41 UTC, so it overlapped attempt 1020 for about 35 s. `supervise_queue_v2.py` then exited at `cool()` with "GPU occupied; supervisor never interrupts or retries occupied work", as designed. 30 of 40 runs were complete. The unrelated job was left alone; the supervisor is restarted only after the GPU has had no compute process for 10 minutes. Sources, configs and queue are unchanged.

## ACDC decoder-only arm read from the wrong run directory (found 2026-11-07)

`analysis/revision/family_statistics.py` read ACDC M2 from `acdc_m2_aux_boundary_only_seed*`, the class-weighted
runs, instead of the matched unweighted runs `acdc_m2_aux_boundary_only_unweighted_seed*` used by every other ACDC
arm. `ACDC_RUN_ARM` now maps M2 to the unweighted directory, and the boundary-transfer probe follows it. Only ACDC M2
rows changed and no Holm decision flipped. M2 vs M6 BF1 went from +0.0052 to 0.0000; M2 vs M0 BF1 from 0.0198 to
0.0145, DSC from 0.0183 to 0.0146, HD95 from -1.52 to -1.65. The response letter must list this as a correction.

## Post hoc analyses added after the frozen Stage 1 analysis (2026-11-07)

`seed_subset_analysis.py`, `boundary_precision_recall.py` and `gate_structure.py` were not registered in
`EXPANSION_PROTOCOL.md`. They are descriptive, use only finished runs, and the manuscript labels them post hoc.

## Stage 2 validator written before any Stage 2 test metric was read (2026-11-07)

`analysis/revision/validate_confirmation.py` was committed with four of 70 runs finished. Its recomputation was
exercised only on runs already reported (ACDC and Synapse C1 seed 1, ISIC M0 seed 4). Its output is pass or fail only.

## Manuscript rewritten without Stage 2 at the authors' instruction (2026-11-07)

The authors asked for a self-contained manuscript from validated evidence, with no reliance on Experiment C. `main.tex` and `supplement.tex` therefore contain no Stage 2 text or slots, and no Stage 2 test metric was read. This departs from rule 5 of `NARRATIVE_CRITERIA_ADDENDUM.md` (every experiment started is reported); see `HUMAN_DECISIONS.md`. The manuscript avoids statements that Stage 2 would falsify, such as claiming no further runs exist. `analysis/revision/make_narrative_assets.py` only reformats frozen outputs. `analysis/revision/patch_architecture_figure.py` writes `figure_architecture_v2.png`; the authors' source PNG is unchanged. The text previously said Welch p <= <0.001 for the teacher head on all datasets; ACDC is 0.002, so it now says p <= 0.002.

## Historical re-evaluation frozen (2026-11-07)

The author chose to run the historical re-evaluation and the manuscript reorganization in parallel with Experiment C. `HISTORICAL_REEVALUATION_PROTOCOL.md` was frozen at 2026-11-07T08:24:14+00:00 before any official-test metric of the April ablation arms or October arms G, C, and D existed. Not committed, because repository lint fails on pre-existing issues in `analysis/revision/make_expansion_assets.py` and `seed_subset_analysis.py`. SHA-256 record instead: protocol 8e770fee3d4f9569, evaluator 4db910b3469d7c2c, statistics cef728a82013daa2. Inference only, on CPU, with pauses during GSL training.

## Manuscript reorganized around the title's question (2026-11-07)

`main.tex` and `supplement.tex` were reorganized after `revision/audit/project_audit_20261107/REPORT.md`; earlier versions are in `revision/audit/manuscript_snapshot_20261107_pre_reorg/`. Results now follow five parts of the title's question. Corrections: alpha and the loss weights are no longer described as fixed before the experiments, because `docs/devlog.md` shows they were set in single-seed validation runs; "pointed the same way on every dataset" was removed; the low retrained U-Net is stated for Synapse as well as ACDC; "nor 2.5D" became "nor matched 2.5D". New post hoc material from `analysis/revision/make_audit_assets.py`: ISIC membership decomposition, validation-image contrasts, development controls, and a record of trainings. Validation and development contrasts use the frozen estimator; the first version of the audit report gave run-level intervals only and was corrected the same day. No new training. Experiment C metrics were not read.

## Experiment C ISIC seed expansion cancelled by author (2026-11-07)

The author requested removal of ISIC2018 seeds 11–17 to reduce compute.
See `CONFIRMATION_SCOPE_AMENDMENT_20261107.md`. The active queue retains 56
ACDC/Synapse runs. Completed ISIC seed-11 artifacts remain preserved. The
original three-dataset confirmation analysis is no longer executable as
registered; an amended analysis must be frozen before test-metric inspection.
Original protocol, configs, source hashes, and experimental artifacts are unchanged.

## All remaining Experiment C training stopped by author (2026-11-07T11:58:35.404671+00:00)

After reviewing the previously reported seed-1–3 summaries, the author
requested stopping all remaining training. `revision/STOP_TRAINING` now
blocks queue execution and automatic supervisor resume. This is an
author-directed compute cancellation after training began; no Experiment C
test metrics were read for this action. Preserve completed extra runs and
partial checkpoints and disclose the cancellation. Original Experiment C
confirmation is incomplete; no confirmation claim is inferred. CPU historical
re-evaluation is a separate job and was not stopped by this training request.

## Historical re-evaluation completed and integrated (2026-11-07)

`analysis/revision/historical_reevaluation.py` finished at 2026-11-07T12:13:14Z; `validation.json` reports PASSED (63 folders, 0 failures, 51 of 51 parity checks passed). Before analysis the SHA-256 of the protocol and both scripts was rechecked and matched `FREEZE.json`. `historical_statistics.py` ran unchanged; its tolerance-2 check reproduced the frozen values. Integration follows protocol Section 9: three Results sentences, one Limitations sentence, full tables in Supp S15. Readings applied: H1 pooled BF1 negative (decoder direction did not hold across pipelines); H5 on the official test not negative (confirmed deficit stated as specific to the retrained set, also in abstract, Discussion and Conclusions); D6 negative with D7 including zero ("consistent with"); D4 and D5 undetermined (Supplement only); tolerance changes two statuses at 4 px (stated; 2 px statuses stand). No status, tier or registered family changed. Additions outside the frozen files: parity and validation keys (`HR.*`) and conditional evidence-class wording in `make_audit_assets.py`. Training was not touched; no Experiment C metric was read.
