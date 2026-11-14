# Scientific-audit corrections — 10 November 2026

## Scope and assessment

The latest independent audit was checked against implementation and primary
completed-run evidence before editing. This is an editorial correction and
verification record, not an evidence freeze. The editorial cutoff is
2026-11-10 13:27 UTC. E3 had resumed and remained ongoing then; no E3 or
Experiment C test metric is inspected or incorporated. No training, inference,
queue operation, experimental-result edit, statistical-method change, freeze
tag or push is performed by this task.

Assessment: the revised documents are scientifically consistent with the
checked completed evidence, conditional on the disclosed design and estimator
limitations. They are not a complete independent reproduction package or a
submission-ready release. Author decisions and pending deposition remain.

## Independent disposition of reported issues

| Issue | Reverification and classification | Correction / remaining boundary |
|---|---|---|
| Historical ISIC contamination | VERIFIED ERROR, historical/high. Independent manifest intersection reproduces 358 training, 84 selection and 77 held-out images among 519 evaluated IDs | The existing withdrawal is retained in main Data and S4. The letters' obsolete claim that main text did not mention it is corrected. No historical experimental output is changed |
| Band loss affects only contour-adjacent errors | VERIFIED ERROR, interpretive/moderate. `scripts/kd_mc/loss.py:36-48` masks the loss, while shared trainable parameters receive its gradients | Main Methods/Discussion and reviewer/editor explanations distinguish direct loss location from effects elsewhere. Reference-error contours and teacher-defined KD bands are distinguished |
| Sibling teacher has no additional information | UNRESOLVED mechanism, moderate. `src/bds_lite/models/bds_lite.py` and the detached loss target establish shared inputs and an auxiliary objective, not identical representations | Replaced by absence of additional external data; learned redundancy remains a hypothesis |
| Capacity/pretraining joint attribution | UNRESOLVED mechanism, moderate. Teacher replacement also changes realization, probability targets/calibration and predicted band geometry; one teacher per configuration | Describe a comparison of teacher configurations. Compact teacher's observed Boundary F1 below its students does not prove absence of useful information |
| Approximately -0.02 shift caused by reconstructed preprocessing | VERIFIED ERROR in follow-up summary, moderate. `posthoc_stage_shift.csv` compares different cohorts and pipelines; `preprocessing_impact.json` measures reconstruction only on validation data | Main/supplement/summary now describe an inseparable cohort/pipeline shift. Validation maxima remain 0.000112 Boundary F1 and 0.0000522 DSC; official-patient effect is unknown |
| Checkpoint selection includes both-empty classes | VERIFIED CORRECT implementation, moderate reporting limitation. `runner.py:46-54` and saved validation CSVs reproduce selection scores | Selection is now defined separately from both-empty-excluded reporting and patient/case recipe means. Alternative checkpoint rankings remain unknown |
| Duplicated residual in registered SE | VERIFIED model-dependent variance inflation, moderate; arithmetic is implemented as registered | Preserve estimator, intervals and original decisions. Expected SE-squared inflation under the stated model is distinguished from a universal interval-coverage guarantee. Post hoc alternatives remain separate |
| Full-image control said to be absent | VERIFIED ERROR, stale/low-moderate. Completed E1 outputs and launcher establish the control | Corrected S1 taxonomy and active documentation; historical protocols/boundaries remain unchanged |
| E3 said not restarted | VERIFIED ERROR, stale/low-moderate. Read-only process observation at cutoff showed resumed registered E3 activity | Main, S21 and both letters use a dated ongoing/incomplete status; no moving result or new completed-run claim is substituted |
| Nominal C4 interval excludes zero but Holm test does not | VERIFIED CORRECT arithmetic; no contradiction | Retain nominal interval, adjusted p and unresolved label. No equivalence or absence conclusion follows |
| Environment / Zenodo / large-artifact availability | INCOMPLETE, moderate-high. Actual overlay resolves core packages outside this repository; DOI remains a placeholder and selected archive omits large dependencies | Main, S16 and active READMEs disclose the gaps. New reproduction-status document explains what remains needed; no clean-environment reproduction or deposition is claimed |
| Metric morphology, undefined distances and label independence | VERIFIED implementation conventions with UNRESOLVED provenance/interpretation | Explicit cross-element Boundary F1 versus square KD-band definition; retain missing-distance, ISIC lesion-identity and Synapse anatomical-name caveats |

The historical Boundary F1 implementation bug described in S4 was not
reconstructed from the unavailable original trainer. The present implementation
correctly returns zero for nonempty boundaries with no matches. No new claim of
independent reconstruction of that historical bug is made.

## Numerical verification

New CPU-only verifier: `analysis/revision/verify_scientific_revision.py`.
Receipts: `revision/validation/scientific_revision_numbers.json` and `.csv`.
Only path resolution is imported from existing analysis; aggregation, variance,
intervals, tests, Holm corrections and sensitivity calculations are implemented
independently. No model is executed and no experimental output is written.

Final result: **1,794 scalar comparisons, zero mismatches**, maximum absolute
error **3.752553823233029e-14**, tolerance **1e-10**. Coverage:

- 25 ACDC confirmation/follow-up contrasts and 20 follow-up Holm values;
  nine confirmation per-class contrasts.
- 60 ISIC all/fresh-seed contrasts and 60 Holm values.
- 120 stage-1 dataset contrasts and 40 pooled contrasts.
- 15 original exploratory KD contrasts, three balanced display contrasts,
  and three retained original-display contrasts.
- 165 matched descriptive metric cells, independently checking mean and SD.
- Ten post hoc crossed-model/bootstrap rows; 68 completed student histories
  and first-maximum validation selection checks.
- 218 evaluation artifact hashes and corresponding checkpoint hashes;
  receipts identify 590 unique primary input files.

No numerical scientific correction was warranted. Retained headline Boundary F1:

| Contrast | Difference [nominal 95% CI] | Decision |
|---|---|---|
| ACDC confirmation, band KD - D | +0.017820 [+0.009829, +0.025812] | Primary positive; size relative to 0.01 unclear |
| Full-image KD - D | +0.010564 [+0.003445, +0.017683] | Exploratory, Holm p=0.019625 |
| Compact-band KD - D | +0.011829 [+0.003706, +0.019952] | Exploratory, Holm p=0.019625 |
| Band - full-image | +0.007256 [-0.000519, +0.015031] | Unresolved, Holm p=0.094549 |
| External-band - compact-band | +0.005991 [+0.000077, +0.011906] | Unresolved, Holm p=0.094549 |
| Retrained ISIC full recipe - D, ten runs | -0.032060 [-0.051768, -0.012353] | Fresh-seed Holm p=0.031447 |

The registered ACDC lower bound remains below 0.01. Seed-paired and crossed-model
post hoc lower bounds above 0.01 do not replace that conclusion. Full-recipe
ISIC differences are not assigned to distillation alone. Teacher configurations
and full-image/band normalization remain confounded for mechanistic inference.

## Independent manuscript review and additional corrections

A separate reviewer examined the revised sources and primary references without
editing shared files. The main corrections were supported. Seven additional
points were corrected: residual "one property" wording in the response,
metric-specific teacher-performance qualification, KD loss versus gradient
magnitude wording, old word counts labeled historical, descriptive teacher
evaluation included in the ACDC cohort description, checkpoint/threshold-specific
selection wording and the unavailable original ACDC converter, and exact metric
morphology. The separate figure exporter was also found to use an older figure
order; its exports now follow architecture, confirmation and follow-up. Source
figure measurements are unchanged. S20's stale hard-coded main-equation number was replaced by an unambiguous description of the recipe-level SE equation.

## Rebuild, preservation and checks

`bash revision/build_revision.sh` rebuilt main/supplement/response and the editor
letter, with 118 resolved locations. Tracked changes were rebuilt using pinned
latexdiff 1.3.4. Final PDFs are 18/60/17/28 pages (main/supplement/response/marked).
Main statistics and limitations/availability pages were visually inspected.
All four PDFs pass extracted-word page-bounds checks. Clean PDFs have no
unresolved references/keys or overfull-box diagnostics; underfull warnings remain.
The marked copy retains latexdiff typography and bibliography rerun warnings;
its extracted words remain inside physical page bounds.

`scientific_revision_consistency.json` records 643 protected tracked paths and
1,108 selected primary files unchanged; 409 used numeric bindings resolve.
Only presentation word-count bindings changed. All reviewer quotations and
comment IDs are preserved. The current main prose count is 7,291 words versus
5,761 submitted (+27%); no claim of reduced length is made.

Repository checks: `.venv/bin/ruff format --check .`, `.venv/bin/ruff check .`,
and CPU-only `.venv/bin/python -m pytest` with a disk-backed TMPDIR. Ruff passes;
pytest reports 147 passed, one existing optional skip. No test is modified.

Intermediate failures are not scientific discrepancies: the new checker first
used a nonexistent `all_se` CSV field and merged two distinct display families;
schema/family handling was corrected without changing assertions/tolerances.
New-checker Ruff defects were corrected. Two obsolete response anchors stopped
builds and were updated. An optional historical root README was absent, so the
existing active READMEs were used. A document scanner initially included a TeX
comment's placeholder key; it now scans actual content. Poppler emitted XML-illegal
text control characters; only those controls were removed before parsing, with
all word boxes still checked. The editorial manifest initially used CRLF line endings and failed the Git whitespace check; LF-only packaging preserves its data fields and resolves that failure. Exact failures and continuations are in PLAN.md's
phase-20 status block. No experimental source or frozen result was repaired to
make a check pass.

## Remaining limitations and author actions

The registered estimator assumptions/coverage, few-run stage-1/2 uncertainty,
post hoc dataset selection, same-patient follow-up, one teacher per configuration,
loss-scale differences, official preprocessing impact, checkpoint-selection
sensitivity, lesion identity, anatomical label provenance, comparator adaptations
and external-literature protocol differences remain limitations. E3 and Experiment C
are incomplete/excluded; their unresolved questions are not answered here.
Portable reproduction, independent storage, actual Zenodo deposition, author
affiliation/approval and release decisions remain outstanding. The provisional
draft and existing archive are not finalized or retagged by this work.

## Local commit blocker

The repository-required ordinary correction commit failed with `Author identity
unknown` and `fatal: unable to auto-detect email address` (exit 128). No commit
was created, no identity/configuration was invented, and no hook was bypassed.
The verified changes remain staged. The intended explicit-path correction commit
excludes PLAN.md and all pre-existing provisional archive/boundary files. Human
action: configure the intended Git author identity before committing reviewed
editorial changes. This packaging blocker does not change the scientific
verification results. No freeze, tag or push was performed.
