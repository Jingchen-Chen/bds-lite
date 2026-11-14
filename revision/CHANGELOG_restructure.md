# CHANGELOG: restructure revision (branch `revision-restructure`)

This log is the basis for the response letter. Every manuscript change gets one entry with these
fields:
- **ID**
- **Phase**
- **Location**: file and section, plus the PDF line in the pre-change build
- **Original**: a summary of the old text
- **Revised**: a summary of the new text
- **Rationale**: the reviewer or editor concern, or the brief item
- **Numerical provenance**: the key → generating script → result file, or "none"

A number enters the manuscript only as a generated `\R{}` key or a generated table. A missing
value is written as `TODO(missing: …)`.

## Phase 1: audit (2026-11-09)

| ID | Location | Original | Revised | Rationale | Numerical provenance |
|---|---|---|---|---|---|
| R1-00 | none (no manuscript change) | — | Audit and plan in `revision/RESTRUCTURE_PHASE1_AUDIT.md` | Brief, Phase 1 | Reference numbers checked in audit §3 against `revision/manuscript/generated/numbers*.tex` and the result files named there |

## Phase 2: analyses run before the text edits (2026-11-09)

No manuscript text was changed in this step.

| ID | Location | Original | Revised | Rationale | Numerical provenance |
|---|---|---|---|---|---|
| R2-A1 | New files; manuscript text pending | Teacher test metrics only on the internal 20 patients, and not in the manuscript | ACDC teacher on official patients 101–150, **post hoc, descriptive**: Boundary F1 0.8778, DSC 0.8146, IoU 0.7426, HD95 2.61 px, ASSD 1.02 px (50 patients). Internal 20 patients (recorded 2026-11-07): Boundary F1 0.8859, DSC 0.8231. These are different test sets. | Brief 2.2; decision 7 | Script `analysis/revision/posthoc_teacher_official.py`. Checkpoint `results/revision_runs/acdc_kd_teacher_seed100/best.pt`, sha256 `735747303c444e13dd57457e3540ee410b5de43421b68fd92e2eda69c88dafa0` (equal to `test_metrics.json` and `cache_complete.json`). Outputs `results/acdc_confirm_eval_posthoc/acdc_kd_teacher_seed100/` (`evaluation_record.json` holds the artifact hashes). Aggregation with `expansion_statistics.unit_values`, written to `revision/restructure_posthoc/teacher_metrics.json`. Internal values: `revision/kd_unified/unified_results.json` → `results.acdc.teacher_test`. |
| R2-A2 | New file; manuscript text pending | Cache generation time "not recorded" | **Re-measured post hoc**: training-set cache pass over 1,350 slices, median 27.3 s over 3 repeats (28.0, 27.3, 25.1 s). The regenerated cache is bitwise identical to the stored cache. Consistency check: the original step's file modification times span 31.8 s for the whole step, which also includes the validation pass, saving and the check. | Additional decision "teacher cache generation time" | Script `analysis/revision/posthoc_teacher_cache_timing.py`; output `results/acdc_confirm_eval_posthoc/teacher_cache_timing.json` |
| R2-A3 | New files; supplement S20 and main-text wording pending | The SE was described as combining run and unit components | Size of the shared run×unit term. Rule fixed in the script before running: "negligible" if every Boundary F1 SE drops by less than 5%. **Result: not negligible.** SE reduction if the term is removed: ISIC2018 0.3–1.6%; Stage 3 6.7%; ACDC and Synapse in Stages 1–2 up to 26–50%. Simulation check (`analysis/revision/posthoc_shared_variance_simulation.py`, 20,000 replicates per scenario): SE² − shared term is unbiased for Var(Δ), and SE² overestimates it. Main-text wording is therefore "conservative", without "slightly". | Decision 8 | Script `analysis/revision/posthoc_shared_variance.py`, which reproduces every recorded Δ and SE to 1e-9 before reporting. Outputs `revision/restructure_posthoc/shared_component.{csv,json}`. |

**Supersedes R2-A3 numbers.** After the author chose Option B, `posthoc_shared_variance.py` was
extended (v2) in four ways:
- each variance component is truncated at zero;
- all five metrics are covered;
- the shared term is varied at 0.5× and 1.5×;
- a plug-in coverage simulation was added.

With truncation, the SE reductions are:
- ISIC2018: up to 1.3%;
- Stage 3: 6.7% (unchanged, because no truncation was needed);
- ACDC and Synapse in Stages 1–2: 6–24% (R2-A3 gave up to 26–50% without truncation).

The untruncated pooled D+F+G − D interval (+0.0041, +0.0161), which was reported to the author
before the decision, is kept in S20 as the first computation. The reading "not negligible" is
unchanged.

## Phase 2: manuscript text (2026-11-09)

The pre-change build is `main.pdf` at `bbf2d9e` (identical to `fa72602`). "Old l." means a line
number in that PDF.

**New keys** are generated in `revision/manuscript/generated/numbers_restructure.tex` (prefix
`RS.`) by `analysis/revision/make_restructure_assets.py`. That script reads only:
- `revision/tables/families/{expansion_interaction,expansion_components,expansion_contrasts,audit_development_controls}.csv`;
- `recovered/followup/results/acdc_context_audit_20261019/screen.json`;
- `revision/restructure_posthoc/{teacher_metrics.json,shared_component.csv,shared_component.json,coverage.csv}`;
- `results/acdc_confirm_eval_posthoc/teacher_cache_timing.json`;
- `revision/kd_multiclass/benchmark.json`;
- the stage-3 decoder config, used for the deployed parameter count.

### R2-S20 (dedicated entry): shared variance term in the recipe-level SE

**⚠ This issue and its post hoc sensitivity analysis must be disclosed proactively in the
response letter.**

| ID | Location | Original | Revised | Rationale | Numerical provenance |
|---|---|---|---|---|---|
| R2-S20a | Main, Statistics "Recipe-level contrasts" (old l. 244–252) | SE described as combining run variation and test-unit SE, with no formula | Explicit Eq. 7. The between-run and test-unit terms are explained, along with why run and unit main effects are not counted twice. The run-by-unit residual enters both terms, so intervals are "conservative" (no "slightly"; Supplementary S20). Welch df come from the run terms only; for Stage 1, the pooled within-cell form with ν = Σ(n_c − 1). | Brief 2.2 (explicit formula); decision 8; Option B | Formula from `analysis/revision/expansion_statistics.py:133–165, 211–227` (audit §2). "Conservative" follows the rule fixed before the first run (`shared_component.json` → `reading`). |
| R2-S20b | Supplement, new S20 | — | Model and derivation. Alternative estimator, disclosed as selected post hoc: residual mean square, components truncated at zero, pre-specified df. The untruncated first computation is also given. Then: size, stability (truncation counts), 0.5×/1.5× sensitivity, zero-exclusion counts, plug-in coverage simulation, limitations, files. Tables S56 (every Boundary F1 comparison + Stage 3 secondary metrics, both estimators) and S57 (coverage). The full CSV covers all 100 comparisons and 15 pooled rows across five metrics. | Option B conditions 1–4 | `RS.n.*`, `RS.red.*`, `RS.*.adj*`, `RS.*.untrunc.ci`, `RS.cov.*`; `tab_s20.tex` and `tab_s20_coverage.tex` ← `make_restructure_assets.py` ← `revision/restructure_posthoc/shared_component.csv` and `coverage.csv` ← `analysis/revision/posthoc_shared_variance.py`, which reproduces every recorded Δ and interval to 1e-9. Unbiasedness check: `analysis/revision/posthoc_shared_variance_simulation.py`. |
| R2-S20c | Main, Results "Does in-network transfer add?" | — | One sentence: post hoc sensitivity interval `RS.tr.nosurf.adj.ci` [+0.0009, +0.0182] (S20). States that the estimator was chosen after the results and does not change the pre-specified reading. | Option B, main text | `RS.tr.nosurf.adj.ci` ← `shared_component.csv` (stage 1, pooled, distillation+gate without surface, `low_k1`/`high_k1`) |
| R2-S20d | Main, Results Stage 3 | — | One sentence: corrected lower bound `RS.s3.adj.low` (+0.0104) given as a number only; post hoc; the decision and its reading are unchanged. It is not described relative to the SESOI. | Option B; F11 | `RS.s3.adj.low` ← `shared_component.csv` (stage 3, boundary_f1, `low_k1`) |
| R2-S20e | Abstract | "did not establish a gain over the decoder-only reference" | "did not establish a gain over D under the pre-specified estimator"; no corrected interval in the Abstract | Option B | — |
| R2-S20f | Main, Limitations, "Statistical power" | — | Sentence: some "undetermined" ACDC/Synapse results reflect this conservative SE (S20) | Option B | — |
| R2-S20g | `revision/CLAIM_BOUNDARY_v2_NOTE_20261109.md` (new) | — | Disclosure-only note: all readings stay pre-specified; corrected intervals are post hoc and change nothing; A1 and F11 are unaffected; the response letter must disclose the issue | Option B | — |

### Other Phase 2 changes

| ID | Location | Original | Revised | Rationale | Numerical provenance |
|---|---|---|---|---|---|
| R2-01 | Abstract | Previous abstract (decoder-only reference; "establishing a positive gain … crosses the SESOI") | The brief's draft, verified number by number, with these changes: (i) "under the pre-specified estimator"; (ii) the probe sentence made descriptive ("found added boundary information in the distillation projection but no detectable change in the features that make the prediction"), with no mechanism in the Abstract per decision 1; (iii) "On ISIC2018, between-run variation exceeded that threshold" (scoped); (iv) the confirmed ISIC2018 deficit of BDS-Lite in the retrained set added. The post hoc ACDC selection disclosure is kept. | Brief 2.5; Phase 1 corrections; decision 1 | `A1.nosurf.*` ← `expansion_interaction.csv`; `CF.*`, `OT.patients` ← `revision/acdc_confirm/results/results.json`; ISIC2018 SDs (Results) ← `XM.*` |
| R2-02 | Introduction (old l. 58–67) | Methodological detail of BDS-Lite in the fourth paragraph; three contributions | Fourth paragraph reduced to the design logic (switchable signals, two designs, fixed backbone). Contributions 1 and 3 merged; 2 contributions remain. | Brief 2.4 | — |
| R2-03 | Related work, Distillation (old l. 80–87) | No D+KD-band/BPKD relation | Sentence: D+KD-band is a simplified BPKD-style instance (Liu et al. 2024) keeping only a boundary-band term; it is an experimental test condition, not a contribution | Brief 2.2 | — |
| R2-04 | Methods, new "Arms and components" + Table 1 | Arms described in prose in "Training and experimental stages" (old l. 189–196); no arm table | New subsection and booktabs Table 1 (Arm × decoder/distill/gate/surface/external KD/deployed graph). One-letter naming: U, U+S, D, D+S, D+F+G, D+G+S, D+F+S, D+F+G+S (BDS-Lite), D+KD-band, plus the 4 comparators. Footnote: arms with α = 0 keep an inert identity gate and P; the deployed function equals U. | Brief 2.3, 3.2; decision 9 | Component settings: `revision/CLAIM_BOUNDARY_v2.md` §0.1; `expansion_statistics.CELLS`; `src/bds_lite/models/bds_lite.py:53` (gate identity at scale 0) |
| R2-05 | Methods order | BDS-Lite → routes → D+KD-band → Datasets → Training and stages → Evaluation | Arms → D reference → BDS-Lite (projection/gate, F, S, objective, three routes, first-order argument) → D+KD-band → Data → Training → Study design and evidence levels → Statistics | Brief 2.3 | — |
| R2-06 | Methods, BDS-Lite, after Eq. 3 (old l. 124–125) | "penalizes probability mass outside the object in proportion to its normalized distance" | Adds that, because φ < 0 inside, minimizing also rewards interior probability mass in proportion to its normalized depth | Brief 2.2 (Eq. 3 correction) | Sign convention: `src/bds_lite/training/losses.py:110–115` |
| R2-07 | Methods, BDS-Lite, surface loss | No comparison with Kervadec/GSL | L_sdf = Kervadec boundary loss with per image–class normalization by s_nk, fixed λ_sdf = 1, no schedule. The Boundary Loss comparator uses unnormalized pixel distances and a region→boundary schedule. GSL normalizes a distance-weighted error and anneals. | Brief 2.2 | `losses.py:127–161`; `scripts/comparators/components.py:47–76` (`kervadec_weights` schedule); S3 comparator table |
| R2-08 | Methods, D+KD-band | "… parameters on ACDC" only | Adds 12.7× the deployed student (1,926,532 parameters) and the BPKD sentence | Phase 1 correction (teacher size); brief 2.2 | `RS.KDC.ratio.deployed`, `RS.KDC.student.deployed` ← `make_restructure_assets.py` (build_model on `results/revision_runs/acdc_confirm_decoder_seed201/config.json`, encoder + seg_decoder; total cross-checked with `revision/kd_multiclass/benchmark.json` 2,197,493); teacher 24,436,804 ← `benchmark.json` |
| R2-09 | Methods, Data (old l. 158–182) | LiTS/BraTS justification in full; "Their conversion reconstructs third-party code …" | One sentence pointing to S3. The conversion sentence rewritten with an explicit subject ("we reconstructed it and applied the reconstruction to these patients"). Raj et al. no longer cited here (one citation, in the Introduction). | Brief 2.4 | — |
| R2-10 | Supplement S3 | — | New paragraph "Why LiTS and BraTS were not added" with the removed main-text text verbatim (citations as author names). Preservation was checked before removal. | Brief 2.4; additional decision | — |
| R2-11 | Methods, Training (old l. 186–188) | "All runs … use the released code on the documented partitions (the retrained set)" | New paragraph defining the *retrained set* and the *earlier pipeline*: the earlier ISIC2018 evaluation was largely on training images; why the retrained set is primary; earlier-pipeline results come only from a post hoc re-evaluation (S15). Supplement preamble aligned ("original" = earlier pipeline). | Brief 2.2 | Facts: S4; `revision/CLAIM_BOUNDARY_v2.md` §6; S15 |
| R2-12 | Methods, Study design and evidence levels + Table 2 (old Table 1, l. 198–225) | Three-stage table without status; separate paragraphs on protocol timing, the stage-2 display change and Experiment C | Table 2 with a Status column and rows for the ISIC2018 ten-run extension, the earlier-pipeline re-evaluation and Experiment C (14 of 70 runs, "Stopped, unused"). The procedure is compressed to one paragraph. Statuses per decision 4. The stage-2 display procedure (old l. 213–221) is preserved in S17 ("Post hoc balanced presentation"); Experiment C (old l. 222–225) in S15; OpenTimestamps (old l. 211) in S18. | Brief 2.4, 3.5; decision 4 | `EXC.completed`, `EXC.registered` (existing keys) |
| R2-13 | Methods, new Table 3 (labels) | SESOI labels and confirmation labels in prose (old l. 254–268) | One table with three vocabularies: SESOI readings (relevant gain/harm, size unclear, no relevant effect, undetermined); confirmation status (confirmed, supported, not established); the stage-3 pre-registered decision (success/failure/inconclusive). It also states how the registered wording "of unclear size" maps to the label "size unclear". | Brief 2.2; decision 2 | `expansion_statistics.classify`, `status`; pre-registration §7, §11 |
| R2-14 | Methods, Statistics | No Cochran's Q caveat | "With three datasets it has very low power, and a large p does not establish homogeneity" | Brief 2.2 | — |
| R2-15 | Methods, Statistics | — | Reporting precision stated (4 decimals for Boundary F1/DSC/IoU; 2 decimals in px for HD95/ASSD) | Brief 2.4 | — |
| R2-16 | Results structure (old l. 270–378) | Stage 1 (ablations, comparators, errors, run-to-run, cost), Stage 2, Stage 3 | RQ1 retraining variation (moved first) → RQ2 D over U → RQ3 in-network transfer (with probes and gate) → RQ4 external-teacher KD (Stage 2 then Stage 3) → RQ5 comparators and published context → RQ6 error structure and cost (deployment and D+KD-band training merged) | Brief 2.3 | — |
| R2-17 | Results RQ1 | Only ISIC2018 SDs | Stage-3 ACDC SDs added (0.0069, 0.0083) for scope; ten-run half-width at 4 decimals | Phase 1 correction (between-run scope); brief 2.4 | `CF.kd.sd`, `CF.dec.sd` ← `results.json`; `RS.X.halfwidth` ← `expansion_contrasts.csv` (median Boundary F1 half-width, 0.0215; was "0.02") |
| R2-18 | Results RQ2/RQ3/RQ5, Limitations: 4-decimal differences | 3-decimal keys `I.*`, `K.*`, `X.*`, `DEV.UA.*`, `CTX01.*` | 4-decimal `RS.I.*`, `RS.K.*`, `RS.X.*`, `RS.DEV.UA.*`, `RS.CTX.*`. Values are unchanged up to rounding, e.g. D − U pooled +0.016 [+0.005, +0.027] → +0.0159 [+0.0053, +0.0265]. | Brief 2.4 | Same source CSVs as the old keys (listed under "New keys"); `screen.json` `dice_delta` for CTX |
| R2-19 | Results RQ3, probes | "in seed-1 probes …" | "In gradient probes of seed 1 only …"; F_s readout labelled "in average precision"; section labelled post hoc | Phase 1 correction (probes) | `MECH.*`, `GR.*` (existing) |
| R2-20 | Results RQ4, Stage 3 | Pre-registered sentence plus a paraphrased SESOI explanation | Pre-registered wording quoted exactly once as "the pre-registered decision, in its registered wording". Mapped to the label "size unclear". | Decision 2 | `CF.*` ← `results.json` |
| R2-21 | Results RQ4, teacher | — | ACDC teacher, post hoc and descriptive: official 101–150 Boundary F1 0.8778, DSC 0.8146; internal 20 patients 0.8859, 0.8231. Different test sets, not interchangeable. No gap-closure calculation. | Brief 2.2; decision 7; teacher conditions | `RS.TCH.*` ← `revision/restructure_posthoc/teacher_metrics.json` (R2-A1) |
| R2-22 | Results RQ5 (old l. 307–322) | LB-UNet numbers and "run-level margin" wording; Swin-Unet/Swin UNETR sentence; GSL seeds 1–3 vs ten runs not reconciled | LB-UNet one sentence. Its numbers and the margin wording moved verbatim to S7 ("Adapted LB-UNet results"), where the margin is defined as the 95% half-width for a difference of two three-run means. Swin sentence moved to S7 ("Published results"); main keeps "the published Swin-Unet is the 2D architecture". Sentence added that the three-run means and the ten-run contrast are different evidence levels. Comparison-reporting paragraph (old l. 235–243) moved into the Table 5 caption and S7. | Brief 2.4, 3.5 | `C.*C4*` (existing) |
| R2-23 | Results RQ6, error locations | "only 19–20% … ACDC 70% … Synapse 47–49%" (original and retrained U-Net, mixed) | Retrained U only: ISIC2018 20%, ACDC 70% on the 20 internal test patients, Synapse 47%; scope stated | Phase 1 correction (70% scope) | `ERR.*.M0.near` (existing; `revision/tables/families/error_bands.csv`) |
| R2-24 | Results RQ6, cost | Gate cost and D+KD-band cost in separate places; cache time "not recorded" | Merged. Lists the arms that deploy the U-Net graph. Practical conclusion: replacing the gate with the identity restores the U-Net graph, changes at most 0.11% of labels and avoids the +8%/+15% latency. D+KD-band: teacher 0.243 h; cache 27 s (median of 3, re-measured post hoc; original not recorded); students 7.63 vs 7.52 h. | Brief 2.3, 3.5; additional decision (cache) | `GATE.labels.runmax`, `RESX.*`, `KDC.acdc.teacher.hours`, `CF.hours.*` (existing); `RS.KDC.cache.sec` ← `teacher_cache_timing.json` (R2-A2) |
| R2-25 | Discussion, first subsection (old l. 379–387) | Repeated +0.0178 [CI] and the SESOI statement | Descriptive answer by condition, with no numbers. The properties in which the settings differ are listed, and the responsible property is stated as untested. | Brief 2.1; decision 1 | — |
| R2-26 | Discussion, new "Why in-network transfer did not establish a gain: a post hoc interpretation" | Partly in "What the two designs differ in" (old l. 388–393) | BDS-Lite only: D_b shares the encoder; the probes are "consistent with" P meeting L_distill without changing F_s. Linked to the FitNets passage in Methods (old l. 133–136). Not established; probes designed post hoc. No comparison with D+KD-band (F15). | Decision 1 | — |
| R2-27 | Discussion, new "Candidate explanations for the D+KD-band gain" | — | Four untested explanations side by side: pretraining/capacity (12.7×), band restriction, general KD effect (secondary DSC of similar magnitude), and error distribution (70% from the stage-1 U on internal patients). The data do not distinguish them; D+KD-full and teacher variation are named. | Decision 1; F3/F4 kept (hypotheses, untested) | `RS.KDC.ratio.deployed`; `ERR.acdc.M0.near`, `ERR.isic2018.M0.near` |
| R2-28 | Discussion, Limitations (old l. 409–423) | One paragraph | Four groups: statistical power; protocol and generalizability; confounding in D+KD-band (no D+KD-full; pretraining, capacity and band changed together; one teacher); metrics and units of analysis | Brief 2.2 | `RS.DEV.UA.*`, `RS.CTX.*` |
| R2-29 | Conclusions (old l. 424–) | "Its magnitude relative to SESOI and its underlying mechanism remain unresolved" | No mechanism; "size unclear"; "which property … was not tested" | Decision 1, 2 | — |
| R2-30 | Arm naming, main text | "decoder-only", "the decoder", "decoder alone", "full BDS-Lite" | D, D+S, D+F+G, D+F+G+S (BDS-Lite), D+KD-band throughout. "decoder-only" is kept only as D's descriptive name and inside the verbatim pre-registered sentence. | Brief 2.3; decision 9 | — |
| R2-31 | Arm naming, supplement prose | Old names and M-codes | Preamble maps the new names to the M-codes and the older label vocabulary of the generated S5–S8 tables. Prose renamed in S3, S5, S7, S12, S13, S15, S17, S18. S15 "Evidence classes" rewritten to the decision-4 classification. | Decision 9, 4 | — |
| R2-32 | Swin-Unet citation | Main text cited `Cao2021SwinUnetArxiv` in Results | Main text cites `Cao2022SwinUnet` (ECCV 2022 Workshops, LNCS 13803, 2023) | Brief 2.4 | — |
| R2-33 | Celaya et al. (GSL) reference | arXiv 2302.03868 | **Unchanged.** No formally published version was found: the NSF PAR record (par.nsf.gov/biblio/10635864) lists "arxiv" with no journal or DOI, and the author's CV lists "arXiv preprint arXiv:2302.03868, in progress, 2024". | Brief 2.4 | — |
| R2-34 | `revision/author_decisions.md` (new) | — | Alternative wordings for the three `\authordecision` markers, which stay unchanged in `main.tex` | Brief constraint 4 | — |

### Open items deferred to Phase 3 (tables and figures)

- Generated table bodies still use old labels:
  - Table 4 (central): "Decoder vs U-Net", "Transfer, …"; the Reading column and the new ΔHD95
    header are not yet added.
  - Table 5A: "Decoder-only", "BDS-Lite (full)", seeds 1–3, boldface.
  - Supplement tables S5–S8 use the older label vocabulary, which the preamble maps.
  - Table 5B cites `Cao2021SwinUnetArxiv`, because its values were transcribed from arXiv
    v1. Moving it to the ECCV-W citation needs the workshop paper's values checked first.
- Figures 1 and 3 keep their old labels and order. The ISIC2018 qualitative figure is still in
  the main text (now Figure 3), pending its move to the supplement and the new Fig. 2.
- The text near Table 5B on aggregation is not yet quantified.
- Word count (`WC.*` keys) and response-letter anchors are not regenerated; this is done in
  Phase 4 with `build_revision.sh`.

## Phase 2 follow-up: non-monotonic S20 sensitivity results (2026-11-09)

The author noted that the pooled D+F+G − D intervals in S20 do not narrow with k:
- k = 0.5: [+0.0011, +0.0181];
- k = 1: [+0.0009, +0.0182], which is wider;
- k = 1.5: [−0.0010, +0.0195], close to the pre-specified [−0.0010, +0.0194].

**Finding: not a bug.** The pattern comes from the truncation rule combined with inverse-variance
reweighting. No result file, estimator or reading was changed, and `posthoc_shared_variance.py`
was not edited or rerun.

How this was checked:
- An independent script recomputed every adjusted SE from the stored components in
  `shared_component.csv`, using an explicit four-branch form of the rule. All 300 values
  (100 comparisons × k = 0.5, 1, 1.5) and every truncation flag matched, with 0 mismatches.
- The pooled rows recomputed to 1e-12.
- An explicit two-way ANOVA reproduced the ACDC shared term exactly (2.2100e-05).
- `make_restructure_assets.py` now stops unless its own recomputation of the pooled
  decomposition matches the CSV to 1e-12.

The mechanism, for ACDC (×10⁻⁵):
- between-run component 2.13, test-unit component 1.08, shared term 2.21;
- at k = 0.5 only the test-unit component is truncated, so V equals the between-run component;
- from k = 1 on, both components are truncated, so V = k · shared;
- V therefore grows with k, and at k = 1.5 it exceeds the pre-specified SE².

ACDC carries 83–89% of the pooled weight, so its SE drives the pooled SE.

Across all 100 comparisons:
- the adjusted SE is not monotone in k in 13 comparisons;
- at k = 1.5 it exceeds the pre-specified SE in 8.

**Response letter:** this item must be disclosed together with R2-S20, including the timing of
the truncation rule (R2-S20i).

| ID | Location | Original | Revised | Rationale | Numerical provenance |
|---|---|---|---|---|---|
| R2-S20h | Supplement S20, "Alternative estimator" + new paragraph "Truncation rule added after the reporting decision" | One sentence: "Truncation was adopted because some component estimates were negative" | States explicitly that the first computation had no truncation, that the reporting decision (the post hoc sensitivity analysis, readings kept on the pre-specified estimator; internally "Option B") was made on that computation, and that truncation was added only afterwards, with all results known. Gives the rationale: negative component estimates, and in 12 of 100 comparisons the untruncated adjusted variance fell below the shared term, which is its lower bound under the model. The untruncated results are retained: SE reductions 6–50% (vs 6–24%), and the pooled interval [+0.0041, +0.0161]. | Author follow-up, item 2 (truncation disclosure) | `RS.n.untrunc.below.shared`, `RS.red.untrunc.small.*`, `RS.tr.nosurf.untrunc.ci` ← `make_restructure_assets.py` ← `shared_component.csv` (`untruncated_se_without_shared`). The untruncated pooled interval equals the bbf2d9e output (delta 0.010099, SE 0.003068). |
| R2-S20i | Supplement Table S56 | 10 columns | New last column: the 95% CI at k = 1 without truncation, for every Boundary F1 comparison and the stage-3 secondary metrics. All other cells are unchanged (checked by diff). | Item 2 (retain untruncated results) | `low_untr`/`high_untr` ← `untruncated_intervals()` in `make_restructure_assets.py`: pre-specified t df; normal for pooled rows |
| R2-S20j | Supplement S20, new paragraph "Why the pooled interval is not monotone in k" + new Table S57 | — | The source of the non-monotonic pattern (as above) in three sentences. The new table gives each dataset's SE, weight, Δ and CI, plus the pooled row, at k = 0 (pre-specified), 0.5, 1, 1.5 and untruncated k = 1. The coverage table is renumbered S57 → S58. | Item 1 (non-monotonicity; truncation-plus-reweighting branch) | `RS.pd.*`, `RS.n.nonmono`, `RS.n.above.pre.k15`; `tab_s20_pooled.tex` ← `pooled_decomposition()` in `make_restructure_assets.py` |
| R2-S20k | Supplement S20, new paragraph "Conclusion" | — | Truncation occurred in 74 of 100 comparisons at k = 1, which indicates unstable variance-component estimates. This supports keeping the pre-specified estimator as the primary basis for inference. | Item 2 (S20 conclusion) | `RS.n.trunc.k1`, `RS.n.comparisons` (existing) |
| R2-S20l | Main, Results "Distillation and gate" (supersedes the R2-S20c sentence) | k = 1 interval only | Intervals at k = 0.5, 1 and 1.5, followed by "so whether the interval excludes zero depends on the assumed magnitude of the shared component". The sentence keeps the post hoc label and says the pre-specified reading is unchanged. | Item 2 (Pooled Transfer) | `RS.tr.nosurf.adj.k05.ci`, `RS.tr.nosurf.adj.ci`, `RS.tr.nosurf.adj.k15.ci` (existing keys) |

The Stage 3 main-text sentence (R2-S20d) is unchanged. Its intervals narrow monotonically with
k, and no component is truncated. `revision/CLAIM_BOUNDARY_v2_NOTE_20261109.md` (author-approved)
is unchanged: it quotes the k = 1 interval, which remains correct.

## Phase 2 follow-up 2: author review of the S20 changes (2026-11-09)

The author accepted the diagnosis (truncation floor plus inverse-variance reweighting, not a bug)
and asked for these edits before Phase 3. No result file, estimator or reading changed.

| ID | Location | Original | Revised | Rationale | Numerical provenance |
|---|---|---|---|---|---|
| R2-S20m | Supplement S20, "Why the pooled interval is not monotone in k" | The mechanism was described per case | Mechanism stated in one sentence: when both components are truncated, the adjusted variance is floored at k·shared, so the floor rises with k. This is why k = 1 is wider than k = 0.5 and why k = 1.5 can exceed the pre-specified SE. ACDC worked example (×10⁻⁵): between-run 2.13, test-unit 1.08, shared 2.21. Adjusted variance: 2.13 at k = 0.5, floor 2.21 at k = 1, floor 3.31 at k = 1.5, against pre-specified SE² 3.21. The 13/8 counts moved to the Conclusion. | Author follow-up, item 1 | `RS.pd.acdc.{run,unit,shared,floor.k15,se2.k0}` ← `pooled_decomposition()` in `make_restructure_assets.py` ← `shared_component.csv` |
| R2-S20n | Supplement S20, "Conclusion" | 74/100 truncation count only | Three counts over 100 comparisons together: truncation 74 at k = 1; adjusted SE non-monotone in k in 13; above the pre-specified SE at k = 1.5 in 8. Together they are evidence of unstable variance-component estimates and support keeping the pre-specified estimator as primary. | Item 2 | `RS.n.trunc.k1`, `RS.n.nonmono`, `RS.n.above.pre.k15` (existing) |
| R2-S20o | Supplement S20, "Truncation rule added after the reporting decision"; Table S56 and S57 captions | Untruncated results listed as retained | Framed as retained for disclosure, not as a preferred or more precise estimate; the narrower intervals are a consequence of negative, unstable component estimates. The pooled [+0.0041, +0.0161] is now shown next to the truncated and pre-specified intervals. It is narrow mainly because the untruncated ACDC variance (1.00×10⁻⁵) fell below its own shared term (2.21×10⁻⁵), which raised ACDC's weight to 94%. | Item 3 | `RS.pd.acdc.var.untr`, `RS.pd.acdc.w.untr`, `RS.tr.nosurf.untrunc.ci`; pre-specified `A1.nosurf.boundary_f1.ci` (existing, `numbers_kd.tex`) |
| R2-S20p | Main, Results Stage 3 (supersedes R2-S20d) | k = 1 lower bound only (`RS.s3.adj.low`) | Same wording as the pooled-transfer sentence: intervals at k = 0.5, 1 and 1.5 ([+0.0101, +0.0255], [+0.0104, +0.0253], [+0.0107, +0.0250]), post hoc, and the pre-registered decision and its reading unchanged. F11 still holds: no comparison with the SESOI. | Item 4 | `RS.s3.adj.k05.ci`, `RS.s3.adj.ci`, `RS.s3.adj.k15.ci` (existing keys) |

S20 runs from supplement p. 49 to p. 52. `RS.s3.adj.low` is no longer used in the manuscript but
stays in the key file.

## Phase 3: figures and tables, first part (2026-11-09)

Author instructions: tasks A, B and C, in that order.
- Every main figure and table uses the pre-specified estimator only.
- No result file or estimator is modified.
- Fig. 1 is not replaced until the author approves.

| ID | Location | Original | Revised | Rationale | Numerical provenance |
|---|---|---|---|---|---|
| R3-A | New files only: `analysis/revision/figure_architecture_v3.py`; `revision/figures/drafts/{architecture_v3_draft.pdf,architecture_v3_draft.png,fig1_side_by_side.png}` | Fig. 1 = raster `revision/figures/architecture.png` (no editable source; audit §1.2) | **Draft only, not in the manuscript.** A vector redraw that keeps the original layout and sampled palette. Corrections against the Methods text and the code: <br>• `L_bnd` → `L_boundary`, `L_sdf-surface` → `L_sdf`, `λ_distill` → `λ_d`; no `λ_seg` <br>• boundary head `c_b → 1`, not `boundary_classes` (all configs have `boundary_classes: 1`) <br>• "1×1 Proj" → `P` <br>• "Boundary Prediction P_b" → `σ(z_b)`, removing the clash with `P` <br>• output convolution `W_o` and logits `z_s`, as in Eq. 1 <br>• D/F/G/S component tags, as in Table 1 <br>• S drawn in panel (c) <br>• new panel (d) for D+KD-band (Eq. 6: frozen pretrained teacher, cached `p^t`, band from the teacher's argmax, `λ_kd = 1`) <br>• objectives box with all three objectives and the deployment rule | Author Phase 3 task A; brief 3.1; audit D3 (the author's D3 choice is not recorded in the repo; a vector redraw was drafted, which also covers option (c)'s D+KD-band panel) | No numbers from result files. Constants from `main.tex` Eqs. 1–6 and `src/bds_lite/models/bds_lite.py:14–22` (`c_b = max(c//2, 8)`, `boundary_classes = 1`) |
| R3-B | Table 5B, Swin-Unet rows (`Cao2021SwinUnetArxiv`) | arXiv v1 values: ACDC DSC 0.9000 (p. 9 Table 2); Synapse DSC 0.7913, HD 21.55 (p. 8 Table 1); source `revision/lit_numbers.csv`, PDF `revision/sources/literature/2105.05537v1.pdf` | **Unchanged; task stopped (blocked).** The ECCV 2022 Workshops chapter (LNCS 13803, pp. 205–218, doi:10.1007/978-3-031-25066-8_9) could not be accessed. `curl` got a Springer JavaScript bot challenge for both the PDF and the chapter page. A web fetch of the PDF URL, the chapter URL and the DOI redirected each time to `idp.springer.com/authorize` (login). No local copy exists; only arXiv v1 is on disk. Per the author's instruction, nothing was decided from memory or from secondary sources. | Author Phase 3 task B | — |
| R3-C | Main Figs. 1–3, Tables 1–6 | — | **No change needed.** All existing main figures and tables already use the pre-specified estimator only (checks below). The Fig. 3(b) forest plot and the Table 4 Reading column do not exist yet; they are deferred Phase 3 items (Phase 2 open items above) and must follow the same rule when built. | Author Phase 3 task C | Checks listed below |

**R3-C checks (2026-11-09).**
1. Grep pattern `restructure_posthoc|shared_component|posthoc_shared_variance|untrunc|se_k0?5|se_k1|se_k15|low_k|high_k|_untr|shared_variance|adj\.|\.adj|RS\.(tr|s3|dec)\.|RS\.pd\.|tab_s20` (plus `RS\.` for the generated tables). It was run over:
   - the generators of every main figure and table: `make_distillation_assets.py` (Table 4 `tab_central_main`, Table 6 `tab_resource_main`, Fig. 2 `kd_confirmation`), `make_comparison_table.py` (Table 5 `tab_lit_main`), `figure_qualitative_retrained.py` and `export_submission_figures.py` (Fig. 3), and `figure_architecture_v3.py` (Fig. 1 draft);
   - the three generated main tables;
   - every `table`/`figure` environment in `main.tex` (Tables 1–3 are inline).

   Result: no match.
2. S20 keys in `main.tex` appear only in two Results paragraphs: l. 264, pooled transfer, and l. 275, stage 3. Both are post hoc sentences, outside any figure or table.
3. The plotted values were checked numerically:
   - Fig. 2's stage-3 interval (`results.json` `primary.contrast`: 0.009829, 0.025812) equals the pre-specified `low`/`high` in `shared_component.csv` and none of its adjusted columns.
   - All Fig. 2 stage-2 rows (`kd_posthoc_balanced/contrasts.csv`) equal the pre-specified intervals.
   - Table 4's pooled transfer row is the pre-specified [−0.0010, +0.0194].
