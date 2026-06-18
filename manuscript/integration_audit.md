# Phase 17 Integration Audit

This audit records exactly which Phase 16 artifacts are integrated into the
Phase 17 submission-candidate manuscript package, which earlier wording is now
obsolete, and what still requires human input before a real submission. It is
the control document for `manuscript/BDS_Lite_submission_candidate_v1.md` and
the `tables/` and `figures/` packages.

Source of truth for this phase:

- `results/submission_blocker_resolution_v1/FINAL_BLOCKER_RESOLUTION_REPORT.md`
- `results/submission_blocker_resolution_v1/manuscript/BDS_Lite_number_corrected_draft.md`
- `results/submission_blocker_resolution_v1/corrected_number_audit.csv`
- `results/submission_blocker_resolution_v1/matched_gsl_execution_report.md`
- `results/submission_blocker_resolution_v1/matched_gsl_main_comparison.csv`
- `results/submission_blocker_resolution_v1/training_cost_final_decision.md`
- `results/submission_blocker_resolution_v1/efficiency_final_wording.md`
- `results/submission_blocker_resolution_v1/final_figure_readiness_review.md`
- `results/submission_blocker_resolution_v1/figure2_matched_caption.md`
- `results/submission_blocker_resolution_v1/figures/figure2_matched_main_results.{pdf,png}`
- `results/submission_blocker_resolution_v1/profiling/training_cost_combined_summary.csv`

## 1. Which Phase 16 corrections must be integrated?

The Phase 16 number audit (`corrected_number_audit.csv`, 303 data rows) closed
all 11 Phase 15 mismatches with source-backed adjudication. The corrected copy
`BDS_Lite_number_corrected_draft.md` already carries every corrected value, so
Phase 17 starts from that copy rather than re-deriving numbers. The integrated
corrected values that anchor the candidate manuscript and tables are:

- ISIC2018 validation means (recomputed source
  `results/final_manuscript_audit_v2/recomputed_main_test_means.csv`):
  U-Net DSC 0.8830, Boundary F1 0.5520, HD95 18.5128, ASSD 5.9192;
  BDS-Lite DSC 0.8828, Boundary F1 0.5597, HD95 17.4435, ASSD 5.6753;
  IoU 0.8091 to 0.8112.
- ACDC test means: U-Net DSC 0.7893, HD95 6.8762, ASSD 2.2080, BF1 0.8313;
  BDS-Lite DSC 0.7847, HD95 7.6230, ASSD 2.4874, BF1 0.8267.
- Synapse test means: U-Net DSC 0.8544, HD95 10.1985, ASSD 3.7515, BF1 0.8608;
  BDS-Lite DSC 0.8573, HD95 10.3413, ASSD 3.8280, BF1 0.8632.
- Resource profile: U-Net 1,927,042 parameters and 15.970 GFLOPs;
  BDS-Lite 2,198,003 training-graph parameters; retained deploy graph
  1,928,114 parameters and 16.079 GFLOPs (1,072 parameters above U-Net,
  approximately 0.056%; about 0.68% more FLOPs); 269,889-parameter boundary
  branch removed.
- Cluster-aware ISIC2018: mean oriented HD95 delta 1.0409 pixels (95% bootstrap
  0.2476 to 1.8092); mean Boundary F1 delta 0.0077 (0.0006 to 0.0146); all
  Holm-corrected seed-level p-values 1.0.

These are carried verbatim into `tables/table2_matched_main_results.*` and the
Results/Discussion of the candidate manuscript.

## 2. Which Phase 15/14 wording is now obsolete?

| Obsolete wording | Replaced by | Reason |
|---|---|---|
| "separate P1B/P2 comparison family" / "separate comparison family" for GSL | Matched three-seed GSL rows under the locked protocol | The Phase 16 matched rerun used the locked seeds, splits, input size, batch sizes, AMP policy, 150-epoch schedule, and max-val-DSC checkpoint rule. |
| "different checkpoint lineage" / "legacy split-file fingerprint" caveats for GSL | Removed for the main comparison; GSL is now in the same matched family | `matched_gsl_execution_report.md` confirms a matched rerun, not the legacy lineage. |
| "GSL was competitive or stronger in its available comparison family" | "GSL remains a strong geometry-aware comparator; matched three-seed means are mixed across datasets in both directions" | The matched rerun shows GSL **lower** on ISIC2018 means, near the comparators on ACDC, and higher on Synapse DSC/Boundary F1 with mixed distance. |
| Phase 15 Figure 2 ("separated families", Panels A/B) and its caption | Phase 16 matched Figure 2 and its conservative caption | `final_figure_readiness_review.md` supersedes Figure 2 for journal use. |
| Any pooled/ranked GSL-vs-BDS-Lite framing | Descriptive, non-superiority framing | No significance is claimed from three seed means. |

## 3. Which matched GSL results replace separate-family wording?

The matched main comparison (`matched_gsl_main_comparison.csv`,
`matched_gsl_execution_report.md`) replaces all separate-family GSL text and
tables. Matched three-seed means (mean ± seed SD):

| Dataset/split | Method | DSC | HD95 | ASSD | Boundary F1 |
|---|---|---:|---:|---:|---:|
| ISIC2018/val | U-Net | 0.8830 | 18.5128 | 5.9192 | 0.5520 |
| ISIC2018/val | BDS-Lite | 0.8828 | 17.4435 | 5.6753 | 0.5597 |
| ISIC2018/val | U-Net+GSL | 0.8662 | 19.7954 | 6.6595 | 0.5399 |
| ACDC/test | U-Net | 0.7893 | 6.8762 | 2.2080 | 0.8313 |
| ACDC/test | BDS-Lite | 0.7847 | 7.6230 | 2.4874 | 0.8267 |
| ACDC/test | U-Net+GSL | 0.7856 | 7.0143 | 2.0902 | 0.8277 |
| Synapse/test | U-Net | 0.8544 | 10.1985 | 3.7515 | 0.8608 |
| Synapse/test | BDS-Lite | 0.8573 | 10.3413 | 3.8280 | 0.8632 |
| Synapse/test | U-Net+GSL | 0.8640 | 10.1285 | 3.8567 | 0.8701 |

Direction summary (descriptive only, no significance): GSL has lower ISIC2018
DSC and Boundary F1 and higher ISIC2018 HD95/ASSD than both comparators; on
ACDC it lies between/near the comparators; on Synapse it has higher DSC and
Boundary F1 and lower HD95, but its ASSD is higher than both. The matched
comparison therefore does **not** support a claim in either direction and is
integrated as a mixed result.

## 4. Which efficiency wording is allowed?

From `efficiency_final_wording.md` and `training_cost_final_decision.md`:

Allowed: the auxiliary boundary decoder is omitted at inference; the retained
graph is close to the U-Net backbone in the recorded parameter and FLOP
profile; controlled no-update profiling on one RTX 5060 Laptop GPU records
method-specific training-step time and peak memory; these local measurements do
not establish end-to-end time-to-convergence or cross-hardware portability. For
GSL, the matched rerun and profile used an exact CPU Euclidean distance
transform to avoid the quadratic-memory pairwise fallback, preserving the
evaluated loss values while making the protocol feasible on the available GPU.

Local profiling numbers integrated (batch 8, 224x224, AMP, no optimizer step;
`profiling/training_cost_combined_summary.csv`): U-Net 39.62 ms/step, 600.29 MB
peak; BDS-Lite 57.17 ms/step, 873.57 MB peak; U-Net+GSL 52.32 ms/step,
600.29 MB peak. Inference latency on the same GPU: U-Net 1.93 ms, BDS-Lite
2.40 ms, U-Net+GSL 1.97 ms.

Forbidden (must not appear anywhere): training-efficient, deployment-ready,
faster, hardware-friendly, clinical deployment, lower total training cost, more
efficient than GSL, time-to-convergence advantage, hardware-independent
efficiency.

## 5. Which figures are journal-ready versus human-polish-needed?

| Figure | Journal status | Action in Phase 17 |
|---|---|---|
| Figure 1 (architecture / train-inference decoupling) | usable after human layout check | referenced from `submission_critical_evidence_v1/figures/figure1_architecture_training_inference_decoupling.{pdf,png}` |
| Figure 2 (matched main results) | **Phase 16 replacement is the journal version** | copied into `figures/figure2_matched_main_results.{pdf,png}` with the conservative matched caption |
| Figure 3 (per-case delta distributions) | usable after human layout check | referenced from existing source |
| Figure 4 (boundary–distance relationships) | usable after human layout check | referenced from existing source |
| Figure 5 (subgroup/organ heatmap) | supplementary preferred; human polish remains | referenced as supplementary |
| Figure 6 (failure cases) | usable as illustrative material after human panel review | referenced from existing source |

No figure is regenerated in Phase 17 (no experiments, no rerender). Phase 15
Figure 2 (separated families) is explicitly **not** used as the journal main
figure.

## 6. What still requires human input before real submission?

These are editorial/production tasks, not missing experiments:

1. Bibliography verification of all 26 references against DOI/publisher records,
   including the year/status of the GSL preprint (ref 13).
2. Author order, affiliations, corresponding author, CRediT roles, funding
   identifier, conflicts of interest, ethics statement, dataset licenses, data
   and code availability persistent identifier.
3. Target-journal template formatting, word/figure/reference limits, embedded
   fonts, accessibility, figure column dimensions, and final caption numbering.
4. All-author approval of final claims, negative results, cover letter, and
   highlights; plagiarism/duplicate-publication checks.
5. JCR category/quartile verification through an institutional source (do not
   assert a quartile in the manuscript).
6. DOCX/PDF proofread of the converted file (the Phase 17 DOCX is a
   convenience conversion, not a journal-template document).

## 7. Fixed positioning preserved

The candidate manuscript preserves the fixed research positioning: boundary
distillation has conditional, task-dependent value; metric families
(overlap/boundary vs. surface distance) can disagree; failure modes are
characterized; and the auxiliary boundary path is decoupled from inference. No
universal-superiority, clinical-deployment, or BDS-Lite-beats-GSL claim is
made. All mixed and negative results (ACDC null, Synapse distance-negative,
GSL mixed) are retained.
