# Follow-up controls for D+KD-band: summary (2026-11-10)

Branch `revision-followup`; local commits only, nothing pushed. Registration:
`revision/followup_registration.md` (commit `f296023`), Amendment 1 (`f289689`).
Progress: `revision/followup_progress.md`. Protocol log: `revision/followup/PROTOCOL_LOG.md`.

**Editorial status, 2026-11-10 13:27 UTC.** E1/E2 results below are unchanged. E3 resumed and remains ongoing; no E3 test metric is inspected or used in this revision. The evidence-freeze draft remains provisional. The current correction record is `revision/SCIENTIFIC_AUDIT_CORRECTIONS_20261110.md`.

## 1. Experimental results (one page)

**Status.** All follow-up results are secondary and exploratory. They reuse the 50 official ACDC
test patients (101–150), whose Stage 3 results had already been read. They were registered
before training. The registration has no externally verified timestamp (Amendment 1).

**Runs.**

| Group | Runs completed | Notes |
|---|---|---|
| E1, D+KD-full | 17/17 | — |
| E2 teacher and cache | done | — |
| E2, D+KD-band with the compact teacher | 17/17 | — |
| E1 + E2 queue overall | — | No failed run and no non-finite loss. 431 thermal-guard pauses, each resumed under the frozen hash checks. Integrity check PASSED. Test patients evaluated once (35 runs, 2026-11-10 11:10–11:19Z). |
| E3, augmentation (optional) | Incomplete | Resumed after the earlier halt; ongoing at the editorial cutoff. No E3 test result is included. |

**Primary endpoint: test Boundary F1.** Stage 3 estimator; Holm adjustment over C1–C4.

| Comparison | Δ [95% CI] | Holm p | Category | Table 3 reading |
|---|---|---|---|---|
| Stage 3: D+KD-band − D (context) | +0.0178 [+0.0098, +0.0258] | — | — | gain, size unclear |
| C2: D+KD-full − D | +0.0106 [+0.0034, +0.0177] | 0.020 | P | gain, size unclear |
| C3: D+KD-band(compact teacher) − D | +0.0118 [+0.0037, +0.0200] | 0.020 | P | gain, size unclear |
| C1: D+KD-band − D+KD-full | +0.0073 [−0.0005, +0.0150] | 0.095 | U | undetermined |
| C4: D+KD-band − D+KD-band(compact) | +0.0060 [+0.0001, +0.0119] | 0.095 | U | gain, size unclear (unadjusted) |

**Registered mapping.**
- Band case **B3**: full-image KD also raised Boundary F1. Whether the band adds is unresolved, so
  the Stage 3 gain cannot be attributed to boundary-specific distillation.
- Teacher case **T2**: the specified same-capacity teacher without pretraining also raised Boundary F1. Its learned information is not proved redundant or equivalent to the student's. The difference between teacher configurations is unresolved.
- The registered category mapping is preserved. Its original broad narrative does not establish absence, equivalence, or a mechanism. C1 and C4 remain unresolved, with positive point estimates about 41% and 34% of the Stage 3 Δ; these ratios are descriptive, not a causal decomposition.

**Secondary results (descriptive).**
- DSC moved with Boundary F1 in C2 and C3 (+0.0100 and +0.0103).
- D+KD-band had smaller HD95 and ASSD than D+KD-full: C1 +1.36 px [+0.34, +2.38] and
  +0.37 px [+0.10, +0.64], Holm within metric p = 0.043 and 0.034.
- D+KD-full showed no relevant HD95 or ASSD change against D.

**The compact teacher (descriptive).** On the test patients it scored Boundary F1 0.7735. That is
below the D mean (0.7797) and below its own students (0.7915). The external teacher scored 0.8778.

**Post hoc analyses (Supplementary S22).**
- **Reference estimators.** A crossed random-effects model (method of moments) and a hierarchical
  bootstrap give intervals close to the pre-specified ones. The mixed-model interval of C1 is
  +0.0002 to +0.0144. Neither changes any registered reading.
- **Error location.** D's errors within 2 px of the contour were 55% (RV), 80% (myocardium) and
  67% (LV). The gain did not follow this share, by structure or across patients: Spearman
  ρ = 0.00, −0.16 and −0.08 over patient–structure pairs. Gains were larger where D scored lower.
- **Stage 1 vs Stage 3 D mean (0.7993 vs 0.7797).** The larger observed shift is between cohorts and preprocessing pipelines: the same Stage 3
  runs score −0.0200 [−0.0231, −0.0169] lower on the official than on the internal patients. The seed-set/trainer point estimates are small (−0.0021 and −0.0011 on fixed patients), with uncertainty; aggregation also changes the displayed means. Different patients and reconstructed preprocessing cannot be separated.

**Answer to the title question supported by the data.** The proposed central argument was that
the benefit "depends on where errors occur and whether the teacher has an informational
advantage". It is **not** supported:
- the compact teacher had no pretraining or capacity advantage, but other teacher properties were not isolated;
- error location did not track the gain within ACDC.

In this compact, unaugmented ACDC setting, distilling a separately trained network's class
probabilities gave positive contrasts against D. The incremental band and teacher-configuration effects remain unresolved; the data do not establish their absence or equivalence.
In-network transfer (BDS-Lite's F + G) did not establish a gain.

## 2. Manuscript changes and why

**Main text (`revision/manuscript/main.tex`).**

| Section | Change | Why |
|---|---|---|
| Abstract | Rewritten around the controls; B3/T2 reading. | Registered mapping. |
| Introduction | Third gap (attribution) and a third contribution. Raj et al. removed. | Brief; the reference does not support the claim it was used for. |
| Methods | Table 1 limited to the core arms. "BDS-Lite" now names only the full recipe. New "Why D, not GSL or Boundary Loss" paragraph. Surface-loss definition moved to S2. New KD-full and compact-teacher controls. Follow-up study-design row and paragraph, including the E3 status. Experiment C stop reason from the author records (compute decision, no test metric read). Statistics: Holm family, cross-dataset pooling scale and ACDC weight (76–84%), post hoc reference estimators. | Brief. |
| Data | Two-sentence correction note on the earlier public version; Bonn present address. | AUTHOR DECISION markers. |
| Results | Core comparisons only. Two sensitivity explanations merged into one sentence. New follow-up section with Table 5 and Fig. 3 (Fig. 2 style). New Table 6: evidence classification of all other arms, including "S lowered Boundary F1 without reducing HD95". Post hoc error location. Cost shortened. | Brief. |
| Discussion | Answer to the title question; mechanism (post hoc, Born-Again Networks); in-network transfer without additional external data (side by side, no head-to-head; learned-information redundancy unestablished); error location; Stage 1 vs Stage 3 D mean and the preprocessing risk; practical implications; limitations. | Brief. |
| Conclusions | Rewritten. | Follow the new reading. |

**Supplement.**
- New sections: S21 (follow-up controls, all metrics and per structure) and S22 (post hoc
  analyses).
- Moved in: the qualitative figure (S12), the matched-protocol table (S7) and the surface-loss
  definition (S2).
- Label `tab:arms` renamed to `tab:s-arms`.

**Letters.**
- Response to reviewers (`revision/response_to_reviewers.tex`):
  - follow-up added to the summary and to E-1, R1-0, R1-4 and R3-R14;
  - Raj et al. dropped (S-2, R3-I7);
  - moved floats re-pointed;
  - prose counts are regenerated from the current manuscript source and printed in both letters; the pre-audit 6,943/+21% count is historical.
- `letter_locations.py` anchors rewritten; all 118 locations resolve.
- Editor letter updated.

**Verification.** The numerical values above are retained. Current independent numerical receipts, document-build checks, source-preservation checks and remaining gaps are recorded in `revision/SCIENTIFIC_AUDIT_CORRECTIONS_20261110.md` and `results/final_manuscript_audit_v2/`. Earlier build/test counts describe the earlier package, not the revised documents.

**Claim-boundary status.** V2 and the author-authorized completed-evidence v3 record are preserved unchanged. The current author instruction authorizes correcting obsolete factual statements and unsupported interpretations in the documents; it does not authorize a new evidence freeze. Instruction-discovery reconciliation in AGENTS.md remains author-owned.

## 3. Outstanding [TODO] items

1. `main.tex` affiliation: `[TODO: department]`, University of Bonn present address. I attached it
   to the first author; confirm which author it belongs to.
2. `main.tex` Data and code availability: `[TODO: Zenodo DOI of the revision release]`.
3. Response letter and editor letter: AUTHOR DECISION, "confirm that all co-authors have read and
   approved".
4. The author owns any future change to `AGENTS.md` and the historical claim-boundary records. The obsolete v2 full-image-control rationale is not used to deny the completed E1 evidence. No boundary source is edited in this correction.
5. E3 has already resumed under the author's earlier instruction. Leave its processes, stop markers, queue, sources and outputs untouched. Its outcomes remain outside this editorial revision. The evidence-freeze draft is provisional until all experiments finish and the author authorizes further action.
6. The pending `.ots` files of `affab9e`: keep as history or delete. They are described as never
   verified.
7. Still open from earlier phases, unchanged:
   - the author's uncommitted records (`revision/DEVIATIONS.md`, `revision/queue_plan.yaml`,
     `numbers.tex`, two figure PDFs, the Experiment C scope amendment);
   - the public-repository image/history decision (`revision/RELEASE_GUIDE.md`);
   - the Fig. 1 draft approval;
   - the ECCV-W chapter for Table 5B (Phase 19).

## 4. Remaining risks

- **Test-set reuse.** The controls and every post hoc analysis use patients whose Stage 3 results
  were read. Reviewers may discount them, and the manuscript labels them exploratory.
- **Power for the attribution contrasts.** C1 and C4 cannot rule out band or teacher
  contributions of about 0.012–0.015, which is up to 80% of the Stage 3 Δ. "Not shown to depend
  on" is the most the data allow.
- **Design limits.**
  - Pretraining and capacity are confounded.
  - Each teacher was trained once.
  - λ_kd was not retuned for the full-image loss, whose effective weight differs.
- **Unaugmented training.** The soft-label benefit may shrink under augmentation, and E3 did not
  finish.
- **Cohort and reconstructed preprocessing.** The roughly −0.02 Boundary F1 shift compares different patients and preprocessing pipelines. It does not measure a preprocessing effect. On validation data the recorded maximum run-level reconstruction effect was 0.000112 Boundary F1 and 0.0000522 DSC; official-patient impact and interaction with distillation remain unknown.
- **Mechanism.** Positive control contrasts do not establish a general KD mechanism. Teacher configurations also change calibration, realization and predicted band geometry, and the full-image loss has a different effective scale. Attribution remains unresolved.
- **Length.** The regenerated word counts in the editor/reviewer letters describe the current source; reviewer 2's concern about verbosity remains relevant.
- **Thermal load.** The laptop GPU ran at 84–85 °C, causing about 430 guarded pauses. The
  resumes are hash-checked, but bitwise equality with uninterrupted training was not shown for
  these runs.
- **Timestamp.** The follow-up has no external timestamp; its order rests on local,
  self-reported git dates and logs.
