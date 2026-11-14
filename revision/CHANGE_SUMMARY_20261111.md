# Change summary, manuscript rewrite of 2026-11-11

Scope: prose, layout, tables and figures from existing evidence. No training run, no new
statistical analysis, no changed number. Claim boundary: `revision/CLAIM_BOUNDARY_v4.md`.
All numbers in `main.tex` are generated keys or generated tables; `verify_final_assets.py`
reports 200 checks, 0 mismatches.

## Numbering changed

| Item | Before | Now |
|---|---|---|
| Boundary-aware methods (new, condensed from S1) | — | Table 1 |
| Studies, runs and test data | Table 1 | Table 2 |
| Arms | Table 2 | Table 3 |
| Matched-protocol scores | Table 3 | Table 4 |
| Boundary F1 contrasts (panels A, B) | Table 4 | Table 5 |
| Official ACDC contrasts | Table 5 | Table 6 |
| Published results (new, from S7 panel B) | — | Table 7 |
| Inference cost (new, from `resource_profile.csv`) | — | Table 8 |
| ISIC2018 contour examples (new) | — | Figure 4 |
| Equations | gate (1) … SE (5) | DSC relation (1), gate (2) … SE (6) |

The author brief's "Table 1" (studies), "Table 4B" and "Table 5" are now Tables 2, 5B and 6.
The supplement's references to main-text tables were updated to match.

## By section

**Abstract.** Rewritten: motivation, design, seven estimates with intervals, the run-to-run SD,
a conditional answer to the title question, one recommendation. BDS-Lite spelled out. No
evidence-level qualifiers. 295 words.

**Introduction.** Five labelled paragraphs (Motivation, Why a compact U-Net, Background,
Research gaps, Contributions). Clinical motivation is cited and followed by the statement that
no clinical outcome is measured. BDS-Lite is introduced directly by its components. The
contributions list design elements and no longer repeats the results.

**Related work.** Same four themes; new Table 1 (method, type, kept at inference, limitation).

**Materials & Methods.**
- Study design explains confirmatory and exploratory once; Table 2 carries the labels.
- New subsection "Framework: contour error, overlap, and the routes of a boundary signal":
  Eq. 1 (moved from the Introduction) and four routes (the three in-network routes, moved from
  the end of the in-network subsection, plus the external-teacher route).
- Arms: Table 3 lists the inference network of D and the distilled students as "U-Net" with a
  note that this holds after pruning and that latency was not measured separately (C11);
  BDS-Lite is no longer described as training-only. Reference methods have a stated rationale.
- Arm renamed D+KD-band_c -> D+KD-band_cmp (display name only; result-file identifier
  `kd_band_peer` unchanged; mapping stated in the supplement preamble). Base width written as
  32 in Arms; `c` remains the channel count in the equations.
- Data: source URLs and DOI, splits and preprocessing in labelled paragraphs.
- Training and evaluation: Python, cuDNN, NumPy versions from the run records; rationale and
  limits of Boundary F1 as primary endpoint.
- Statistics: SESOI described as a preset analysis threshold, not a clinical threshold.

**Results.** Reorganised into: Run-to-run variation; The boundary decoder; In-network
distillation; External-teacher distillation; Is the gain specific to the boundary band or to
the teacher?; Distillation under augmentation; Published results and inference cost; Contour
examples. Negative phrasings replaced by estimates and intervals. Table 5B gains fresh-seed
intervals (`expansion_contrasts.csv`, columns `fresh_low`, `fresh_high`). Table 6 group
headings and Fig. 3(a) subtitles use study numbers instead of evidence labels. The
proportional-transfer paragraph was removed from Results and Discussion (it remains in S23).
Retained v4 qualifiers: "gain of unclear size" (C04), controls "not independent confirmations"
(C05), Holm-adjusted readings (C06), "a benefit, a harm and equivalence are all unestablished"
and no interaction tested (C13).

**Discussion.** Opens with the answer. New subsection on why the complete BDS-Lite recipe
scored below the decoder alone, from S10, S13, S15 and S19 (S22 holds nothing on this point;
see the to-confirm list), stated as diagnostic observations. Regularization is given as an
interpretation that was not tested. Implications ordered for practitioners and for study design.

**Limitations.** Five items reduced to three (scope; precision of the in-network results;
replication cohort and controls), each tied to a conclusion. nnU-Net, 3D and tumour datasets
are one scope sentence; the detail is in the response letter.

**Conclusions.** Model type and purpose, findings with numbers, limits, future work.

**References.** Venue notes and three arXiv `v1` suffixes removed.

**Supplement.** Arm label, main-text cross-references, stale statements (resource table,
qualitative figure, availability pointer) updated. No analysis or number changed.

**Response letter.** `revision/response.md`, rendered from `response.template.md`; numbers come
from the generated keys, line numbers from the compiled `main.pdf`, and quoted table values are
checked against their source files. `response_to_reviewers.tex` is superseded and not built.
