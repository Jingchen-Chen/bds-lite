# Reviewer-coverage matrix (updated for the manuscript of 2026-11-07)

Source of each comment: the decision letter (`~/Downloads/reviewer.md`). All 40 quoted fragments in
`revision/response_letter_v2.tex` were checked verbatim against it, with 0 mismatches. Reviewers cited
no line numbers. "Evidence" states whether the request is fully met, not whether the result is favorable.
**The response letter has not been updated for the 2026-11-07 manuscript:** 27 of its 86 named locations point to old headings and several statements are out of date (see `HUMAN_DECISIONS.md`).

| ID | Request | Justified? | Where addressed | Evidence |
|---|---|---|---|---|
| E-1 | More comparisons, ablations, statistics, qualitative results | Editorially required | All Results subsections; Statistical analysis | Sufficient, except nnU-Net/Swin UNETR not run (stated) |
| E-2 | Clear method and theoretical foundations | Yes | BDS-Lite; How boundary information can reach the prediction; Supp S2 | Sufficient; theory kept deliberately small |
| E-3 | Why compact U-Nets only | Yes | Introduction para 5; Limitations | Sufficient as rationale; no result beyond one backbone |
| E-4 | Clinical relevance and impact | Yes | Introduction para 1; Clinical relevance; Conclusions | **Partial.** EF, myocardial mass, mm not computed (geometry unverified); no impact claim made |
| R1-0 | General assessment | n/a | Letter only | n/a |
| R1-1 | Literature with contributions and limitations | Yes | Related work; Supp S1 | Sufficient |
| R1-2 | Theoretical framework | Yes | How boundary information can reach the prediction (Eq. 5); Why the transfer added so little; Supp S10 | Sufficient; labelled post hoc, part of it failed and is reported |
| R1-3 | Compare with state of the art | Yes | Other boundary methods and published results; Table 8; Supp S7 | Sufficient for 4 matched comparators + verified literature |
| R1-4 | Ablation (main weakness) | Yes | Does distillation add to the decoder?; The full recipe and its surface loss; Tables 4-6; Fig. 4; Supp S3, S6, S13 | Sufficient for main effects. **Interaction estimated but unresolved** at three seeds per cell (stated) |
| R2-1 | Justify compact U-Net | Yes | Introduction para 5 | Sufficient |
| R2-2 | Clinical relevance of boundary errors | Yes | Introduction para 1; Clinical relevance | Partial, as E-4 |
| R2-3 | Verbose prose | Yes | Whole manuscript | About 7,100 -> 8,100 prose words: the central argument, mechanism and ten-seed results added length. **Longer than the original**; secondary detail moved to the supplement |
| R2-4 | Define stopgrad | Yes | BDS-Lite | Sufficient; verified against `losses.py` (`.detach()`) |
| R2-5 | Explain alpha, justify 0.25 | Yes | BDS-Lite; Does distillation add to the decoder?; Supp S11 | Sufficient; 0.25 presented as a-priori bound, not an optimum |
| R2-6 | Positive assessment; "single-contour" reading | n/a | Letter; Why BDS-Lite did not exceed simpler recipes | The reviewer's single-contour reading is **withdrawn** and the letter says why |
| R3-S | Significance vs practical relevance; rationale; efficiency evidence | Yes | Statistical analysis; How large an effect can be resolved?; Deployment cost | Sufficient; run-level margin added |
| R3-A1 | Novelty in abstract | Yes | Abstract; Introduction para 3 | Sufficient |
| R3-A2 | Numbers in abstract | Yes | Abstract | Sufficient |
| R3-I1 | Why these tasks | Yes | Datasets; Table 1; Supp S3 | Sufficient; no longer called a complexity gradient |
| R3-I2 | How boundaries affect results | Yes | How boundary information can reach the prediction; Which errors change | Sufficient |
| R3-I3 | Why not liver/brain tumors | Reasonable; no experiment | Datasets | Text rationale only; no new dataset |
| R3-I4/I5 | Introduction structure; gap | Yes | Introduction (motivation, background, gaps, approach, numbered contributions); Related work | Sufficient |
| R3-I6 | Component choices | Yes | How boundary information can reach the prediction | Sufficient |
| R3-I7 | Recent reference | Optional (staff note) | Datasets (Raj et al. 2026) | Cited once where relevant |
| R3-I8 | Dataset images and boundaries | Yes | Fig. 2 | **Partial.** ACDC/Synapse intensity images omitted (permission unconfirmed) |
| R3-M9 | Per-component gain; nnU-Net, Swin UNETR, Attention U-Net | Partly (premise wrong: no attention gates or multi-scale decoder) | Does distillation add to the decoder?; Other boundary methods; Limitations | Attention U-Net run; **nnU-Net and Swin UNETR not run**, reasons stated |
| R3-M10 | Inter-slice continuity; 2.5D/3D | Yes | The full recipe and its surface loss; Supp S9; Limitations | **Partial.** Volume metrics only; no 2.5D/3D experiment; states continuity is not preserved |
| R3-M11 | Small targets collapsing | Yes | BDS-Lite; Which errors change; Supp S8 | Sufficient; all-case analysis added |
| R3-R12 | Underperformance | Yes | Why BDS-Lite did not exceed simpler recipes; Limitations | Sufficient; mechanisms labelled as interpretation |
| R3-R13 | Qualitative results | Yes (strongly requested) | Fig. 5; Supp S12 | Sufficient for ISIC; label maps for ACDC/Synapse |
| R3-R14 | Tabulated tests, error analysis, ablations | Yes | Statistical analysis; Tables 3-7; Supp S5-S8 | Sufficient |
| R3-R15 | State-of-the-art comparison "mandatory" | Editorially required | Other boundary methods and published results; Table 8 | Context only; not protocol-matched (stated) |
| R3-O | Novelty, small gains, comparator rationale | Yes | Introduction para 3-4; Does boundary distillation add to training-only supervision?; Model sets and compared arms | Answered by conceding the point with evidence |
| S-1 | All comments answered and in manuscript | Required | Letter; location checker | **Open: letter not updated; 27 of 86 locations stale** |
| S-2 | Suggested references optional | Policy | Letter | One citation, relevance stated |

Rows marked partial, and the two comparators not run, are the likeliest points of further reviewer pushback.
