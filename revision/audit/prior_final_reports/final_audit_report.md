# Final Audit Report

- Fixed positioning: preserved.
- Corrected number audit: PASS, 303 rows, zero unresolved.
- Matched GSL protocol: COMPLETE, nine runs, seeds 1–3 on three datasets.
- GSL evaluations: COMPLETE, ISIC2018 validation plus ACDC/Synapse validation
  and test.
- GSL memory equivalence: PASS on distance maps, loss components, total loss,
  and logits gradients in the tractable reference cases.
- Controlled cost profile: COMPLETE for U-Net, BDS-Lite, and U-Net+GSL; local
  hardware evidence only.
- Figure package: Phase 16 matched Figure 2 available; remaining figures
  retain Phase 15 source data and require only human production review.
- Submission recommendation: CONDITIONAL GO.

## Phase 17 integration audit

- Submission-candidate package: created under
  `results/submission_candidate_manuscript_v1/`.
- Corrected numbers: integrated into manuscript and Tables 2–4/6.
- Matched GSL comparison: integrated as the journal main table (Table 2) and
  in Results/Discussion; separate-family wording removed.
- Figure 2: Phase 16 matched replacement copied into the package.
- Efficiency wording: restricted/allowed wording applied; forbidden terms
  absent.
- Mixed/negative results (ACDC null, Synapse distance-negative, mixed GSL):
  retained.
- DOCX: produced via LibreOffice headless (no pandoc available).
- Required checks: PASS (ruff format/check, pytest, git diff --check on the
  Phase 17 paths).
- Submission recommendation: CONDITIONAL GO (no new blocker).
