# For the author to confirm (2026-11-11)

Items that could not be sourced, need an author action, or are author decisions.

## Missing material or author action

1. **Code and data availability.** No DOI exists. The submitted version cited the GitHub URL
   `https://github.com/Jingchen-Chen/bds-lite` (a repository URL, not a persistent archive), and
   the licensing audit states that the history of the local branch contains restricted images
   and must not be pushed as is. The "Data and code availability" section was removed from the
   main text at your request. Needed: where the code is supplied (supplemental file, clean
   repository, or archive with DOI) and the statement for the submission system. Placeholders
   remain in Supplementary S16 and twice in `response.md`. The build fails on them only with
   `BDS_RELEASE=1`.
2. **Operating system of the experiments.** Not in the run provenance records. Not stated in
   the manuscript. Supply it if a record exists.
3. **Rationale for SESOI = 0.01.** The protocols record the value and that it was fixed
   (`analysis/revision/prereg_revision.md`, `EXPANSION_PROTOCOL.md`, the pre-registration); none
   records why. `prereg_revision.md` says the thresholds were chosen "after historical findings
   were known, before independent-test effects"; the manuscript now says the same (set after
   earlier development results were known, before the independent-test effects were computed),
   calls them preset analysis thresholds and gives no rationale. Please confirm this timing.
4. **Tracked-changes PDF.** `revision/manuscript/tracked_changes.pdf` predates this rewrite.
   `make_tracked_changes.py` needs a local latexdiff path. The editor letter now carries a
   placeholder instead of the claim that it is included.
5. **Acknowledgments.** The AI-tool statement names ChatGPT only. Decide whether the tools
   used in this revision must be disclosed under PeerJ policy. Funding, competing interests and
   author contributions were not written.
6. **Images of the datasets.** Figure 4 shows ISIC2018 reference and predicted masks only.
   Not shown: ISIC photographs (per-contributor licences, not verified per image) and any ACDC
   or Synapse image or annotation (no stated permission to republish). PeerJ's own figure rules
   could not be read during the licensing audit. Reviewer 3 (points on dataset images and
   qualitative results) is answered as "partly done".

## Decisions taken that you may want to change

7. **Old response letter.** `revision/response_to_reviewers.tex` is marked superseded and no
   longer built. Its PDF is stale and contradicts the new manuscript; I did not delete it.
   Submit `response.md` only, or tell me to port it back to LaTeX.
8. **Earlier-pipeline sentence in the Limitations.** Limitation 2 keeps one sentence on
   re-scored checkpoints of the earlier pipeline (BDS-Lite minus D on ISIC2018 +0.026
   [-0.078, +0.129]; pooled decoder gain -0.005 [-0.017, +0.006]) and on the 4 px tolerance.
   I kept it because boundary rows C02 and Supplementary S15 say these two results are
   pipeline- and tolerance-specific, which changes how they are read. It is not framed as
   history. Remove it only if you judge that S15 suffices.
9. **Task F sources.** The brief named S10, S13 and S22. S22 concerns the ACDC replication
   (reference estimators, error location, D mean across cohorts) and has nothing on the
   BDS-Lite recipe. The paragraph uses S10, S13, S15 and S19 instead.
10. **Length.** Main-text prose is 6,484 words (5,761 in the submitted version; 5,418 before
    this rewrite). The new tables, the framework subsection and the PeerJ items account for the
    growth.
11. **Selection rule of Figure 4.** Reference-mask area at the 10th, 30th, 50th, 70th and 90th
    percentile ranks, seed 1. In the 10th-percentile image D has Boundary F1 0.00 and U 0.13;
    the rule was not changed after seeing this.
12. **Not done by decision:** the augmentation-by-distillation interaction (no value is quoted
    anywhere); a restricted-image version of the qualitative figure.

## Process

13. **Commit blocked for entangled files.** At the start of this session the working tree
    already held uncommitted work of earlier sessions in files this task had to edit (list in
    the completion report). These files are left uncommitted; files that were clean at the
    start are committed.
14. **Claim ledger.** `revision/plan/claim_ledger.md` and
    `revision/manuscript/claim_evidence_table.csv` were not re-audited row by row against the
    rewritten text; the text was checked against `CLAIM_BOUNDARY_v4.md`.
15. **Unrelated generated key.** Regenerating the follow-up assets changes the unused key
    `FU.thermal.resumes.e3` from 40 to 190, because `revision/followup/supervision/events.jsonl`
    has uncommitted additions from another process. I restored the committed value.
16. **Supplement prose** still uses the stage vocabulary and repository paths; it was synced,
    not rewritten.

## Added after the review of 2026-11-11 (second round)

17. **Swin-Unet citation.** The arXiv entry was removed; Table 7 and the supplement now cite the
    ECCV 2022 Workshops version (Cao et al., 2023). The published values (ACDC DSC 90.00;
    Synapse DSC 79.13, HD 21.55) were verified against the arXiv v1 text
    (`revision/sources/literature/2105.05537v1.pdf`, Tables 1–2); the ECCV PDF is not in the
    repository, so the values were not re-checked against it. Confirm that the proceedings
    version reports the same numbers.
18. **TransUNet** keeps two distinct entries on purpose: the 2021 arXiv paper (source of the
    published values in Table 7) and the 2024 Medical Image Analysis paper (cited in the
    Introduction), which are different articles.
