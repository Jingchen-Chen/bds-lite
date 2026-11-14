# Decisions and checks that require the authors

## Current author decisions after the editorial review (2026-11-08)

This list supersedes the historical lists below. The active manuscript, supplement and
`response_to_reviewers.tex` now include the ACDC replication and Experiment C disclosure.
The active response's locations are regenerated from the PDFs; `response_letter_v2.*` is historical.

- Resolve the visible affiliation/present-address, earlier-public-PDF/preprint note, revision DOI
  and co-author-approval markers. Review the AI tool inventory as recorded in Acknowledgments.
- Commit your existing stop/deviation/queue records for Experiment C. Agents leave them unstaged.
- Complete the release decisions in `RELEASE_GUIDE.md`, including the earlier public image/history
  issue. The revision release is prepared locally; it has not been published.
- Confirm submission-system authorship, affiliations, funding and competing interests before
  any resubmission. No journal communication or submission occurred during this editorial review.
- The approved claim boundary still has its old DRAFT header; its authority comes from `AGENTS.md`.

Scientific and layout decisions: `EDITORIAL_REVIEW_20261108.md`. Numerical and build evidence:
`../results/final_manuscript_audit_v2/final_audit_report.md`.

---

## Current list after the manuscript reorganization (2026-11-07). Supersedes the lists below where they differ.

The authors chose to reorganize the manuscript and run the historical re-evaluation in parallel with Experiment C. Pre-reorganization files: `revision/audit/manuscript_snapshot_20261107_pre_reorg/`.

1. **Read the reorganized manuscript.** It now answers the title by five conditions, and it no longer calls the decoder the best-supported result. The original BF1 gain stays out of the lead.
2. **Experiment C disclosure (still open, now pressing).** The authors cancelled ISIC2018 seeds 11-17 and then stopped all remaining Experiment C training on 2026-11-07 (`DEVIATIONS.md`; `CONFIRMATION_SCOPE_AMENDMENT_20261107.md`; `STOP_TRAINING`). The registered confirmation cannot run as written, and an amended analysis would have to be frozen before any of its test metrics is read. The manuscript, the S15 record, and the letter still omit Experiment C; the letter's `\human` marker is unchanged. Decide the disclosure wording and whether any amended analysis is run. No Experiment C metric has been read.
3. **Historical re-evaluation: integrated.** Finished 2026-11-07, validation PASSED (63 folders; 51 of 51 parity checks). Integrated by Section 9 of the protocol: three sentences in Results (Relative to what?; Why the transfer changed little), one in Limitations, full tables in Supp S15. Consequences to check: the confirmed BDS-Lite deficit is now stated as specific to the retrained set (abstract, Discussion, Conclusions); the decoder's direction did not hold across pipelines; BF1 statuses depend on tolerance (at 4 px, decoder vs U-Net not established, BDS-Lite vs decoder supported). Nothing became Confirmed.
4. **Commit: done locally** (no push). Earlier note: Repository lint fails on pre-existing issues in `analysis/revision/make_expansion_assets.py` and `seed_subset_analysis.py`, so the protocol freeze is recorded by SHA-256 only. Decide whether to fix those two files and commit, which would give the freeze a git timestamp.
5. **Alpha and weights provenance: resolved from the devlog.** The manuscript now says they came from single-seed validation runs and were frozen before the matched evaluation. `response_letter_v2.tex` line 171 still says otherwise and must change with the letter rewrite.
6. **Response letter.** Rewritten on 2026-11-07 against the reorganized manuscript: all 35 comments verbatim. Updated for the historical re-evaluation (summary list, R1-4, R3-R14, R3-O); 104 locations verified. Still to update for Experiment C once item 2 is decided.
7. **Unreported families.** Supp S15 now lists all earlier families, including the April grid, EGE-UNet, and the p2 and p2b runs. Confirm that you are content to disclose them.

---

## Current list after the narrative revision (2026-11-07). Supersedes the lists below where they differ.

1. **Experiment C disclosure.** The manuscript omits it as instructed, but `NARRATIVE_CRITERIA_ADDENDUM.md` rule 5 says every experiment started is reported, and `CONFIRMATION_PROTOCOL.md` will be in any public release. If it finishes before submission or publication, decide whether to add a disclosure or its results. A primary result against the decoder would change the abstract and conclusions.
2. **Response letter.** `revision/response_letter_v2.tex` predates the ten-seed results and this rewrite. 27 of 86 locations are stale, and it still says for example that GSL did at least as well as BDS-Lite on every dataset. It must be rewritten before resubmission. It must also list the ACDC decoder correction.
3. **Title.** Kept, as `plan.md` 5.1 and `EXPANSION_PROTOCOL.md` section 9 fix it. Option if you prefer the sharper question: "Does boundary distillation add to training-only boundary supervision in compact medical image segmentation?" Changing it also means updating the submission system.
4. **Code availability.** The bracketed marker is gone from the text, so the statement is true only after the revision code is pushed and tagged. Add the Zenodo DOI if you mint one.
5. **Figure 1.** Corrected by `analysis/revision/patch_architecture_figure.py` (distillation target now leaves q_b; b_g added). Check it, or replace it from your source file.
6. **Still open from the list below:** provenance of alpha = 0.25 and of the loss weights as fixed before the experiments; AI statement; Synapse class names; PeerJ limits and metadata.

---
## Current list after the manuscript reconstruction (2026-11-05). Supersedes the lists below where they differ.

1. **Confirm the new emphasis.** `main.tex`, Results "BDS-Lite versus U-Net" and Abstract. Item 2 below approved the submitted models as primary. They are still reported first, but the text now treats an effect as established only if it appears in both model sets, because the retrained set does not reproduce the Boundary F1 gain. If you disagree, the Abstract, that subsection, and Conclusions need changing.
2. **Confirm the negative framing.** The paper now says plainly that distillation, the gate, and the surface loss gave no measurable benefit. Title unchanged. This is your scientific position to accept.
3. **Confirm two provenance statements I could not verify from records.** `main.tex` BDS-Lite: "We fixed alpha = 0.25 ... before the experiments" and "The weights were fixed before the experiments". Inherited from earlier text.
4. **AI statement** (`revision/manuscript/acknowledgments_revision.tex`, unchanged as requested). It says every number comes from released scripts, which holds only once item 5 is done.
5. **Release, then DOI.** Push the release fixes and add `analysis/revision/make_rewrite_assets.py` and `validate_rewrite.py`, tag v2, mint the Zenodo DOI, then replace `[HUMAN: tag v2-revision and Zenodo DOI]` in `main.tex` (Data and code availability) and rebuild. The submitted PDF must not contain that marker.
6. **Figure 1** (`revision/manuscript/figure_architecture.png`, your source file). The arrow into L_distill leaves the 1x1 conv block, but the target is q_b, before that conv. The gate formula also omits the bias b_g. Please correct.
7. **ACDC/Synapse images** (Fig. 2, Supp S12). Omitted because permission is unconfirmed. Ask PeerJ if you want them; `dataset_examples_with_images.*` is ready.
8. **PeerJ limits and metadata.** Confirm abstract and section requirements on PeerJ's site (279 words now; the page was not readable from here). Enter authors, funding, and competing interests in the submission system.
9. **Optional, would strengthen the paper:** if any copy of trainer commit 5914d35 exists, it could explain the U-Net gap between sets. More seeds on ACDC would tighten the run-level margin. No training was run for this rewrite.
10. **Confirm Synapse class names** (inferred; Supp S3), and read all PDFs in full.

---
## Earlier list (updated after the GitHub audit, 2026-11-02)

Ordered by impact. Nothing has been sent, submitted or pushed. Evidence: revision/GITHUB_AUDIT.md;
strategy: revision/REVISION_STRATEGY.md.

0. **APPROVED 2026-11-02:** items 1 and 2 below (submitted models stay primary; ISIC official test replaces the 519-image evaluation). Verification completed: revision/validation/official_isic_verification.json; repeated-lesion sensitivity: supplement S11.
0b. **Zenodo deposit? (downgraded 2026-11-03: optional)** The GSL retrain reproduces the Phase16 training logs exactly (GITHUB_AUDIT.md U1 update), so the original GSL checkpoints are no longer needed; still useful only as a byte-level confirmation. Did you deposit the release bundles (`checkpoints_gsl` with the nine Phase16 U-Net+GSL checkpoints)? They are not on this machine. If they exist, download them into `results/gsl/checkpoints/`; they would allow evaluating the original GSL models on the official ISIC test and make the GSL retrain a replication rather than a replacement.
1. **ISIC2018 correction (critical).** The submitted ISIC U-Net/BDS-Lite checkpoints were trained on a
   1,815/389 partition (split seed 1) but reported on the 2,075/519 partition's 519 images, 358 of
   which were their training images. Proposed: withdraw the 519-image ISIC numbers and evaluate the
   *same submitted models* on the official 1,000-image test (done: Boundary F1 +0.031
   [0.026, 0.036], practically superior; DSC/HD95/ASSD practically equivalent). Approve, or choose
   an alternative (e.g. retrain ISIC on the documented partition).
2. **Primary foundation.** The draft keeps the submitted models as the primary comparison (ACDC and
   Synapse reproduce the original Table 3). My earlier proposal to replace them with the p1b family
   is withdrawn; p1b is used only as replication and as the pipeline-matched reference for
   ablations/comparators. Confirm.
3. **Retractions in my earlier drafts:** ACDC class weights concern only p1b, not the submission.
   Already corrected in manuscript/letter.
4. **Clinical metrics (E-4/R2-2).** EF / myocardial mass / mm need native ACDC NIfTI (pixdim, ED/ES);
   decide whether to obtain them.
5. **Licences (WP7a).** ACDC/Synapse intensity images withheld (`*_with_images` variants ready).
6. **Reviewer text.** Paste verbatim comments and line numbers into revision/response_letter_v2.tex.
7. **R3-I7 citation** (Raj et al. 2026): neutral scope citation; keep or delete.
8. **Synapse class names** inferred from descriptors; confirm.
9. **Tier C** (extra dataset / 2.5D / nnU-Net): not planned.
10. **[CLAIM-CHECK]** items in main.tex (ABCD-rule and radiotherapy OAR citations; Table 1 metric families).
11. **AI disclosure, abstract word limit, author metadata, final full read, tag + Zenodo DOI.**
12. **Public GitHub release defects** (seed-collision in release train.py, GSL script path, deleted
    Zenodo manifest, incorrect 'vestigial split_file' documentation): fix before tagging v2.

## Status 2026-11-05 (defaults applied; reverse any of them if you disagree)

Resolved without author input:
- **#10 CLAIM-CHECK:** ABCD rule (Nachbar et al. 1994) and abdominal OAR contouring (Jabbour et al. 2014)
  cited, BibTeX taken from the DOI registry; Table 1 metric families checked against each source paper
  (GSL also reports ASD; Boundary DoU reports HD and Boundary IoU; Psi-Net Dice/Jaccard/HD). No CLAIM-CHECK left.
- **Framing:** Contributions paragraph and abstract now put the controlled characterization first and
  describe BDS-Lite as the recipe under study.
- **Abstract length:** ~330 words, shorter than the submitted abstract (~370); PeerJ's own limit page
  could not be read (JavaScript-only), please confirm.
- **#4 Clinical metrics:** kept as limitation (letter decision placeholder removed).
- **#5 Licences:** ACDC/Synapse remain label-map panels; letter says image versions can be added if
  the editor confirms permission.
- **#7 Raj et al. 2026:** kept as a neutral scope citation.
- **#9 Tier C / nnU-Net / extra dataset:** not run; letter gives the reasons.
- **Letter:** word-count answer generated (WC.* keys); all 61 named locations verified
  (`analysis/revision/check_letter_locations.py`).
- **#12 Release defects R1-R5:** fixed in a separate clone `/home/jc/bds-lite-release-v2`, branch
  `release/v2-fixes`, commit adccd1b on top of GitHub 3104cb3 (unit tests pass). **Not pushed.**

Update 2026-11-05 (decision letter supplied by the authors):
- Letter: all 33 comment slots now quote the decision letter verbatim ([\ldots] marks omissions);
  added R1-0 (Reviewer 1 general assessment) and S-2 (staff note on suggested references); internal
  F-codes removed from the letter and manuscript headings; 71 named locations verified, 0 missing.
- Clinical relevance (E-4, R2-2): Discussion ``Clinical relevance'' now relates each measured error
  type to its clinical use per task, using only generated first-order keys (S.*) and existing results.
- Tracked changes: `revision/manuscript/tracked_changes.pdf` built by
  `analysis/revision/make_tracked_changes.py` (latexdiff 1.3.4, pinned sha256; submitted source
  `recovered/peerj-submission/main.tex` vs revised main.tex). Rebuild after any manuscript edit.
- The decision letter gives no date and only a soft 30-day target; the letter no longer asks for it.

Still requires the authors:
1. Confirm the AI-use statement (`revision/manuscript/acknowledgments_revision.tex`).
2. Push the release fixes, merge the revision code, tag v2 and mint the Zenodo DOI
   (Data availability placeholder in main.tex), then rebuild main.pdf and tracked_changes.pdf.
3. Submission-system metadata (authors, order, contributions, affiliations, funding, competing
   interests): PeerJ uses only the system entries and they cannot be changed after acceptance.
4. Confirm the abstract limit on PeerJ's site; read the compiled PDFs in full.
5. If the resubmission is later than ~30 days after the decision, optionally add one sentence to the
   letter's opening thanking the editor for the extra time.
