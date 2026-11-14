# For the author to confirm (2026-11-12, review-round revision)

Adds to `AUTHOR_TO_CONFIRM_20261111.md`; items 1–5 there (code and data availability, operating
system, SESOI rationale, tracked-changes PDF, acknowledgments) are still open. Status on 14 November:
item 1's placeholders were replaced by the GitHub URL (item 12 below), but no DOI exists; item 4's
placeholder is still in `letter_to_editor.md` ("[AUTHOR ITEM: regenerate the tracked-changes PDF …]")
and `tracked_changes.pdf` dates from 11 November; item 5: the acknowledgments still name ChatGPT only.

1. **Figure 3 licence.** Resolved 13 November: the ACDC terms (CC BY-NC-SA 4.0, non-commercial
   scientific research, citation of Bernard et al. 2018) were checked; the author states that PeerJ
   publishes other articles that declare the same licence. The placeholders in S16, R3.11 and the
   editor letter were removed. Journal policy was not re-checked (PeerJ policy pages returned HTTP 403).
2. **Figure 3 inputs** come from `/home/jc/bds-followup/data/raw/acdc_official_test` (outside this
   repository) through your untracked script; I did not regenerate or verify the figure.
3. **Interaction test.** Reported post hoc in Results and S24 (+0.022, 0.011 to 0.032); removed
   from the Abstract on 13 November.
4. **"A recipe of our own design."** The Abstract calls BDS-Lite "our" recipe and the Introduction
   says "a recipe that we designed for compact U-Nets"; response R3.2 now quotes this wording (13
   November, fourth pass). Confirm, and add any earlier publication or preprint of BDS-Lite if one
   exists.
5. **Relation to the first version** is now one short paragraph in Study design and one sentence
   in the Introduction (13 November); details are in the response letter and S4/S5.
6. **Reproducibility wording in the response letter** (Section 2, item 3): it says the submitted
   claim that everything was available was too broad, because the training script is not preserved.
   Check this against what the public repository actually contained.
7. **Length.** Main text 19 pages, 6,505 words of prose (5,761 in the submitted version, +13%;
   three 2025 rows moved to S1; repeated sentences removed); Abstract 300 words (326 submitted);
   supplement 42 pages (was 67), with fixed-model image-level test tables, duplicate
   tables and long listings no longer printed (values in the evidence archive, S16). R2.3 is still
   "Partly done"; change it to "Done" if you are satisfied with the length.
11. **Three-decimal rounding.** The replication interval prints as 0.010 to 0.026; the text quotes
    the unrounded lower bound 0.0098 because the registered reading depends on it. Fig. 2 was
    regenerated with three decimals.
8. **New references.** Yuan et al., CVPR 2020, pp. 3903–3911 (checked against the CVF listing).
   Fourth pass: Sutassananon et al. 2025 (Sci Rep 15:41835), Lin et al. 2025 (Med Image Anal
   103:103615), Dong, Du and Xu 2025 (Front Comput Sci 19(9):199705), checked in Crossref with
   abstracts read; Raj et al. 2026 is now described from its abstract (ViT slice classifier paired
   with a Swin-UNet segmenter), not as one U-shaped network.
9. **Lint.** Resolved 13 November: `ruff.toml` excludes `analysis/qualitative_acdc` and
   `revision/figures/make_qualitative_figure.py` at the author's request; repository-wide ruff passes.
12. **Repository.** The submission's data statement names https://github.com/Jingchen-Chen/bds-lite;
   S16 and both letters now cite it. Status on 14 November: the revision is committed on branch
   `review-edits`; this clone still has no git remote. The GitHub repository is public, its only
   branch `main` was last pushed on 13 July 2026, and none of its commits is in this clone (local
   history starts at 33a2251, 2 November), so nothing of the revision is public yet. Before pushing,
   see the public-release plan of 14 November (reviewer text and a Synapse CT image are in local
   history). Co-author approval is covered by the submission form.
10. **Old LaTeX letter.** `response_to_reviewers.pdf` is now rendered from `response.md`;
    `response_to_reviewers.tex` remains superseded and unbuilt.
13. **No DOI** (resolved 14 November, history-free snapshot release). The author has no DOI and
    wants none cited: the `[ZENODO DOI]` placeholder added earlier the same day was removed from
    Supplementary S16, the public README and `CITATION.cff`. The response letter, editor letter and
    PeerJ checklist give only the repository URL. Software author in `CITATION.cff` and
    `.zenodo.json`: Jingchen Chen only (author's decision); the article citation keeps all four
    authors.
