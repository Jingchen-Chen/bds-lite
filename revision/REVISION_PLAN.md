# Revision plan: PeerJ CS-2026:06:142937 (second-round rewrite, 2026-11-08)

Governing documents: `AGENTS.md`; the claim boundary `revision/CLAIM_BOUNDARY_v2.md` (approved
2026-11-08, committed `8e19749`, pointed to by `072bcdb`); reviewer comments
(`/home/jc/Downloads/reviewer.md`, quoted verbatim in the response letter); submitted version
`recovered/peerj-submission/main.tex` (identical in text to the review PDF
`peerj-reviewing-142937-v0.pdf`).

Rule set used throughout: every number is a `\R{}` key generated from a committed file; every claim
sits inside an A-row of the claim boundary; F1-F17 are never crossed; D1-D8 and Dev-1-Dev-3 are
disclosed (main text: one sentence and a pointer; supplement: detail).

---

## 1. The argument

**Three sentences.**
1. Boundary distillation can only be credited with what it adds to the training-only boundary
   supervision it builds on, and effects of this size are as small as the variation between
   retrainings, so we judge it against a decoder-only reference and over independent runs.
2. Judged this way, BDS-Lite's in-network boundary feature distillation did not establish a gain in
   any studied condition, whereas external-teacher boundary-band distillation gave positive
   exploratory estimates on all three datasets, none surviving Holm correction, and, in a
   pre-registered replication on held-out ACDC patients (ACDC selected after those results), raised
   Boundary F1 by +0.0178 [95% CI +0.0098, +0.0258], of unclear size.
3. In our experiments boundary distillation therefore helped in one condition only, external-teacher
   boundary-band distillation on ACDC, and the data neither identify which aspect of that condition
   is responsible nor show that the gain transfers to other datasets.

**Answer to the title question.** Claim boundary §4, verbatim in substance. The Discussion's first
subsection gives it with a summary table; the abstract and Conclusions give it in short form.

---

## 2. Inventory of experiments and decisions

Decision key: **Main** = main text; **Supp** = supplement only; **Disclosed** = named, with no result
used; **Omitted** = listed only in the run ledger (`revision/audit/project_audit_20261107/experiment_ledger.csv`,
summarized in Supp S15). Disclosure obligations D1-D8 and Dev-1-Dev-3 are mandatory and are marked.

### 2.1 Training studies

| # | Study (dates 2026) | Runs | Decision | Where | Reason |
|---|---|---|---|---|---|
| T1 | Single-seed exploration that set alpha and the loss weights (Apr 26-27) | 37 | Disclosed | Methods (provenance of alpha, lambda); S15 ledger | Reviewer 2 asked how alpha = 0.25 was chosen; tuned on validation, so no result is used |
| T2 | Original 13-arm grid: U-Net and BDS-Lite (the "original set", the submitted models) (Apr 28-May 3) | 18 | Supp + one Results sentence | S4, S5; Results "How large an effect" | The submitted ISIC2018 result must be corrected; same recipe, different pipeline shows how far a recipe moves |
| T3 | Original grid, other 11 arms | 99 | Supp (four arms re-evaluated post hoc); rest Omitted | S15; Results one sentence (A1.6) | A1.6 bounds the scope of the confirmed ISIC2018 deficit (claim boundary A1.5 wording limit) |
| T4 | p1 gate pilot (May 7) | 15 | Omitted | ledger | Superseded by the matched gate ablation |
| T5 | p1b U-Net and BDS-Lite references (May 7-8) | 36 | Main | Retrained set | References of the retrained set |
| T6 | p2 GSL and Boundary DoU baselines (May 8) | 18 | Omitted | ledger | Superseded by the reproduced GSL; DoU ACDC baseline flagged in April audit |
| T7 | p2b BDS-Lite without gate + GSL (May 8-9) | 9 | Omitted | ledger | Not validated under the current evaluator |
| T8 | EGE-UNet (May 22) | 9 | Omitted | ledger | Evaluated on the contaminated 519-image set; dropped |
| T9 | Phase 16 U-Net+GSL (Jun 9-10), the submitted comparator | 9 | Supp | S7 (exact reproduction by retrain) | The submitted GSL rows are reproduced, so the retrain replaces them |
| T10 | E1 long-schedule GSL (May 10) | 1, stopped | Omitted | ledger | Invalid: GSL term inactive |
| T11 | October ISIC2018 arms U, S, B, F (Oct 17-18) | 12 | Main | Same models as retrained cells M0, M1, M3, M6 | Part of the retrained set |
| T12 | October development controls G, C, D | 9 | Supp + one Results sentence | S15 | Gate-cost hypothesis; development images, post hoc test re-evaluation |
| T13 | October augmentation arm UA | 3 | Supp + Limitations sentence | S15; Limitations | Quantifies the "no augmentation" limitation |
| T14 | October ACDC controls (three-slice context and others) | 15 | Three-slice: Supp + Limitations sentence; rest Omitted | S15 | Reviewer 3 comment 10 (2.5D) |
| T15 | Registered revision runs: ablations M1-M5, comparators GSL, C1, C3, C4 (Nov 2-5) | 75 | Main | Results; Table "arms" | Ablation and comparator requests (editor, R1, R3) |
| T16 | Gradient diagnostics, seed 1 (Nov) | 3 | Supp | S10 | Single-seed mechanism context |
| T17 | alpha and lambda sensitivity | 18 | Supp + one Results sentence | S11 | Reviewer 2 comment on alpha |
| T18 | Frozen expansion: ISIC2018 seeds 4-10 of four arms, decoder+surface cell, corrected ACDC decoder (Nov 5-7) | 40 | Main | Results; Table ten-run contrasts | Fresh-seed confirmation; factorial; A1.5 |
| T19 | Expansion and Experiment C reproduction gates | 12 | Supp sentence | S13 | Pipeline check, no test evaluation |
| T20 | **Experiment C** (Nov 7), stopped by the authors | 70 registered: 14 completed, 1 incomplete, 55 never started | **Disclosed (D7)** | Methods one sentence; S15 | Required disclosure; no result used (F17) |
| T21 | ISIC2018 external-teacher band KD: teacher + D+KD-band seeds 1-3 | 4 | Main | Results; Table KD; S17 | A2.1 |
| T22 | ISIC2018 KD seeds 4-10 and D+KD-full control | 0 (cancelled before any result, Amendment 1) | Disclosed | S17 | Explains why there is no D+KD-full control (F1) |
| T23 | ACDC and Synapse external-teacher band KD: 2 teachers + 6 students | 8 | Main | Results; Table KD; S17 | A2.2, A2.3 |
| T24 | **Pre-registered held-out ACDC replication**, seeds 201-217 | 34 | Main + new figure | Results; Fig. KD; S18 | A3, A4; D1-D6, D8, Dev-1-Dev-3 |
| T25 | Pre-freeze trainer equivalence check (50 steps) | 3 paths | Supp | S18 | Basis of the claim-boundary §0.2 statements |

### 2.2 Evaluation-only studies

| # | Study | Decision | Where | Reason |
|---|---|---|---|---|
| V1 | Official ISIC2018 test evaluation of all ISIC2018 models | Main | Methods, Results | Replaces the withdrawn evaluation |
| V2 | Prediction recovery and its validation | Supp | S4, S5 | Provenance |
| V3 | Volume-level metrics | Supp + one sentence | S9; Results "errors" | Reviewer 3 comment 10 |
| V4 | Error location, missed structures, false alarms, all-case distances | Main (short) + Supp | Results "errors"; S8 | Reviewer 3 comments 11, 14 |
| V5 | Surface-distance CDF | Omitted | - | Not needed for any claim |
| V6 | Gate removal at inference | Main sentence | Results "Cost" | Deployment question |
| V7 | Mechanism probes (teacher head on q_s, readout from F_s, gate structure, boundary precision/recall) | Main (three sentences) + Supp figure | Results; S13 | Explains why the in-network transfer changed little |
| V8 | Image-level repeatability (seed subsets) | Main (two sentences) + Supp | Results "How large"; S14 | Statistical rigor requests |
| V9 | ISIC2018 membership decomposition of the withdrawn result | Supp + one sentence | Methods; S4 | Root cause of the correction |
| V10 | Validation-image contrasts | Main sentence + Supp | Results; S13 | Scope of the decoder gain |
| V11 | Historical re-evaluation and Boundary F1 tolerance | Main (two sentences) + Supp | Results; S15 | A1.6; tolerance dependence |
| V12 | Deployment and training-cost profile | Main | Results "Cost" | Reviewer 3 efficiency request |
| V13 | Repeated-lesion sensitivity | Supp + pointer | S4 | Independence of the official ISIC2018 test |
| V14 | Deterministic qualitative selection | Main figure | Results "errors" | Reviewer 3 comment 13 |
| V15 | Preprocessing reconstruction gate and its impact (confirmation) | Main sentence + Supp | Methods; S18 | D3 |
| V16 | Seed correlation (Dev-3) | Supp | S18 | D2 |
| V17 | Exploratory ACDC models on the official set; replication seeds on the internal split; seed-paired sensitivity | Supp | S18 | A3.5, A5.1, A5.2: descriptive context only (F7) |
| V18 | Published results verified against their sources | Main table | Results | Reviewer 3 comment 15 |
| V19 | Older evaluation-only studies (May-June audits, ACDC audits; MedSAM weights never run) | Omitted | ledger | No claim depends on them |

---

## 3. Section-by-section outline (main text target <= 5,500 prose words; submitted: 5,761)

Word counts use `analysis/revision/make_manuscript_assets.py::prose_words` (Introduction to
Acknowledgments; tables, figures and LaTeX removed; each key one word).

| Section | Purpose | Evidence | Displays | Budget |
|---|---|---|---|---|
| Abstract | Question, design, numbers, answer | A1.1-A1.6, A2, A3 | - | <= 300 words, < 3,000 characters |
| Introduction | Motivation (contours; clinical role in one sentence per task), background, gaps (reference; measurement), approach, contributions | Lit; S.* first-order keys | - | 600 |
| Related work | Literature with contributions and limitations; reviewer-requested references engaged | Lit | Supp S1 table | 320 |
| Methods: BDS-Lite | Exact method; alpha role and provenance; stop-gradient definition; losses; small-structure handling (Focal Tversky named as not used) | Code; devlog provenance | Fig. 1 architecture | 430 |
| Methods: routes to the prediction | Theoretical framework (three routes; first-order DSC relation; why Boundary F1 is primary) | - | - | 220 |
| Methods: external-teacher band distillation | Second design, defined against the same decoder-only network; deploys as plain U-Net | KD protocols | - | 180 |
| Methods: datasets | Why these tasks, why not LiTS/BraTS; ISIC2018 correction; official ACDC test patients | tab_datasets; S4; D3 | Table 1; Fig. 2 synthetic targets | 330 |
| Methods: training, arms and studies | Protocol and hardware; model sets; arms; five studies and their registration; Experiment C sentence | Protocols | Table 2 (studies) | 380 |
| Methods: metrics and statistics | Metrics; recipe-level estimator; readings; confirmation rules; KD Holm; replication decision rule | Protocols | - | 380 |
| Results: how large an effect can be resolved | Retraining variation; image-level tests mislead; two pipelines | XM, SS, RL, P keys | - | 170 |
| Results: in-network distillation vs decoder | Decoder vs U-Net; full vs decoder (A1.5, scoped); transfer path (A1.1-A1.4); attribution; mechanism; alpha | X, I, K, MECH, H keys | Table 3 (ten-run contrasts), Table 4 (central), Fig. 3 (forest) | 480 |
| Results: external-teacher band distillation vs decoder | A2 exploratory; D1 selection; A3 literal; A4 with HD95; D4 overlap; Dev pointer | KD keys | Table 5 (KD), Fig. 4 (new) | 430 |
| Results: comparators and published results | GSL lead vanished; LB-UNet disagreement; literature context | C, X keys | Table 6 (all arms), Table 7 (literature) | 220 |
| Results: errors and qualitative examples | Error location; missed small structures; all-case sign change; qualitative | ERR, CB, AC, QUAL keys | Fig. 5 (contours) | 220 |
| Results: cost | Deployed overhead; KD costs (D8) | RESX, TRAIN, KD cost keys | Table 8 (resource) | 120 |
| Discussion: when does boundary distillation help | The answer (claim boundary §4); withdrawn single-contour conclusion | All above | Table 9 (summary) | 260 |
| Discussion: what the two designs differ in | Hypotheses only (F1, F3, F4, F15) | Mechanism | - | 190 |
| Discussion: practical implications and clinical relevance | Order of work; per-task clinical role; no clinical claim | S.* keys | - | 180 |
| Discussion: limitations | Pointers, not repetition | - | - | 220 |
| Conclusions | Short answer and recommendation | - | - | 130 |

Moved out of the main text (to the supplement, with one-sentence pointers): ten-run scatter figure
and means table (S13), mechanism figure (S13), error-geometry figure (S8), additive component table
(S13), two-model-sets table (S5), ISIC2018 correction details (S4), development controls,
historical re-evaluation, tolerance (S15), confirmation details (S18), KD details (S17), Experiment C
(S15).

Four sentences that said "could not establish any condition" (old main.tex lines 35, 56, 351, 400)
are narrowed to BDS-Lite's in-network distillation (F8) in the abstract, Introduction, Discussion
and Conclusions.

---

## 4. Reviewer matrix

IDs follow the response letter. "Flag" marks comments the evidence cannot fully answer; the reply
says so plainly.

| ID | Comment (short) | Change (revised section) | Evidence | Planned reply | Flag |
|---|---|---|---|---|---|
| E-1 | Experimental evaluation: comparisons, ablations, statistical rigor, qualitative results | Methods "Training, arms and studies"; Results (all subsections) | T15, T18, T21-T24; V4, V8, V14 | Six ablations, four comparators, ten-run arms with fresh-seed confirmation, two distillation designs, pre-registered held-out replication, qualitative figure | - |
| E-2 | Clarity and theory of the method | Methods "BDS-Lite", "routes" | Code | Equations, stop-gradient, alpha, three routes, first-order relation, measured routes | - |
| E-3 | Why only compact U-Nets | Introduction; Limitations | - | Deployment budget and controlled comparison; scope stated | Partial: no larger backbones trained |
| E-4 | Clinical relevance and impact | Introduction; Discussion "practical implications and clinical relevance" | S.* keys | Per-task clinical role; no clinical indices because spacing and frames could not be verified | Partial: no clinical indices |
| S-1 | Staff note: every letter change in the manuscript | Whole | Location check | Every pointer checked against the compiled PDF | - |
| S-2 | Staff note: suggested references only if relevant | Introduction; Datasets | Crossref | One suggested reference, cited where relevant | - |
| R1-0 | General assessment | - | - | Thanks; the requested ablation changed the answer | - |
| R1-1 | Literature review with contributions and limitations | Related work; S1 | Lit | Added | - |
| R1-2 | Theoretical framework | Methods "routes"; Results mechanism | V7 | Framework stated before results and tested | - |
| R1-3 | Comparison with state of the art, even reproduced | Results "comparators"; Table 7 | T15, V18 | Four comparators under one protocol; published results as context | Partial: no nnU-Net |
| R1-4 | Absence of ablation prevents attribution | Results "in-network distillation"; Table 4 | T15, T18 | Ablation with decoder-only reference; attribution limits (interaction unresolved) | - |
| R2-1 | Compact U-Net focus not justified | Introduction | - | As E-3 | Partial |
| R2-2 | Clinical relevance of boundary errors | Introduction; Discussion | S.* keys | As E-4 | Partial |
| R2-3 | Verbose | Whole | Word counts | Main text shorter than submitted despite new results; numbers in letter | - |
| R2-4 | stopgrad(q_b) undefined | Methods "BDS-Lite" | Code | Defined | - |
| R2-5 | alpha purpose and 0.25 | Methods "BDS-Lite"; S11 | T1, T17 | Role, provenance (single-seed validation), sensitivity | - |
| R2-6 | Positive assessment; benefits for single-contour datasets | Discussion "when does it help" | A1, A3 | Thanks; single-contour conclusion withdrawn: it rested on the invalid ISIC2018 evaluation; the only established gain is on ACDC with a different design | - |
| R3-S | Significance vs practical relevance; justification vs existing methods; deployment efficiency evidence | Methods "statistics"; Results "cost" | SESOI; V12 | Smallest effects of interest; recipe-level intervals; cost table | - |
| R3-A1 | Novelty not defined in abstract | Abstract; Introduction | - | We state what is examined and what is new (the matched test, the replication), and do not claim novelty for the components | - |
| R3-A2 | No quantitative results in abstract | Abstract | A1-A3 | Added | - |
| R3-I1 | Why these three tasks | Methods "datasets" | Table 1 | Structure, number, absence; not a difficulty scale | - |
| R3-I2 | How boundaries impact results | Methods "routes"; Results "errors" | S.*, ERR | First-order relation; where errors lie | - |
| R3-I3 | Why not liver/brain tumour | Methods "datasets" | Raj et al. | 3D and multimodal, outside the compact 2D scope and budget | Partial: not run |
| R3-I4 | Introduction structure | Introduction | - | Motivation, background, literature, gaps, approach, contributions | - |
| R3-I5 | Research gap and how addressed | Introduction | - | Two gaps; matched reference and recipe-level statistics | - |
| R3-I6 | Component selection | Methods "BDS-Lite", "routes" | - | Purpose and expected effect of each; tested | - |
| R3-I7 | Recent references for the segmentation claim (doi:10.1002/ima.70424) | Introduction; Datasets | Crossref record | Cited substantively in both places | - |
| R3-I8 | Dataset images with annotated boundaries | Methods "datasets" (Fig. 2); Results (Fig. 5) | LICENSING_AUDIT | Synthetic targets; CC0 ISIC contours; no photographs or ACDC/Synapse annotations | Partial: licence |
| R3-M9 | Gain per component; nnU-Net, Swin UNETR, Attention U-Net | Results; Limitations | T15 | Ablation of each actual component; Attention U-Net trained; nnU-Net and Swin UNETR not run, with reasons | Partial |
| R3-M10 | 2D vs 3D/2.5D continuity | Results "errors"; Limitations; S9, S15 | V3, T14 | Not preserved; volume metrics; development 2.5D control as context only | Partial |
| R3-M11 | Small targets and collapse to background | Methods "BDS-Lite"; Results "errors" | CB keys | Per-class Dice; missed-structure rates; no collapse, small structures not protected | - |
| R3-R12 | Underperformance; justify limitations | Results; Discussion | A1, V7 | Agree; measured reasons; limitations | - |
| R3-R13 | Qualitative comparison with ground truth | Results; Fig. 5; S12 | V14 | Deterministic selection; CC0 contours | Partial: ISIC2018 only |
| R3-R14 | Significance tests in tables, error analysis, ablation | Tables 3-6; S5-S8, S13-S18 | All | Tables of contrasts with intervals and Holm p; error analysis | - |
| R3-R15 | State-of-the-art comparison mandatory | Results; Table 7 | V18 | Published results as context; protocols differ; nnU-Net not run | Partial |
| R3-O | Novelty, modest gains, comparator rationale | Abstract; Introduction; Discussion | All | No novelty claimed for BDS-Lite; contribution is the test and the replication; comparator rationale | - |

---

## 5. Critique of the plan

### Round 1

1. *Is the logic linear?* Mostly. The weak point is chronology: the external-teacher study was
   added after BDS-Lite's distillation failed and after the first submission. A reader could take it
   as planned from the start. **Revision:** the studies table gives each study's order and
   registration status, and the Introduction says the second design was added during the revision.
2. *Does every section earn its place?* "Related work" and the Introduction's background overlapped
   in the previous draft (both listed the method families with citations). **Revision:** the
   Introduction keeps two sentences of background and points to Related work, which carries the
   contributions and limitations Reviewer 1 asked for. "What changed from the original submission"
   duplicated the correction paragraph and the Introduction. **Revision:** folded into two sentences
   of the first Discussion subsection.
3. *Redundancy.* The replication's limits are part of its pre-registered wording (A3) and were
   also listed in Limitations. **Revision:** Limitations refers back instead of repeating. The ISIC2018
   correction is stated in full once (Methods) and referred to elsewhere.
4. *Would a skeptical reviewer accept every sentence?* Likely objections and answers:
   - "ACDC was chosen after seeing results": stated as post hoc (D1) wherever the replication is
     reported; Holm-adjusted exploratory p reported beside it.
   - "The KD gain may come from a strong pretrained teacher, not from boundaries": the text says the
     data cannot identify which aspect is responsible (F1, F3).
   - "+0.018 Boundary F1 is small": labelled "of unclear size" against a pre-registered 0.01; no
     relevance claim for secondary metrics (F16); HD95 not established is reported.
   - "Decoder-only arms differ between A1 and A3": the summary table caption names the differences
     (seeds, runs, trainer, test patients) and the rows are not compared (F14, F15).

### Round 2

1. *Linear?* Results order checked: precision first (why intervals look the way they do), then
   design 1 against the reference, design 2 against the same reference, then context. The
   comparators come after both designs because they answer "would something standard do as well",
   which is secondary to the title. Accepted.
2. *Each section earns its place?* "How large an effect can be resolved" could be cut to Methods,
   but the retraining SD and the image-level repeatability are measured results, and they justify
   the estimator; kept at about 170 words. The mechanism figure is secondary and post hoc and moves
   to the supplement; three sentences stay. The error-geometry figure moves to the supplement; the
   qualitative figure stays (Reviewer 3, comment 13).
3. *Redundancy.* The summary table in the Discussion repeats numbers given once in the Results.
   Accepted as the single place where the answer is assembled; the Discussion prose refers to the
   table and does not restate the numbers. The Conclusions contain no numbers.
4. *Skeptical reviewer.* Two further objections: (a) "the four comparators were trained under your
   protocol, which may handicap them": stated once, in Limitations, with LB-UNet named; (b) "the
   replication's decoder-only arm used a different trainer from the exploratory one": stated in S18
   with the 50-step bitwise check and its limit (§0.2). Accepted.
5. *Budget.* Sum of budgets 5,460 words, below 5,761. Abstract <= 300 words.

The plan is stable after round 2; no further structural change was justified.

---

## 6. Work log

- 2026-11-08T14:52:21Z: `analysis/revision/auto_refresh.py` (pid 1101238) stopped with SIGTERM
  before any manuscript file was touched; it was idle (no child step running). Logged in
  `revision/auto_refresh.log` (git-ignored). Its last refresh (14:01:20Z) had rewritten only the
  word-count keys in `generated/numbers.tex` and re-rendered two figure PDFs.
- 2026-11-08: commits `8e19749` (claim boundary, unmodified), `072bcdb` (`agents:` pointer),
  `936da2e` (Experiment C metric-free status record and scanner; scanner reformatted for ruff,
  output byte-identical).
- Left unstaged (the author's own edits): `revision/DEVIATIONS.md`,
  `revision/CONFIRMATION_SCOPE_AMENDMENT_20261107.md`, `revision/queue_plan.yaml`,
  `revision/queue_confirmation_acdc_synapse.yaml`.
- External checks: PeerJ AI policy read from the Internet Archive copy of
  `peerj.com/about/policies-and-procedures/` (2026-05-14; the live page is rendered by JavaScript);
  PeerJ CS abstract limit 500 words / 3,000 characters (archived author instructions, 2025-11-07).
  New references verified in the Crossref registry: 10.1002/ima.70424, 10.1007/978-3-031-08999-2_22,
  10.1109/ISBI.2019.8759329.
- 2026-11-08, drafting and verification: manuscript rewritten (`0ba032f`); letters and build chain
  (`30f4f02`); release guide (`7e6fcb9`). An independent verifier (fresh agent, read-only, no
  Experiment C outputs) ran three rounds: round 1 found 7 blocking and 10 substantive items (fixed in
  `ece5f9d`), round 2 one new substantive item (LiTS/BraTS reasoning) and one borderline F8 item
  (fixed in `e772e17`), round 3 none apart from author items (DOI placeholder; uncommitted Experiment
  C records; the claim boundary's stale status header).
- Final word counts (`prose_words`): main text 5,761 submitted, 5,374 revised (-7%); abstract 326 to
  316 words.
- Release dry run on the final tree: 1,694 files, single commit, `check_release_clean.py`: clean.
  The existing public repository fails the same check (HEAD `3104cb3` holds
  `manuscript/BDS-Lite.pdf` with dataset images; its history holds failure panels):
  `revision/RELEASE_GUIDE.md` decision D-a.
- 2026-11-08, second pass (author brief "fix the problems below"): revision history removed from the
  main text (correction in S4 and the letter); caveats stated in full once; two-result abstract;
  study and status labels defined once; ACDC selection noted in Methods; image-level markers removed
  from the main text; qualitative figure regenerated from the retrained set (decoder-only vs BDS-Lite,
  manifest v3); literals replaced by keys; DSC as 0-1 differences; US spelling; S10 cited; PeerJ
  figure files in revision/submission/figures/; [AUTHOR DECISION] markers for affiliation, preprint
  note and co-author approval. Commits 77660f3, 53d3610, d3d78c5. A fresh verifier (second agent)
  found 3 blocking and 8 substantive items, all fixed in d3d78c5.

- 2026-11-08, autonomous editorial review: positive ACDC direction separated from SESOI magnitude;
  literal registered Results wording retained; all five original figures and nine tables assessed;
  four figures and six compact/main tables retained, supporting evidence moved intact to S3/S7/S13/S19;
  Methods corrected to distinguish pooled residual from Welch variance; boundary-map loss support and
  post hoc feature-probe interpretation clarified. Current response float names and pages are derived
  from compiled labels. Independent raw-CSV calculation: 572 quantities, zero mismatches. Scope,
  validation, repair log and open author decisions: `EDITORIAL_REVIEW_20261108.md` and root final audits.

- 2026-11-09 Figure 2 review: selected Table 2 in main plus the complete vector forest in S13
  (Figure S6), rather than a three-pooled-row graphic duplicating the table. Dataset, ten-run and
  uncertainty results are unchanged. ACDC confirmation is now main Figure 2; qualitative panels
  are Figure 3. Response pointers and separate submission exports follow the compiled numbering.

- 2026-11-09 external-teacher presentation: move the redundant main contrast table to S18;
  retain the main vector graphic (now Figure 2) with cleaner aligned labels; separate Results
  into Cross-dataset exploration and Independent ACDC confirmation. Consolidate design and
  selection detail in Methods, preserving essential Results disclosures, all Holm/SESOI
  qualifications and the literal primary passage. No experiment or statistical analysis rerun.

- 2026-11-09 three-stage narrative: organize main Results and Table 1 as Ablation and
  Comparisons → Exploratory Knowledge Distillation → ACDC Confirmation. Embed the
  ISIC2018 ten-run extension within stage 1; retain all full statistics and frozen
  protocol identifiers in the supplement. Stage 2 keeps the 3-vs-10 ISIC reference.
  Methods explicitly distinguishes presentation stages from prospectively planned
  chronology; all selection, registration, correction and SESOI disclosures remain.

- 2026-11-09 Discussion follow-up: remove the redundant answer table and replace
  it with a concise direct answer in prose, retaining the exact positive ACDC
  interval, SESOI distinction, unresolved in-network effects and confirmed scoped
  ISIC2018 deficit. Synchronize S18 comparison wording and reviewer references.

- 2026-11-09 matched comparison: Table 3 separates seven directly evaluated models
  from published references; absolute scores reconstructed from 63 existing runs,
  with original-PDF receipts. Attention U-Net is included; earlier-pipeline and
  challenge-Jaccard rows remain in S7. No training or inference.
- 2026-11-09 later author instruction: promote balanced seed-1–3 KD sensitivity
  to the main display and recompute the three-dataset Holm family, explicitly
  post hoc. New outputs are isolated in kd_posthoc_balanced/; frozen 3-vs-10 and
  ACDC confirmation analyses are unchanged. The already-planned seed-matched
  contrast is distinguished from its post hoc display/Holm promotion.

- 2026-11-09 final comparison refinement: numerical best means only, unrounded
  ACDC DSC ranking, source HD labels with pinned evaluator receipts, 19 main rows
  and complete multiclass LB-UNet adaptation evidence in S7/Results. Static CPU
  graph/count/checkpoint-shape audit explains the nine-parameter head difference
  and possible partial-accounting origin of 0.038M; exact paper accounting remains
  unverified. No training, inference or frozen experimental change.

- 2026-11-09 reporting-scope follow-up: Table 3A uses the same six models
  (M0, M2, M6, GSL, C1, C3) on every dataset, totaling 18 rows. All adapted
  LB-UNet results remain in S7, with all three absolute summaries repeated at
  main-table precision; its published reference remains in B. Main Results
  acknowledges favorable ISIC2018 and unfavorable multiclass results. Scope
  is independent of performance; existing data and statistical analyses are
  unchanged. Unrounded means determine the nine primary-panel highlights.

- 2026-11-09 deployment-profile follow-up: Table 4 retains U-Net, deployed
  BDS-Lite, gate-collapsed BDS-Lite and Attention U-Net, with parameters, FLOPs
  and fp32 GPU batch-1/16 latency. CPU timings and the adapted LB-UNet row
  remain in the unchanged full S19 table, consistent with Table 3 scope.
  The caption and efficiency response explicitly retain gate latency overhead
  and direct readers to all supplementary measurements. No profiling rerun,
  experimental data change, selection or statistical re-analysis is involved.

  Complete LB-UNet source fields for ISIC2018 and Synapse are additionally
  rendered in S19 Table S55, including AMP and timing IQRs. The existing
  full-profile table and all source records remain byte-identical.
