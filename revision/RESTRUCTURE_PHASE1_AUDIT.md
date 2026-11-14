# Restructure revision: Phase 1 audit and plan (2026-11-09)

Branch `revision-restructure`, created from `revision/peerj-major` at `fa72602`. This phase is
read-only. No manuscript, figure, table, analysis or result file was changed. Training and
inference were not run. The author's uncommitted records (`revision/DEVIATIONS.md`,
`revision/queue_plan.yaml`, the two figure PDFs, and the word-count lines in `numbers.tex`)
came over to this branch unstaged and are left untouched.

Governing documents: `AGENTS.md`, `revision/CLAIM_BOUNDARY_v2.md` (approved, cited as "CB"
below), and `preregistration/acdc_confirmation.md` (frozen). The background regenerator
`analysis/revision/auto_refresh.py` is stopped (`revision/auto_refresh.log`, 2026-11-08T14:52Z),
so it cannot overwrite generated files.

---

## 1. Source inventory

### 1.1 Manuscript and supplement

| Item | Path | Notes |
|---|---|---|
| Main text | `revision/manuscript/main.tex` | 284 lines, class `wlpeerj.cls`, compiled with tectonic. Numbers come in through `\R{key}`, and tables through `\input{generated/...}`. |
| Supplement | `revision/manuscript/supplement.tex` | S1–S19 |
| Number keys | `revision/manuscript/generated/numbers{,_extra,_expansion,_audit,_kd}.tex` | Generated. 7,923 keys. Must never be hand-edited. |
| Build | `revision/build_revision.sh` → `analysis/revision/make_distillation_assets.py`, `revision/build_check.sh`, `analysis/revision/letter_locations.py`, `render_editor_letter.py` | Fails on unresolved keys or `??` |
| Response letter | `revision/response_to_reviewers.tex` | Anchors in `analysis/revision/letter_locations.py` must be updated whenever the text changes |
| Reviewer comments | `revision/reviewer.md` (untracked), `revision/peerj-reviewing-142937-v0.pdf` (untracked) | |
| Tracked changes | `analysis/revision/make_tracked_changes.py` | Baseline: `recovered/peerj-submission/main.tex` (the original submission). Uses pinned latexdiff 1.3.4 (not vendored). |
| Bibliography | `revision/manuscript/references_zotero.bib` | |

### 1.2 Figures

| Fig. | File | Generator / source | Editable at source? |
|---|---|---|---|
| 1 | `revision/figures/architecture.png` (3563×2475 px, 330 dpi; identical to `revision/manuscript/figure_architecture_v2.png`) | `analysis/revision/patch_architecture_figure.py` patches the authors' raster `revision/manuscript/figure_architecture.png` (two fixes: q_b arrow and b_g). **No vector or editable source is in the repo**, and no `.drawio`, `.pptx`, `.svg`, `.vsdx` or TikZ file for it exists under `/home/jc` (searched to depth 4). The PNG metadata (sRGB, gamma 0.45455, 330 dpi) does not identify the authoring tool. `scripts/generate_architecture_figure.py` draws an unrelated older matplotlib schematic. | **No** (raster only) |
| 2 | `revision/figures/contours_isic2018_retrained.pdf` | `analysis/revision/figure_qualitative_retrained.py`; manifest `revision/figures/qualitative_manifest_v3.json`; export `export_submission_figures.py` | Yes |
| 3 | `revision/figures/kd_confirmation.pdf` | `analysis/revision/make_distillation_assets.py`, reading `revision/acdc_confirm/results/per_seed.csv` and `results.json`, and `revision/kd_posthoc_balanced/contrasts.csv` | Yes |
| S-figs | `run_level`, `error_geometry`, `gradient_cosine`, `synthetic_targets`, `structure_errors`, `central_question`, `ten_seed`, `transfer_mechanism` | `make_narrative_assets.py`, `make_distillation_assets.py`, and others | Yes |

### 1.3 Tables (main text)

| Table | Content | Source |
|---|---|---|
| 1 `tab:studies` | Three stages | Hand-written LaTeX in `main.tex` with `\R{}` keys |
| 2 `tab:central` | Stage-1 pooled contrasts | `generated/tab_central_main.tex` ← `make_distillation_assets.py` ← `revision/tables/families/expansion_interaction.csv` |
| 3 `tab:lit` (A/B) | Matched means ± SD, seeds 1–3; published references | `generated/tab_lit_main.tex` ← `make_comparison_table.py` |
| 4 `tab:resource` | Deployment profile | `generated/tab_resource_main.tex` ← `make_distillation_assets.py` ← `revision/tables/resource_profile.csv` |

### 1.4 Results, analyses and checkpoints

| Evidence | Analysis code | Outputs | Per-case and checkpoints |
|---|---|---|---|
| Stage 1, matched ablation and pooling | `analysis/revision/expansion_statistics.py` (`experiment_b`, `components`) | `revision/tables/families/expansion_interaction.csv`, `expansion_components.csv` | `results/revision_runs/*_m{0..7}*`, `fs.per_case_path` |
| ISIC2018 ten-run extension | `expansion_statistics.py::experiment_a` | `revision/tables/families/expansion_contrasts.csv` | `isic_path()`; seeds 1–3 of U and BDS-Lite from `revision/official_isic/test/` |
| Stage 2, original frozen analysis | `analysis/kd3/unified_analysis.py` | `revision/kd_unified/` | `results/revision_runs/*_kd_*` |
| Stage 2, post hoc balanced display | `analysis/revision/balanced_kd_presentation.py` | `revision/kd_posthoc_balanced/` | same |
| Stage 3 | `scripts/acdc_confirm/analyze.py` (calls the frozen `expansion_statistics.contrast`) | `revision/acdc_confirm/results/{results.json,contrasts.csv,per_class.csv,per_seed.csv}` | Checkpoints `results/revision_runs/acdc_confirm_{decoder,kd_band}_seed{201..217}/best.pt` (34). **Official-test predictions and per-slice metrics already exist:** `results/acdc_confirm_eval/<run>/preds_official_test.npz` (1,076×224×224 uint8) and `per_case_official_test.csv`. |
| External teachers | `scripts/kd_mc/teacher.py`, `analysis/kd3/evaluate_test.py` | `revision/kd_unified/unified_results.json` → `results.<ds>.teacher_test` | `results/revision_runs/{isic2018,acdc,synapse}_kd_teacher_seed100/best.pt` (the ACDC sha256 starts `735747303c444e13`, matching `test_metrics.json`). The cached probabilities are in `*_kd_teacher_cache/train_teacher_prob_fp16.npy`. |
| Mechanism probes and gate | `boundary_transfer_probe.py`, `gradient_analysis.py`, `gate_structure.py` | keys `MECH.*`, `GR.*`, `GATE.*` | |

---

## 2. Statistical implementation audit (authoritative for the Methods formula)

**Code read.**
- `analysis/revision/expansion_statistics.py`, `contrast()` (lines 133–165), `linear_contrast()` (211–227), the pooling in `experiment_b()` (277–295), and `classify()` (119–130).
- `balanced_kd_presentation.py::contrast()` (51–75) reimplements the same estimator line for line.
- `scripts/acdc_confirm/analyze.py` and `analysis/kd3/unified_analysis.py` call `ex.contrast` directly.

### 2.1 Unit values
For run r of arm a and test unit u:
1. Take metric values per sample (image or slice) and per foreground class. Pairs where both
   prediction and reference are empty are non-finite and dropped.
2. Average over classes (class macro) per sample.
3. Average over the samples of the unit. Units are images on ISIC2018, patients on ACDC
   (`patient\d+`) and cases on Synapse (`case\d+`, `fs.cluster_of`).

This gives `y[a,r,u]`. For HD95 and ASSD the sign is flipped. `U` is the set of units that have
a defined value in **every** run of both arms.

### 2.2 Two-arm contrast (ISIC2018 ten-run extension, stages 2 and 3)

```
m[a,r]   = mean_u y[a,r,u]                      (run mean over common units)
Δ        = mean_r m[A,r] − mean_r m[B,r]
SE_run²  = s_A²/n_A + s_B²/n_B,   s_a² = sample variance (ddof 1) of the run means m[a,·]
d_u      = mean_r y[A,r,u] − mean_r y[B,r,u]
SE_unit² = Var_u(d_u)/|U|         (ddof 1)
SE       = sqrt(SE_run² + SE_unit²)
df       = (s_A²/n_A + s_B²/n_B)² / [ (s_A²/n_A)²/(n_A−1) + (s_B²/n_B)²/(n_B−1) ]   (Welch–Satterthwaite, run terms only)
CI       = Δ ± t_{0.975, df}·SE ;   p = 2·P(T_df > |Δ|/SE)
```

### 2.3 Stage-1 linear contrasts (2×2 effects and transfer)

```
estimate = Σ_c w_c · mean_r m[c,r]
SE_run²  = [ Σ_c Σ_r (m[c,r] − m̄_c)² / Σ_c (n_c − 1) ] · Σ_c w_c²/n_c   (pooled within-cell variance; only the cells in the contrast)
df       = Σ_c (n_c − 1)      (e.g. transfer M3 − M2 with 3 + 3 runs: df = 4)
d_u      = Σ_c w_c · mean_r y[c,r,u];  SE_unit² = Var_u(d_u)/|U|
SE, CI   as in 2.2 with t_{0.975, df}
```

**Pooling across datasets.**
- Weights: `w_d = 1/SE_d²`.
- Pooled estimate: `Σ w_d est_d / Σ w_d`, with `SE = sqrt(1/Σ w_d)` and a **normal** interval
  (±1.96).
- `Q = Σ w_d (est_d − pooled)²`, with `p = χ²₂` upper tail.

The additive component model (`components()`) is a separate OLS over runs. It is not used in the
new formula box.

### 2.4 Does the combination double-count uncertainty? (finding: almost, but not exactly)

**Model.** Take the two-way random model

`y[a,r,u] = μ_a + ρ_{ar} + υ_u + γ_{au} + ε_{aru}`

with run, unit, arm×unit and residual run×unit components. The test units are shared by both
arms.

- **SE_run.** `s_a²` estimates `σ²_ρ + σ²_ε/|U|`. The unit main effect `υ_u` and `γ_{au}` are
  constant within an arm, so they cancel.
- **SE_unit.** `Var_u(d_u)` estimates `2σ²_γ + σ²_ε(1/n_A + 1/n_B)`. The run effects `ρ` are
  constant across units, so they cancel. `υ_u` cancels because the units are common.
- **Target.** The variance of Δ over new runs *and* new test units is
  `σ²_ρ(1/n_A+1/n_B) + 2σ²_γ/|U| + σ²_ε(1/n_A+1/n_B)/|U|`.

**Result.** The run main effect enters only SE_run, the arm×unit effect enters only SE_unit, and
the unit main effect enters neither. **The one exception is the run×unit residual term
`σ²_ε(1/n_A+1/n_B)/|U|`, which enters both components.**

So SE² exceeds the target by exactly that term, and the interval is slightly conservative.
Requested item 2.2 asks the manuscript to say that the procedure "does not double-count". The
accurate statement is:
- the two components capture different sources;
- they share only the run-by-unit residual term, which is divided by the number of test units;
- that sharing makes the interval slightly conservative.

The size of this term has not been computed. It could be estimated descriptively from the
existing per-case files (labelled post hoc). Without that, the manuscript should state the
direction (conservative) and not quantify it.

**Degrees of freedom.** In both the Welch and the pooled versions, df counts only the run
components; SE_unit gets no df correction. This is the implementation, and the formula box must
say so.

### 2.5 Interpretation labels in code
- `classify()` returns `relevant gain`, `relevant harm`, `no relevant effect`, `gain, size
  unclear`, `harm, size unclear` and `undetermined`. The manuscript tables print the size-unclear
  readings as "size unclear".
- `status()` returns `confirmed`, `supported by all seeds, not confirmed by fresh seeds` and
  `not established`.
- Stage 3 has a third, pre-registered vocabulary: `success` / `inconclusive` / `failure`, with
  the qualifier "of unclear size" (pre-registration §7, §11). The unified label box must contain
  all three sets.

---

## 3. Reference numbers in the task prompt versus the repository

| Prompt reference | Repository value (key → source) | Status |
|---|---|---|
| Pooled transfer BF1 +0.0092 [−0.0010, +0.0194] | `A1.nosurf.boundary_f1.*` ← `expansion_interaction.csv` | Matches |
| Stage 3 BF1 +0.0178 [+0.0098, +0.0258], 17 seeds, 50 patients | `CF.*`, `OT.patients` ← `acdc_confirm/results/results.json` | Matches |
| Stage 3 DSC ≈ +0.0136 | `CF.dsc.delta` +0.0136 [+0.0061, +0.0211] (secondary, descriptive) | Matches. F16 applies: never lead with it. |
| "≈70% of erroneous ACDC pixels within 2 px of a contour" | `ERR.acdc.M0.near` = `S0` = 70%; decoder-only (M2) 68%; D+KD-band not measured | **Scope differs.** These are stage-1 models (seeds 1–3) on the **internal 20-patient** test, measured from the *reference* boundary. Nothing was measured on patients 101–150 or for D+KD-band. The wording must carry this scope. |
| Teacher "≈11× larger" | `KDC.acdc.ratio` = 11 = 24,436,804 / 2,197,493 | Matches only against the **training-time** student, which includes D_b. The deployed student is ≈1.93 M, which gives ≈12.7×. Keep the phrase "training-time size". |
| Gate changes ≤0.11% of labels; latency +8–15% | `GATE.labels.runmax` 0.11; `RESX.BDSLitedeployed.gpu_fp32_b1…pct` +8, `…b16…pct` +15 | Matches. GPU fp32, ISIC2018 input. |
| "Between-run variation exceeded that threshold" (abstract draft) | ISIC2018 ten-run SDs 0.0186–0.0257 (`XM.*.boundary_f1.sd`); **ACDC stage 3 SDs 0.0069 and 0.0083** (`CF.kd.sd`, `CF.dec.sd`) | **False as an unscoped statement.** It holds on ISIC2018 only. |
| "Probes indicated that the distillation loss was met by the projection" (abstract draft) | `MECH.*` show increased head AP on q_s and no F_s readout difference. Gradient cosines are 0.00–0.03 (seed-1 probes only). | **Stronger than the evidence.** The current text says "consistent with the hypothesis". Recommend "were consistent with". |
| GSL > D at seeds 1–3, < D at ten runs | Seeds 1–3, ISIC2018: GSL 0.3838 vs D 0.3794 (`tab_lit_main`). Ten runs: GSL − D `X.gsl.decoder.boundary_f1.all.*`, confirmed. | Matches |
| D+KD-band positive on all three datasets, none survives Holm | Balanced: ISIC +0.0523 (Holm 0.293), ACDC +0.0239 (0.088), Synapse +0.0150 (0.500). Original: ISIC +0.0376 (0.496). | Matches in both families |
| Four ISIC2018 arms have 10 runs | U, D, BDS-Lite and GSL: 10 each. U-Net+Boundary Loss and Attention U-Net: 3 each. | Matches |
| FitNets passage, Methods "≈ lines 133–136" | Source line 122 ("How boundary information can reach the prediction"). PDF line numbers to be confirmed after the build. | Location note only |
| "+0.0178 … appears four times" | Abstract, Stage 3 Results and Discussion opening carry the number; the Conclusion restates it without the number | Matches (3 + 1) |
| Raj et al. cited more than once | `Raj2026ViTSwinUNet` in Introduction (l. 62) and Datasets (l. 140) | Matches |
| External teacher's ACDC test metrics, "same protocol as Stage 3" | **Not in S18.** Internal 20-patient test only: BF1 0.8859, DSC 0.8231 (`kd_unified/unified_results.json` → `results.acdc.teacher_test`, same `unit_values` estimator, frozen stage-2 evaluation). **Not evaluated on patients 101–150.** | Partially available; see §4 |

---

## 4. Availability, by requested change

**Legend.** **Ready**: can be done from existing files. **Needs approval**: needs inference, or
an author or claim-boundary decision. **Blocked**: missing data or source files.

| # | Change | Status | Notes |
|---|---|---|---|
| 2.1 | Discussion: conditions, mechanism, candidate thesis | **Needs approval** | The numbers exist. The thesis conflicts with CB F3 ("teacher strength or any teacher property explains when KD helps" is forbidden) and touches F15 (no claim *between* A1 and A3). It needs an author ruling; see §6, D1. |
| 2.2a | Teacher ACDC metrics | **Ready** (internal set) / **Needs approval** (official 101–150) | Inference plan in §5 |
| 2.2b | BPKD sentence | Ready | Bib entry `Liu2024BPKD` exists. Check the paper's band/body description before writing. |
| 2.2c | Define "retrained set" / "earlier pipeline" | Ready | Sources: S4, S15, `OFFICIAL_ISIC_FINDINGS.md`. 5 occurrences in main, 10 in the supplement. |
| 2.2d | L_sdf vs Kervadec vs GSL | Ready | `src/bds_lite/training/losses.py:110–161`; `scripts/comparators/components.py` (`kervadec_weights` schedule) |
| 2.2e | Eq. 3 explanation (interior reward) | Ready | φ < 0 inside (`losses.py` docstring) |
| 2.2f | Recipe-level SE formula | Ready | §2 above. The "does not double-count" wording must be corrected as in §2.4. |
| 2.2g | Unified label box | Ready | Must also include the stage-3 success/inconclusive/failure set |
| 2.2h | Cochran's Q note | Ready | |
| 2.2i | Limitations in four groups | Ready | |
| 2.3 | Methods and Results reorder, arm names | Ready | Unnamed cells need names: M1 (U+S), M4 (D+S+gate), M5 (D+S+distill). |
| 2.4 | Move LiTS/BraTS, Swin, LB-UNet, OTS, Exp. C | Ready, with care | **The LiTS/BraTS justification is not yet in the supplement** (0 hits); it must be added to S3 before removal. Swin UNETR is in S7. LB-UNet is in S7. OTS is in S18. Experiment C is in S15. |
| 2.4 | "of unclear size" → "size unclear" | **Needs approval** | CB A3 and author decision 4 require the literal §11 sentence, which contains "of unclear size". Proposal: keep the one verbatim pre-registered sentence in the Stage 3 Results and use the "size unclear" label everywhere else, with the mapping stated in the label box. |
| 2.4 | References: Swin-Unet ECCV-W 2023; Celaya GSL | Ready / check | `Cao2022SwinUnet` (inproceedings) and `Cao2021SwinUnetArxiv` both exist. For Celaya, a web search found only arXiv 2302.03868 plus an NSF PAR copy (par.nsf.gov purl 10635864) that suggests a journal version. The venue must be verified before replacing the reference. |
| 2.5 | Abstract | Ready, with two corrections | Scope "between-run variation" to ISIC2018, and soften "probes indicated". Every number is otherwise verified (§3). |
| 3.0 | Style module, REFERENCES.md, PeerJ figure rules | Ready | PeerJ CS figure requirements: the author-instructions page was not read yet (the search returned only the page URL). It must be fetched in Phase 3. |
| 3.1 | Fig. 1 notation + D+KD-band panel | **Blocked at source** | Raster only. Deliverables: (i) an exact manual-edit checklist; (ii) a separate vector D+KD-band panel in the sampled palette. Alternative: a full vector redraw (author choice; §6, D3). |
| 3.2 | Arms × components table | Ready | Correction to the brief's "Deployed graph" idea: D, D+S, D+KD-band and stage-3 D *contain* P and a gate at α = 0 in the checkpoint. That gate is exactly the identity, so the deployed function equals U-Net, and the graph equals U-Net once the inert layers are dropped. The table should say this. |
| 3.3 | New Fig. 2 (ACDC cases) | **Blocked (permission)** | `revision/LICENSING_AUDIT.md`: the ACDC site states no licence or republication terms, and no ACDC labels have been shown so far. Mirrors (Hugging Face) report CC BY-NC-SA 4.0, citing a humanheart-project licence page that was not opened. **Even if that is correct, NC-SA material is not compatible with PeerJ's CC BY article licence.** Contour/FP/FN maps would reproduce the reference annotations. Per the brief, this figure is stopped. The data for the alternative exist without inference (per-slice CSVs and predictions for all 34 runs). |
| 3.3 | Move old Fig. 2 to the supplement | Ready | |
| 3.4 | Expanded Fig. 3 (forest plot) | Ready | Every row is read from CSV/JSON: `expansion_interaction.csv` (pooled transfer; **normal** CI); `expansion_contrasts.csv` (ISIC2018 full vs D, all-seed t-CI); `kd_posthoc_balanced/contrasts.csv` and/or `kd_unified/contrasts.csv`; `acdc_confirm/results/results.json`. The marker class for each row needs an author mapping (§6, D4). |
| 3.5 | Table 1 status column and rows | Ready, after D4 | |
| 3.5 | Table 2 Reading column and header | Ready | `classification` column in `expansion_interaction.csv` |
| 3.5 | Table 3A with ten runs and n column | Ready | Per-case files for seeds 1–10 exist for the four arms (`isic_path`). `make_comparison_table.py` must be extended, keeping its per-run reconciliation check. |
| 3.5 | Text near Table 3B: U-Net volume DSC from S9 | Partially | S9 gives absolute 3D DSC for the **original-pipeline and follow-up** U-Net (`VOL.acdc.S0.dsc` 85.9%, `VOL.acdc.R0/M0` 84.2%; Synapse `S0` 70.2%, `R0/M0` 68.3%). There is no 3D value for the retrained-set seeds 1–3 models in Table 3A. **Aggregation** (2D slice vs 3D volume) can be quantified for the same models. **Training protocol** (pretraining, augmentation, 18 vs 14 Synapse cases, resolution) cannot be separated, because no matched runs exist. State that limit. |
| 3.5 | Table 4 + D+KD-band cost | Ready, with one gap | Teacher 0.243 h; students 7.63 vs 7.52 h. **Cache-generation time was never recorded**, so it gets `TODO(missing: …)` or an explicit "not recorded" (CB D8 already states it as not recorded). |
| 4.2 | latexdiff | **Needs approval** | Which baseline? See §6, D5. |
| Opt. | D+KD-full materials | Ready (prepare only) | Templates: `configs/revision/acdc_confirm/*`, `scripts/kd_mc/loss.py` (band mask). Must not touch `scripts/kd_mc/` or frozen files; new files only. |

---

## 5. Proposed read-only inference (none run; each needs explicit approval)

**I1. External teacher on the official ACDC test patients (for 2.2a).**
- Checkpoint: `results/revision_runs/acdc_kd_teacher_seed100/best.pt` (sha256 prefix
  `735747303c444e13`).
- Config: the same folder's `config.json`.
- Model: `scripts.kd_mc.teacher.build_teacher(config, pretrained=False)`, loaded with
  `load_state_dict(strict=True)`. This is the loader of `analysis/kd3/evaluate_test.py`.
- Data: `data/processed/acdc_official_test`, hash-checked against the manifest the way
  `scripts/acdc_confirm/evaluate_test.py` checks it (1,076 slices, 50 patients).
- Evaluation: the unchanged `scripts/revision/train_run.py::evaluate`, with the same loader
  settings as stage 3.
- Environment: `/home/jc/bds-followup/.venv-kd/bin/python` (smp 0.6.0).
- Output: a **new** folder `results/acdc_confirm_eval_posthoc/acdc_kd_teacher_seed100/`.
  Nothing under `results/acdc_confirm_eval/` or `revision/acdc_confirm/` is touched.
- Aggregation: `expansion_statistics.unit_values` → run mean, the same estimator as the stage-3
  arm means.
- Label: post hoc, descriptive. It is a single model, so it has no interval over runs.
- This reuses the held-out set after the confirmation, which leaves the frozen decision
  unaffected.

**Not needed.** The new Fig. 2 (if permitted) and the per-case alternative figure need no
inference. The predictions and per-slice metrics already exist for all 34 stage-3 runs.

---

## 6. Decisions required from the author before Phase 2

- **D1. Thesis vs claim boundary.** The candidate thesis is a teacher-property explanation (CB F3
  forbids it outright). It also implicitly contrasts A1 and A3 (F15). Options:
  - (a) The author approves a dated addendum to `revision/CLAIM_BOUNDARY_v2.md`. It would allow
    this sentence **only** as a labelled post hoc hypothesis in the Discussion, never in the
    Abstract or Conclusions, with the explicit statement that no head-to-head test was made.
  - (b) Keep the thesis out of the text, and only list the untested explanations.

  Recommendation: (a). It matches the brief's "post hoc hypothesis" framing, and the claim
  boundary file is the author's to amend.
- **D2. "of unclear size".** Keep the literal pre-registered sentence once (CB A3), and use
  "size unclear" elsewhere? Recommended.
- **D3. Fig. 1.**
  - (a) The author has the original diagram file (PowerPoint, draw.io or similar) and applies
    the checklist.
  - (b) A full vector redraw in matplotlib, matching the sampled palette.
  - (c) Raster checklist + a separate vector D+KD-band panel only (the brief's fallback).
- **D4. Evidence status per study** (Table 1 and Fig. 3 markers). Proposed mapping:
  - Stage-1 ablation: "Pre-registered (analysis plan; informed by earlier exploratory runs)" or
    "Exploratory". Which one?
  - ISIC2018 ten-run extension: Pre-registered, with a fresh-seed confirmation rule.
  - Stage 2 original analysis: Exploratory (protocol frozen).
  - Stage 2 balanced 3 vs 3: Post hoc display.
  - Stage 3: Pre-registered (confirmatory).
  - Experiment C: Stopped, unused.
  - Historical re-evaluation: Post hoc.

  For Fig. 3(b), Stage 2 rows: balanced only, or both families?
- **D5. latexdiff baseline.**
  - (a) The original submission `recovered/peerj-submission/main.tex`, which is what reviewers
    saw and what the existing `tracked_changes.pdf` uses.
  - (b) This branch's starting point `fa72602`.

  Recommendation: (a) for submission, plus (b) as an internal review aid.
- **D6. ACDC Fig. 2.** Confirm that the figure is stopped. Approve the alternative: a
  per-patient statistical plot of the seed-averaged D+KD-band − D Boundary F1 difference on the
  50 official patients, by structure (RV, Myo, LV), with no image content, labelled post hoc.
- **D7. Inference I1.** Approve or decline.
- **D8. Item 2.2f wording.** Accept the corrected "does not double-count" statement (§2.4). The
  alternative is an extra post hoc computation of the size of the shared term.
- **D9. Names for M1/M4/M5.** Proposed: U+S, D+S+G (BDS-Lite without distillation), D+S+T_d
  (BDS-Lite without gate). These would be used in the supplement only.
- **D10. PLAN.md.** This task has no phase in `PLAN.md`. Adding a new phase section would change
  the plan's structure, which AGENTS.md reserves to the author. The status is therefore recorded
  here and in the CHANGELOG. Authorize a "Phase 19" block, or keep it this way?

---

## 7. Implementation plan (Phases 2–4), subject to the decisions above

General rules for every phase:
- Every new number is added as a generated key. New script functions go in a new
  `analysis/revision/make_restructure_assets.py`, which reads only existing CSV/JSON.
- Each number is logged in the CHANGELOG with its script and source path.
- One commit per phase (`paper(phase-R2)`, `paper(phase-R3)`, `docs(phase-R4)`), after
  `ruff format --check`, `ruff check` and `pytest`.
- Material is moved into the supplement before it is deleted from the main text, and the
  preservation is checked with `grep` before and after.

**Phase 2 (text).**
1. Add to the supplement first:
   - S3: the LiTS/BraTS paragraph, verbatim;
   - S7: the Swin-Unet/Swin UNETR sentence and the extended LB-UNet discussion;
   - S15: the Experiment C procedure (check against what is already there);
   - S18: OTS (already present; check completeness).
2. Restructure Methods in the 8-part order. Add:
   - the arms table (definitions: U, D, D+S, D+T, D+T+S = BDS-Lite, D+KD-band);
   - definitions of the retrained set and the earlier pipeline;
   - the L_sdf / Kervadec / GSL distinction;
   - the Eq. 3 correction;
   - the SE formula box (§2);
   - the unified label box (three vocabularies);
   - the Cochran's Q caveat;
   - the BPKD sentences (Related work and Methods).
3. Restructure Results into RQ1–RQ6, with run-to-run variation first and the costs merged in
   RQ6. Remove repeated numbers from the Discussion.
4. Discussion opening: the conditions-based answer and the post hoc mechanism hypothesis, worded
   per D1 and with the scope of §3 (internal-test 70%; training-time 11×). Then the Limitations
   in four groups, including no D+KD-full control and the confounding of pretraining, capacity
   and band.
5. Introduction: trim, and merge contributions 1 and 3. Then the abstract, from the brief's
   draft with the two corrections.
6. Number formatting pass (4 decimals; 2 decimals in px), at the generator level and not by hand.
7. Write `revision/author_decisions.md` with alternative wordings for the three
   `\authordecision` markers. The markers themselves are left in place.
8. References: Swin-Unet to ECCV-W 2023; Celaya only if a journal version is verified.

**Phase 3 (figures and tables).**
1. Write `revision/figures/REFERENCES.md`, after consulting SciencePlots (MIT) and forestplot
   (MIT) for design ideas only, without copying code.
2. Fetch the PeerJ CS figure rules.
3. Sample the Fig. 1 palette, then write `revision/figures/style.py`.
4. Fig. 1 per D3.
5. New Fig. 2 per D6 (no ACDC image content unless permission is shown).
6. Move the ISIC2018 contour figure to the supplement.
7. Fig. 3: panel (a) unchanged in content, and panel (b) as the forest plot, all read from files.
8. Tables 1–4 per the brief.
9. Render every figure to PNG, inspect it, and build a contact sheet.

**Phase 4 (verification).**
1. `bash revision/build_revision.sh`, with the `letter_locations.py` anchors updated.
2. Page-by-page PDF inspection, and a check of S1–S19 numbering and cross-references.
3. latexdiff per D5.
4. Scan for `AUTHOR DECISION` (3 expected), `TODO`, `decoder alone`, `retrained set`.
5. Numerical consistency audit with `verify_editorial_numbers.py`, extended for the new keys.
6. Final report, response-letter outline, and the required
   `results/final_manuscript_audit_v2/` outputs.
7. D+KD-full preparation (config draft, post hoc statement, runtime estimate from the stage-3
   `training_epoch_hours`). Nothing is run.
