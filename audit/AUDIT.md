# BDS-Lite Public Release — Stage 1 Audit (RE-ANCHORED to the submission candidate)

**Date:** 2026-06-17
**Source of truth:** `results/submission_candidate_manuscript_v1/` (Phase 17). The
canonical text is `manuscript/BDS_Lite_submission_candidate_v1.md`; tables are
`tables/table1..6_*.csv|md`; figures are Figures 1–6.
**Supersedes:** my earlier audit of `manuscript_v2/main.tex`. That document is **not**
the submission; its certification did not transfer. This file replaces it.
**Gate (Section B) status:** ✅ **CLEARED** — every divergent load-bearing number
(cluster statistics, 24 failure panels) traces to a producing script **and** a
machine-readable output, confirmed **verbatim**. No fabricated statistic; nothing was
regenerated.

Read-only respected: source tree untouched; this is the only file written.

---

## A. The submission candidate and its evidence inventory

**Title:** *When Does Boundary Distillation Help Lightweight Medical Image
Segmentation? A Controlled Study of Applicability, Failure Modes, and Inference
Decoupling.*

**Experimental comparators (IN scope):** U-Net, BDS-Lite (full), **U-Net+GSL**
(Phase-16 matched three-seed rerun). The paper's framing is conditional/mixed with an
explicit *no-superiority* stance versus GSL.

**Tables (source-of-record):**

| Table | What | Backing artifact | Producer |
|---|---|---|---|
| 1 | Dataset/protocol | `data/splits/{isic2018,acdc,synapse}/…` | (manifests) |
| 2 | Matched 3-seed main results (U-Net/BDS-Lite/U-Net+GSL) | U-Net+BDS: `results/final_manuscript_audit_v2/recomputed_main_test_means.csv`; GSL: `results/submission_blocker_resolution_v1/gsl_memory_bounded/matched_gsl_summary.csv` | `scripts` + `aggregate_phase16_gsl.py` |
| 3 | **Cluster-aware statistics** | `results/midterm_q2_rescue/analysis/cluster_level_statistics.csv` | `results/midterm_q2_rescue/scripts/generate_rescue_analysis.py` |
| 4 | Training/inference cost | `results/submission_blocker_resolution_v1/profiling/training_cost_combined_summary.csv` | `…/scripts/profile_*` |
| 5 | Failure & limitation summary | `cluster_level_statistics.csv` + `failure_case_manifest.csv` + `recomputed_main_test_means.csv` | `generate_rescue_analysis.py` |
| 6 | Final claim-evidence | (all of the above) | — |

**Figures (all present):** F1 architecture, F2 matched main results (journal version
`submission_blocker_resolution_v1/figures/figure2_matched_main_results.{pdf,png}`,
copied into the candidate package), F3 per-case delta distributions, F4
boundary–distance relationships, F5 subgroup/organ heatmap, **F6 failure cases
(24 panels)**. F3–F6 derive from `generate_rescue_analysis.py`; F6 composites the
24 deterministically selected panels (individual PNGs also on disk under
`results/midterm_q2_rescue/figures/failure_cases/`, count = 24).
*Caveat:* `submission_critical_evidence_v1/figures/figure2_main_results_separated_families.*`
is the **superseded** Phase-15 Figure 2 — must not be shipped as the main figure.

---

## B. GATE — verification of the claims that diverge from manuscript_v2

These were never checked before. For each, I located the **producing script** and the
**machine-readable output**, and confirmed the manuscript value **verbatim** (raw value
→ displayed value). I did **not** re-run anything.

### B.1 Cluster-aware statistics — ✅ all methods implemented, values verbatim

- **Producer:** `results/midterm_q2_rescue/scripts/generate_rescue_analysis.py`,
  function `cluster_statistics()` (lines 476–525).
- **Output:** `results/midterm_q2_rescue/analysis/cluster_level_statistics.csv`
  (16 rows: 3 datasets × 5 metrics + header).
- **Methods present in the code** (one column each in the output):
  - Cluster bootstrap 95% CI → `bootstrap_mean_ci(values, iterations=5000)` (percentile
    2.5/97.5) → `bootstrap_ci_low/high`.
  - Paired signed-rank → `scipy.stats.wilcoxon(values,…)` → `wilcoxon_statistic`,
    `p_value_two_sided`; Holm via `holm(...)` → `p_value_holm`, `decision_alpha_0_05`.
  - Sign-based effect size → `(n+ − n−)/n_nonzero` → `sign_effect_size`.
  - Leave-one-cluster-out → mean after `np.delete` each cluster →
    `leave_one_cluster_out_mean_min/max`.
  - Cluster unit = image (ISIC), patient (ACDC `^(patient\d+)`), case (Synapse
    `^(case\d+)`), oriented so positive = favorable.

| Manuscript value | Raw value in `cluster_level_statistics.csv` | ✓ |
|---|---|---|
| ISIC HD95 oriented delta **1.0409 px** | `1.0408505574877533` | ✅ |
| ISIC HD95 95% CI **0.2476 – 1.8092** | `0.24764992655053736` / `1.8091775235862282` | ✅ |
| ISIC Boundary F1 delta **0.0077** | `0.007690171895420696` | ✅ |
| ISIC Boundary F1 CI **0.0006 – 0.0146** | `0.0005707902178147403` / `0.01463469670680593` | ✅ |
| ISIC HD95/BF1/DSC Holm p **0.0029**; ASSD **0.0001** | `0.002918…`, `0.002918…`, `0.002918…`, `9.82e-05` | ✅ |
| ACDC n=**20** patients, Synapse n=**12** cases | `n_clusters` 20 / 12 | ✅ |
| ACDC/Synapse all Holm p **1.0**, not significant | every `p_value_holm` = 1.0 | ✅ |
| ISIC DSC −0.0003, IoU 0.0021, ASSD 0.2207 | `-0.000264`, `0.002126`, `0.220689` | ✅ |

**Reproducibility caveat (not a defect):** the bootstrap uses a module-level seeded RNG
(`np.random.default_rng(20260606)`) consumed earlier by `subgroup_rows()` in the same
run; CI endpoints are deterministic only when the full script runs in its fixed order.
The committed CSV is the artifact of record. (Per your instruction I did **not**
regenerate.)

### B.2 24-panel failure set — ✅ deterministic selector + manifest + figure, verbatim

- **Producer:** `generate_rescue_analysis.py`, `selection_rows()` (815–872, deterministic
  metric criteria) + `plot_failure_cases()` (875–911).
- **Output:** `results/midterm_q2_rescue/analysis/failure_case_manifest.csv` —
  **exactly 24 data rows**; 24 individual panels on disk; composited into
  `results/submission_critical_evidence_v1/figures/figure6_failure_cases.{pdf,png}`.
- **Panel budget = 24, deterministic:** per dataset `clearly_worse(2) + clearly_better(2)
  + similar_dice_boundary(2)` = 6; ACDC adds `typical_acdc_negative(3)`; Synapse adds
  `distance_metric_failure(3)` → **6 + 9 + 9 = 24** (matches manifest category counts).
- **Named cases (manuscript §5.5) confirmed verbatim:**

| Manuscript | Manifest row | ✓ |
|---|---|---|
| `ISIC_0016060`: DSC −0.223, BF1 −0.218 | `delta_dsc=-0.22343815…`, `delta_boundary_f1=-0.21809296…` | ✅ |
| `ISIC_0013140`: favorable | `isic2018,clearly_better,1,ISIC_0013140` | ✅ |
| `patient052_frame01_slice_5`: +DSC, worse HD95/ASSD | `delta_dsc=+0.076…`, `delta_hd95=+26.1…`, `delta_assd=+17.3…` | ✅ |
| `case0004_slice134`: Synapse distance failure | present (clearly_worse + distance_metric_failure) | ✅ |

**Verdict: no STOP-the-line finding.** Both divergent claim families are fully
script-backed and output-backed, and every checked number is verbatim.

### B.3 Other candidate-specific numbers (also verbatim)

| Manuscript claim | Source CSV | ✓ |
|---|---|---|
| ISIC favorable fractions 56.1/56.8/56.9/57.8/56.3% | `per_case_distribution_summary.csv` (0.5607/0.5684/0.5687/0.5783/0.5626) | ✅ |
| Synapse favorable 53.1% DSC / 44.8% HD95 / 48.2% ASSD | `…` (0.5313/0.4476/0.4817) | ✅ |
| ISIC HD95 trimmed 0.9505 vs 1.0409 | `outlier_sensitivity.csv` (`0.9504646…`/`1.0408505…`) | ✅ |
| Synapse HD95 −0.1003 → −0.2106 (5% trim) | `outlier_sensitivity.csv` (`-0.10028…`/`-0.21062…`) | ✅ |
| ISIC 12.4% similar-DSC/BF1-gain; Synapse ~21% DSC+/dist− | `tradeoff_pattern_counts.csv` (0.12379; 0.2122/0.2098) | ✅ |
| Gate-removal DSC 0.8607→0.8610 / 0.7847→0.7851 / 0.8466→0.8449 | `gate_removal_summary.csv` | ✅ |
| GSL matched means (ISIC DSC 0.8662, HD95 19.7954, …) | `matched_gsl_summary.csv` | ✅ |
| Cost: U-Net 39.62 / BDS 57.17 / GSL **52.32** ms-step | `training_cost_combined_summary.csv` (`52.3193…`) | ✅ |
| Deploy +1,072 params (1,928,114 vs 1,927,042); 269,889 branch | reproduced from code (Stage 1) | ✅ |

---

## C. manuscript_v2 vs submission_candidate_v1 — relationship & diff

Both share the same code, `data/splits/`, metrics, and the U-Net/BDS-Lite ISIC/ACDC/
Synapse three-seed means. They are **different papers** in framing, statistics, baselines,
and figures.

| Aspect | `manuscript_v2/main.tex` (NOT submitted) | `submission_candidate_v1` (**submitted**) |
|---|---|---|
| Thesis | Bounded recipe + per-image effect | "**When does it help**" — applicability/failure-mode study |
| 3rd comparator | **EGE-UNet** (trained, in all tables) | **U-Net+GSL** (Phase-16 matched, in all tables) |
| GSL | cited; "competitive/stronger" from old separate-family artifacts | **experimental matched 3-seed comparator**; mixed both directions, no-superiority |
| EGE-UNet | experimental | **citation-only** (ref [21], Related Work) → **OUT of experimental scope** |
| MedSAM | absent | absent → OUT |
| Boundary DoU | cited (code present) | citation-only (ref [15]) → not in results |
| Inferential stats | per-image multi-seed Wilcoxon+Holm (EGE incl.) | **cluster-aware**: bootstrap CI + Holm signed-rank + sign effect + LOCO (`generate_rescue_analysis.py`) |
| Failure figure | 12-panel qualitative | **24-panel deterministic failure set** (Fig 6) |
| Figures | 4 (arch, per-seed, qualitative, gate) | 6 (arch, matched main, per-case dist, boundary–distance, subgroup heatmap, failure) |
| Resource source | `resource_profile_isic2018_comparison.csv` (deploy graph, 1.78/1.75 ms) | `training_cost_combined_summary.csv` (full-graph profile, 1.93/2.40/1.97 ms) + deploy params |

**Comparator scope decision (from what the candidate *uses*, not manuscript_v2):**
- **IN:** U-Net, BDS-Lite full, U-Net+GSL (matched). Ship their code, configs/snapshots,
  splits, per-case CSVs/aggregates; checkpoints + prediction arrays → Zenodo.
- **OUT (experimental artifacts):** EGE-UNet (`configs/experiments/*_ege_unet_*`,
  `outputs/runs/*_ege_unet_*`, `outputs/checkpoints/*ege*`, `outputs/evaluations/*ege*`,
  the EGE rows of `recomputed_main_test_means.csv`, `per_image_multiseed/*`), MedSAM
  foundation checkpoints, MobileViT/Attention-UNet runs. `src/.../ege_unet.py`,
  `boundary_dou.py` may remain as cited-but-unused code or be dropped — your call (§G).

---

## 1. Coverage table — submission-candidate claims → artifacts

Status: **Present** / **Partial** / **Missing**.

### 1.1 Source code (unchanged from Stage 1; all still back the candidate's Method §3)
| Component | Status | Path |
|---|---|---|
| U-Net backbone (1,927,042 params, reproduced) | Present | `src/bds_lite/models/unet.py` |
| Boundary decoder h_φ; bounded gate α=0.25 `q_s{1+α(g−0.5)}` | Present | `src/bds_lite/models/bds_lite.py` |
| Boundary BCE+binary Dice; Smooth-L1 vs `stopgrad`; signed-distance surface | Present | `src/bds_lite/training/losses.py` |
| **GSL comparator (IN scope)** | Present | `src/bds_lite/losses/gsl.py` (port; upstream commit recorded) |
| Corrected Boundary F1 (0 for disjoint non-empty; 2-px tol) | Present | `src/bds_lite/evaluation/metrics.py:77-99` |
| DSC/IoU/HD95/ASSD | Present | `src/bds_lite/evaluation/metrics.py` |
| Data pipeline + converters; training loop; evaluation | Present | `src/bds_lite/{data,training,evaluation}/`; `scripts/{convert_*,train,evaluate}.py` |

### 1.2 Locked configuration
| Setting | Status | Evidence |
|---|---|---|
| λ_ce=λ_dice=λ_s=1, λ_b=λ_d=0.05, α=0.25, 224², 150 ep, max-val-DSC, seed 2026 | Present | `configs/base.yaml`; per-run `outputs/runs/*_{unet,bds_lite_full}_seed*/config_resolved.yaml` |
| **U-Net+GSL matched configs (committed!)** | Present | `results/submission_blocker_resolution_v1/gsl_memory_bounded/configs/phase16_matched_gsl_{isic2018,acdc,synapse}_seed{1,2,3}.yaml` (150 ep, seed-2026 split, `use_gsl:true`, `gsl_alpha_schedule:step5`) + run `config.json` snapshots |
| Top-level U-Net/BDS-Lite experiment YAMLs | **Partial** | `*_bds_lite_full.yaml`/`*_unet.yaml`/`configs/models/bds_lite.yaml` not committed; recipe recoverable from per-run `config_resolved.yaml` (α=0.25 etc. confirmed) |

### 1.3 Splits (authoritative = `data/splits/`)
| Dataset | Counts | Seed | Disjoint | Status |
|---|---|---|---|---|
| ISIC2018 | 2075/519 (no test) | 2026 | image-disjoint ✓ | Present |
| ACDC | 1350/186/366 (70/10/20 pt) | 2026 | patient-disjoint ✓ | Present |
| Synapse | 1738/473/1568 (14/4/12 case) | 2026 | case-disjoint ✓ | Present |
- `sha256` stamped; `tests/unit/test_split_integrity.py` enforces. GSL rerun uses the
  same seed-2026 manifest (`…/gsl_memory_bounded/splits/isic2018_locked_seed2026.json`).
- **Stale (delete/quarantine):** top-level `splits/isic2018_split.json` (1815/389/390,
  seed 1); the 389-image ISIC `outputs/evaluations/isic2018_*_val.json`.

### 1.4 Statistics & analysis code (cluster-aware — the candidate's core)
| Method | Status | Path |
|---|---|---|
| Cluster bootstrap 95% CI; Holm signed-rank; sign effect; **leave-one-cluster-out** | Present | `results/midterm_q2_rescue/scripts/generate_rescue_analysis.py` → `analysis/cluster_level_statistics.csv` |
| Seed-level Wilcoxon + Holm (all p=1.0) | Present | `results/final_manuscript_audit_v2/seed_level_wilcoxon_summary.csv` |
| Per-case distribution, outlier/trim, trade-off, subgroup, per-class | Present | `analysis/{per_case_distribution_summary,outlier_sensitivity,tradeoff_pattern_counts,subgroup_summary,per_class_summary}.csv` |
| Deterministic 24-panel failure selection | Present | `generate_rescue_analysis.py::selection_rows` → `analysis/failure_case_manifest.csv` (24) |
| GSL matched aggregation | Present | `…/scripts/aggregate_phase16_gsl.py` → `matched_gsl_summary.csv` |

### 1.5 Resource / profiling
| Item | Status | Path |
|---|---|---|
| Params, FLOPs, latency, peak mem | Present | `src/bds_lite/evaluation/profiling.py`; `training_cost_combined_summary.csv` |
| No-update RTX-5060 training-step profile (`optimizer_step:false`) | Present | `submission_blocker_resolution_v1/profiling/{training_cost_combined_summary,gsl_training_cost_summary}.csv` + `environment.json` |
| Gate-removal | Present | `analysis/gate_removal_summary.csv` ← `results/post_recovery_eval/p1b_gate_comparison/summary.csv` |

### 1.6 Per-case + aggregate outputs behind tables
All present and lightweight (keep in git): `recomputed_main_test_means.csv` (U-Net/BDS
rows), `matched_gsl_summary.csv`, `cluster_level_statistics.csv`, the `analysis/*.csv`
family, `outputs/evaluations/*.json|csv` (per-case). **Heavy re-derivation input:**
`outputs/evaluations/predictions/*.npy` (22,859, 1.2 GB) + `data/processed/` → Zenodo.
(GSL has no per-case prediction arrays; it is aggregate-only by design —
`analysis_run_summary.json: gsl_prediction_arrays_available=false`.)

---

## 2. Secrets / PII / paths (re-scanned over in-scope phase dirs)

- **Clean:** `midterm_q2_rescue/scripts`, `analysis/*.csv`, the GSL `configs/`, GSL run
  `config.json` (no absolute local paths), the candidate manuscript/tables. No API/W&B keys or emails.
- **Strip/exclude:**
  - `results/midterm_q2_rescue/analysis/generate_rescue_analysis.log` — contains
    `<source-root>/...` and `.venv/...` warning tracebacks → exclude all `.log`.
  - `manuscript_v2/build/main.fls`, `docs/devlog.md` (`<unrelated-local-path>/…`) —
    from Stage 1; exclude/scrub.
- **Action:** re-run a path/secret scan over the *final* staged set, esp. GSL `logs/*.log`
  and `runs/*/train.log` if any are shipped.

## 3. Large binaries → Zenodo (NOT git)

Stage-1 set **plus** the GSL matched rerun:
- `results/submission_blocker_resolution_v1/gsl_memory_bounded/checkpoints/` — 9 runs ×
  (best/last + 6 epochs) `.pt` → **Zenodo**.
- Keep in git: GSL `configs/`, `evaluations/*.csv`, `runs/*/config.json`+`metrics.jsonl`,
  `matched_gsl_summary.csv`, `splits/*locked_seed2026.json`.
- Carried from Stage 1: raw data 28 G (download + DOI only); `results/checkpoints` 22 G,
  `outputs/runs/*.pt`, predictions 1.2 G → Zenodo; **MedSAM foundation checkpoints 507 M
  → exclude entirely**; EGE-UNet checkpoints → exclude (out of scope).

## 4. Junk / cache to exclude
`.venv/`, `__pycache__/` (incl. `submission_blocker_resolution_v1/scripts/__pycache__`),
`.ruff_cache/`, `.pytest_cache/`, `src/bds_lite.egg-info/`, `manuscript_v2/build/`,
`build/`, `_archive/`, `archive/`, `.codex/`, `.agents/`, all `*.log`.

## 5. Environment pinning
`results/submission_blocker_resolution_v1/profiling/environment.json` (and per-run
`environment.json`): **Python 3.14.4, torch 2.11.0+cu130, CUDA 13.0, RTX 5060 Laptop GPU**.
`pyproject.toml` has ranges only; full pin → generate `requirements.txt` from `.venv`
in Stage 3 or mark TODO. No guessed versions.

## 6. Paper claims with NO backing artifact
**None.** Every numeric/structural claim in `BDS_Lite_submission_candidate_v1.md` and
Tables 1–6 traces to a present, script-produced artifact, spot-verified verbatim
(§B). The candidate's own Table 6 marks non-claims ("BDS-Lite beats GSL", "clinical
deployment", "faster training") as **not claimed**, consistent with the integrity rules.

---

## §G. Decisions for you (Stage 2 gate)

1. **Gate result:** B is cleared — confirm you accept it so we proceed to the Stage 2
   tree proposal.
2. **EGE-UNet / Boundary DoU code:** drop `src/.../ege_unet.py` + `boundary_dou.py`
   (cited-but-unused), or keep them with attribution? (Their *experimental* artifacts are
   OUT either way.)
3. **Config-of-record:** ship per-run `config_resolved.yaml` snapshots for U-Net/BDS-Lite
   (scrubbed) + the already-committed GSL `phase16_matched_gsl_*` configs. Optional clean
   reconstructed YAMLs only if labeled "reconstruction" and diff-verified — want them?
4. **ISIC split / `_test` misnomer:** ship `data/splits/` only; delete stale
   `splits/isic2018_split.json` + 389-image `*_val.json`; keep the 519-image `_test.json`
   names and document the `_test`→validation misnomer. Confirm.
5. **LICENSE:** MIT for original code (default). I'll inventory vendored third-party
   (GSL port `gsl.py`, U-Net, + EGE/BoundaryDoU if kept) with origin+license+citation.
6. **Stale provenance docs:** do **not** ship `manuscript_v2/claim_evidence_table.md` or
   `docs/MANUSCRIPT_CLAIM_TRACE.md`; replace with a fresh trace generated from this audit.
   Confirm.
7. **Release location:** sibling `<release-root>/` (current) — confirm.

*Stopping for your clearance of Section B before any Stage 2 work.*
