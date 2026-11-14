# BDS-Lite Major Revision Plan — PeerJ CS-2026:06:142937

> **Objective:** Address every comment from the Editor and three reviewers with point-by-point responses and substantive revisions. Complete additional experiments, revise the manuscript, prepare the response letter, and resubmit under the constraint of **one RTX 5060 Laptop GPU (8 GB VRAM)**.
> This document is the **single source of instructions** for execution agents such as Codex and Claude. Every agent must read Sections 0 and 1 in full before starting work.

---

## 0. General Rules for Execution Agents (Mandatory)

### 0.1 Academic Integrity (Highest Priority)
1. **Never fabricate numbers, p-values, citations, or published results.** Every number in the manuscript must be traceable to a script-generated JSON/CSV file in the repository. An incorrect p-value was previously identified and withdrawn in this project; this rule must be strictly enforced.
2. **Generate all manuscript tables and numerical values programmatically.** `analysis/revision/make_tables.py` must generate `.tex` fragments, included in the manuscript via `\input{}`. Never manually copy numbers into `.tex`.
3. Verify all literature SOTA numbers (e.g., TransUNet on Synapse) against the original paper PDF or official repository. Record the source and page/table number in `revision/lit_numbers.csv`. Never rely on memory.
4. **Let results determine the narrative, not the reverse.** If ablations show that the ISIC benefit comes entirely from SDF surface loss, state exactly that. Retain the manuscript's positioning: honest, conditional conclusions rather than chasing SOTA.
5. For the paper suggested by Reviewer 3 (R3-I7, `doi:10.1002/ima.70424`), read the original work and assess relevance. **Cite it only if genuinely relevant** (the PeerJ Staff Note explicitly permits omission). Explain the decision truthfully in the response letter.
6. Update the AI-use disclosure in the Acknowledgments to cover AI tools (Claude, Codex, ChatGPT) used for coding or language editing during revision. PeerJ requires disclosure.

### 0.2 Locked Experimental Protocol
- **All new training runs must use the locked protocol:** the same split manifest (split seed 2026), training seeds 1/2/3, 224×224 input, AdamW (weight decay 1e-5), cosine schedule over 150 epochs, gradient clipping at 1.0, AMP, and the original batch sizes/learning rates (ISIC: batch size 8 / lr 3e-4; ACDC and Synapse: batch size 16 / lr 1e-4). No online data augmentation. Select the checkpoint by maximum validation DSC.
- **Never use the test set for any selection**, including model, hyperparameters, checkpoint, or threshold.
- Freeze the environment at PyTorch 2.11.0 + CUDA 13.0. Before starting, run `pip freeze > revision/env_freeze.txt`. Do not upgrade packages.
- If a comparator runs out of memory on 8 GB VRAM, first use gradient accumulation to preserve effective batch size, and document it in `revision/DEVIATIONS.md`. Do not silently alter the protocol.

### 0.3 GPU Usage Rules (Single-GPU Laptop)
- **Only one GPU training job may run at a time.** Route all training sequentially through `scripts/revision/run_queue.py` (implemented in WP0). Agents must not independently launch concurrent training jobs.
- Use `revision/.gpu.lock` to prevent collisions.
- The queue must support resuming: each run has a unique `run_id`; skip runs already containing `final_metrics.json`; save `last.pt` every 10 epochs for recovery.
- Each completed run must save `best.pt`, all validation/test predictions (`preds_{split}.npz`, uint8 label maps suffice), `final_metrics.json`, training logs, and the resolved configuration. Subsequent qualitative figures, error analysis, and volume-level evaluation depend on these prediction files.
- Do not run CPU-intensive evaluation in parallel with GSL training (which performs CPU-based EDT); doing so slows training and contaminates step-time measurements.
- Document laptop precautions in `README_revision`: AC power, high-performance mode, disabled sleep, and adequate cooling. Sample and log `nvidia-smi` temperatures and clocks for every run so throttling can be detected.

### 0.4 Workflow Conventions
- Assign each Work Package (WP) to a Git branch or set of commits. Prefix commit messages with `[WPx]`.
- Track progress in `revision/STATUS.md` using checkboxes for each WP. Append a one-sentence summary of every new experiment to `revision/RESULTS_LOG.md`.
- Tasks marked **[HUMAN]** must be decided or performed by Jingchen. Agents prepare the materials and stop for the human decision.
- Assume the manuscript source uses LaTeX (PeerJ template); WP0 must verify this if necessary.

---

## 1. Reviewer Comments → Action Matrix (Master Index)

ID convention: E = Editor; R1/R2/R3 = Reviewers 1/2/3; S = Staff. The response letter must address **every row** below.

| ID | Comment summary | Action | WP | Priority |
|---|---|---|---|---|
| E-1 | Experimental evaluation: more SOTA comparisons, ablations, statistical rigor, and qualitative results | See R1-3/R1-4/R3-R13/R3-R14/R3-R15 | WP2/4/6/7 | Must |
| E-2 | Clarity and completeness of methods and theoretical foundations | Complete formulas/dimensions, pseudocode, and a Theoretical Rationale section | WP5/WP9 | Must |
| E-3 | Why focus exclusively on compact U-Net? | Add a “Scope and rationale” paragraph and position architecture comparators | WP4/WP9 | Must |
| E-4 | Clinical relevance and potential impact | Discuss clinical implications; add ACDC clinical errors (EF, myocardial mass) and metrics in mm | WP1/WP9 | Must |
| R1-1 | Literature review should discuss other works' contributions, limitations, and problems | Add independent Related Work subsection and comparison table | WP9 | Must |
| R1-2 | Missing theoretical framework | Add Theoretical Rationale (error decomposition, gradient locality, gradient conflict) with testable predictions | WP5/WP9 | Must |
| R1-3 | Compare with reproducible SOTA reference works | Reproduce comparators under a matched protocol, plus a reference table of literature results | WP4 | Must |
| R1-4 | Missing ablations (major weakness) | Component ablations M1–M5 | WP2 | Must |
| R2-1 | Insufficient rationale for compact U-Net | Same as E-3 | WP9 | Must |
| R2-2 | Insufficient discussion of clinical relevance of boundary errors | Same as E-4 | WP1/WP9 | Must |
| R2-3 | Prose is too long and obscures meaning | Streamline entire manuscript (rules in §5.10) | WP9 | Must |
| R2-4 | `stopgrad(q_b)` undefined | Define it in Methods | WP9 | Must |
| R2-5 | Role of α unclear; 0.25 unjustified | Explain bounded modulation and run α sensitivity analysis | WP3/WP9 | Must |
| R2-6 | Positive assessment of AI Application positioning | Thank reviewer and clarify contribution framing | WP10 | — |
| R3-S | Statistical significance vs practical importance; rationale versus existing boundary methods; deployment-efficiency evidence | Practical relevance in WP6, comparators in WP4, efficiency in WP1h | WP1/4/6 | Must |
| R3-A1 | Abstract fails to articulate novelty | Rewrite Abstract to state novelty | WP9 | Must |
| R3-A2 | Abstract lacks quantitative results | Include key numerical results and CIs | WP9 | Must |
| R3-I1 | Why these three tasks? | Explain dataset choice and quantify structural-complexity descriptors | WP1e/WP9 | Must |
| R3-I2 | How do boundaries influence segmentation outcomes? | First-order error decomposition (§5.4) and clinical metrics | WP5/WP9 | Must |
| R3-I3 | Why not liver/brain-tumor datasets? | Textual rationale; optional additional 2D brain-tumor dataset (WP8) | WP8/WP9 | Must (text) / Could (experiment) |
| R3-I4 | Reorganize Introduction as motivation/background/literature/gap/contributions | Restructure Introduction | WP9 | Must |
| R3-I5 | What is the research gap and how is it addressed? | Explicit gap paragraph | WP9 | Must |
| R3-I6 | More discussion of BDS-Lite component choices | Add Methods “Design rationale” mapping each component to expected purpose | WP9 | Must |
| R3-I7 | Include recent work (suggested `doi:10.1002/ima.70424`) | Verify relevance before deciding; add relevant 2024–2026 literature | WP9 | Should |
| R3-I8 | Show dataset samples and annotated boundaries | New dataset example figure: image / GT / boundary target / SDF | WP7 | Must |
| R3-M9 | Individual component gains; nnU-Net/Swin UNETR/Attention U-Net comparisons | Ablations (WP2), comparators (WP4); clarify BDS-Lite has neither attention gates nor a multi-scale decoder component | WP2/WP4 | Must |
| R3-M10 | Inter-slice continuity for 2D slices; comparison with 3D/2.5D | Add volume-level 3D metrics and discussion; optional 2.5D experiment | WP1b/WP8 | Must (evaluation) / Could (2.5D) |
| R3-M11 | Preventing small targets (<1–2% of volume) from collapsing into background | Explain per-class soft Dice; stratify by foreground area; missed-structure rate | WP1c/WP9 | Must |
| R3-R12 | BDS-Lite underperforms in many cases; discuss limitations | Add Discussion subsection “When and Why BDS-Lite Does Not Help” | WP5/WP9 | Must |
| R3-R13 | Missing qualitative results (strong request; reason for major revision) | Qualitative comparisons and error maps; ISIC CC-0 usable; verify ACDC/Synapse licenses | WP7 | Must |
| R3-R14 | Statistical tests, error analysis, ablations | Extend statistics to all comparisons and perform error decomposition | WP1f/WP6 | Must |
| R3-R15 | SOTA comparisons on these datasets are “mandatory” | WP4 plus standard-protocol evaluation (WP1a/1b) | WP1/WP4 | Must |
| R3-O | Limited novelty, small/inconsistent gains, comparator selection rationale | Clear positioning, ablations, and comparator rationale | WP9/WP10 | Must |
| S-1 | Response letter must address all points; promised edits must appear in manuscript | WP10 and WP11 consistency check | WP10/11 | Must |

---

## 2. Identified Factual Issues in the Manuscript (Must Correct)

| # | Issue | Location (review PDF line numbers) | Resolution |
|---|---|---|---|
| F1 | Manuscript says official ISIC2018 test labels are “unavailable.” **This is outdated:** the ISIC Challenge data page now offers 2018 Task 1 Validation GT and Test GT (1,000 test images), under **CC-0** (https://challenge.isic-archive.com/data/) | L177–180, L477–480 | WP1a: evaluate every checkpoint on official test (1,000) and validation (100); retain original 519-image val results as the “model-selection split.” Revise/remove the fourth limitation |
| F2 | Manuscript claims dataset terms prohibit publishing patient images. **Not true of ISIC** (CC-0); ACDC and Synapse need verification | L345–349 | WP7: verify licenses; include ISIC qualitative figures in main text |
| F3 | `L_sdf-surface` is named but **has no equation** | Eq. 5, L146–150 | WP9: give the complete equation, based on implementation |
| F4 | `L_GSL` has no equation; “alpha schedule” unspecified | Eq. 6, L206–212 | WP9: specify equation and schedule |
| F5 | `stopgrad` undefined (R2-4); α=0.25 and λ_b=λ_distill=0.05 lack justification | L126–131, L153 | WP9 + WP3 |
| F6 | Gate tensor shape unclear: is g per-pixel/per-channel or global? What are W_g dimensions? Does boundary decoder predict per-class or union boundaries (“boundary_classes” in Fig. 1)? | L123–136, Fig. 1 | Verify code in WP0; clarify in WP9 |
| F7 | Text says “3×3 morphological gradient” whereas Fig. 1 says “Erosion / XOR / Dilate.” These are equivalent but inconsistent | L137–139, Fig. 1 | Use consistent terminology |
| F8 | Inference latency grows 24% (1.93→2.40 ms) despite only 0.68% more FLOPs; unexplained | Table 4 | WP1h: investigate added elementwise-gate kernel launches and batch-1 overhead; measure multiple batch sizes and CPU |
| F9 | How does slice-level macro DSC handle “empty classes”? Only Boundary F1 empty-boundary rules are stated | L241–246 | WP0: verify empty-mask conventions for DSC/IoU/HD95 and document in Methods; helps explain why slice-level numbers are not directly comparable with literature |
| F10 | Gate-collapse analysis uses another checkpoint family and evaluation pipeline, confusing readers | L375–387 | WP1i: evaluate gate collapse on matched checkpoints using the unified pipeline; replace/supplement old analysis |

---

## 3. Compute Budget (Single RTX 5060 Laptop)

### 3.1 Estimation Method
WP0 must **measure actual wall-clock times** from existing training logs for each run type, replacing the prior estimates below. Prior estimates use Table 4 step times (U-Net 39.6 ms, BDS-Lite 57.2 ms, GSL 52.3 ms, batch size 8):
- Compact U-Net runs: approximately 0.6–0.9 h/run (150 epochs including validation).
- BDS-Lite runs: approximately 0.9–1.2 h/run.
- Swin-Unet (ImageNet pretrained): approximately 1.5–2 h/run.
- nnU-Net 2D (250 epochs, one fold): approximately 5–8 h/dataset.

One complete matched experiment = 3 datasets × 3 seeds = **9 runs**.

### 3.2 Tiers and Budget

| Tier | Work | Runs | Estimated GPU time |
|---|---|---:|---:|
| **A (Must)** | Reproducibility check (U-Net ISIC s1 + BDS-Lite with gradient logs on all 3 datasets, s1) | 4 | ~4 h |
| A | Ablation M1: U-Net + SDF only | 9 | ~7 h |
| A | Ablation M3: BDS-Lite − SDF | 9 | ~9 h |
| A | Ablation M4: BDS-Lite − distillation | 9 | ~9 h |
| A | Ablation M5: BDS-Lite − gate (α=0) | 9 | ~9 h |
| A | Comparator C1: U-Net + Boundary Loss (Kervadec 2021) | 9 | ~7 h |
| A | Comparator C3: Attention U-Net (compact width) | 9 | ~8 h |
| A | Comparator C4: LB-UNet (lightweight boundary auxiliary; most relevant) | 9 | ~7 h |
| A (conditional) | Retrain GSL if its checkpoints/predictions are missing | 9 | ~8 h |
| **Tier A subtotal** | | **67 (+9)** | **~60–70 h** |
| **B (Should)** | Ablation M2: U-Net + auxiliary boundary head only | 9 | ~9 h |
| B | Comparator C2: U-Net + BoundaryDoU Loss (Sun 2023) | 9 | ~7 h |
| B | Comparator C5: Swin-Unet (2D transformer reference) | 9 | ~15–18 h |
| B | α sensitivity: α ∈ {0.1, 0.5, 1.0} × ISIC + Synapse × 3 seeds | 18 | ~18 h |
| B | λ sensitivity: λ_b=λ_distill ∈ {0.01, 0.2} × ISIC × 3 seeds | 6 | ~6 h |
| **Tier B subtotal** | | **51** | **~55–60 h** |
| **C (Could)** | nnU-Net 2D reference (own protocol, 1 seed, 250 epochs) | 3 | ~15–24 h |
| C | Extra dataset (2D brain tumor or Kvasir-SEG): U-Net/BDS-Lite/GSL/M1 × 3 seeds | 12 | ~8–12 h |
| C | Add seeds 4 and 5 for main methods (U-Net/BDS-Lite/GSL) | 18 | ~16 h |
| C | 2.5D variant (3 adjacent slices as channels; ACDC + Synapse; U-Net/BDS-Lite) | 12 | ~10 h |

**Conclusion:** Tiers A+B require approximately 120–130 GPU hours. At around 18 laptop GPU hours per day, that is **7–8 days**, feasible within a 30-day revision period. Tier C depends on progress. If behind schedule, Tier B C5 and sensitivity studies may be reduced to seed 1 only, explicitly labeled “single-seed reference.”

### 3.3 Queue Order (Uninterrupted Once Started)
1. Reproducibility checks (4 runs) → continue only after passing.
2. M1 → M3 → C1 → C3 → C4 → M4 → M5 → (conditional) GSL.
3. Tier B: M2 → C2 → α sweep → λ sweep → C5.
4. Tier C added upon **[HUMAN]** decision.

While Tier A uses the GPU, carry out WP1 CPU-side evaluation concurrently (except during GSL runs).

---
## 4. Work Packages (Detailed)

### WP0 — Repository Audit and Reproducibility Baseline (Days 1–2)
**Objective:** Establish the status of existing code, data, checkpoints, and prediction files, and build the training queue.

Tasks:
1. Write `revision/REPO_MAP.md` documenting:
   - Training/evaluation/analysis entry points; config structure; model registration mechanism.
   - Whether every completed matched run has `best.pt` and prediction files (**especially GSL per-image predictions**, which manuscript L350 says are unavailable).
   - Implementation details verified against code: exact `L_sdf-surface` equation; GSL equation and alpha schedule; gate tensor shapes; boundary decoder architecture/output channels (per-class or union); Smooth L1 β; empty-mask handling for DSC/IoU/HD95.
   - ACDC/Synapse conversion path (TransUNet H5?), preservation of original spacing, and availability of original NIfTI files locally.
   - Which patients appear in the ACDC test set, and whether both ED and ES frames are present.
   - Whether the 12 Synapse test cases match the standard TransUNet 12-case test list (if so, literature comparability improves substantially; note that this study trained on only 14 cases versus TransUNet's 18).
2. Run `pip freeze > revision/env_freeze.txt`; record GPU driver version.
3. Implement `scripts/revision/run_queue.py` and `revision/queue.yaml` following §0.3.
4. Add prediction saving if absent from current training scripts; add an optional gradient-conflict logging hook to BDS-Lite training for WP5 (disabled by default).
5. **Reproducibility check:** rerun U-Net ISIC seed 1 and compare against the original `final_metrics.json`. Tolerances: |ΔDSC| ≤ 0.005 and |ΔHD95| ≤ 1 px. If exceeded, investigate first (cuDNN nondeterminism, data version). **Do not start the queue until the check passes.**
6. Measure run durations and update §3.2.

**Acceptance:** Complete `REPO_MAP.md`; resumable queue; passed reproducibility check recorded in `RESULTS_LOG.md`.

---

### WP1 — Evaluation-Only Extensions (Primarily CPU; Parallel to GPU Queue)
Store all scripts in `analysis/revision/`. Make them work with **every method**, including newly added ones: input = list of run directories; output = CSV/JSON.

**WP1a — Official ISIC Test/Validation Evaluation** (F1, R3-R15)
- Download 2018 Task 1 Validation Input/GT (100 images) and Test Input/GT (1,000 images) from https://challenge.isic-archive.com/data/. License: CC-0; document source in README.
- Create arrays using **identical preprocessing** (scale to [0,1], 224×224, continuous interpolation for images, nearest-neighbor for masks).
- Compute all five metrics on the 224 grid. Additionally compute Jaccard and thresholded Jaccard at original resolution according to the official ISIC protocol (upsample predictions to original size) for literature comparison.
- Statistical unit: image (n=1,000). This provides ISIC with a test set fully independent of checkpoint selection for the first time in this study.

**WP1b — ACDC/Synapse Volume-Level (Stacked 3D) Evaluation** (R3-M10, R3-R15, Limitation 5)
- Stack 2D slice predictions along z into 3D volumes; upsample to original in-plane resolution, consistent with TransUNet inference (224 prediction → nearest-neighbor resizing to original dimensions).
- Report two sets of metrics:
  1. **TransUNet-compatible protocol:** volume-level DSC and HD95 (first verify whether the official TransUNet `test_single_volume` implementation passes spacing to HD95; follow the code exactly), for literature comparison.
  2. **Physical units:** if original NIfTI spacing is available, report HD95/ASSD in mm and surface Dice / NSD (τ = 2 mm), for clinical interpretation.
- 3D metrics naturally reflect inter-slice consistency. Discuss discontinuities caused by 2D training as a quantifiable issue (e.g., difference in number of 3D connected components versus GT).

**WP1c — Small-Structure and “Collapse” Analysis** (R3-M11)
- Report DSC/HD95 by class (ACDC: RV/Myo/LV; Synapse: eight organs) in the supplement, with a main-text summary of small organs such as gallbladder and pancreas.
- Stratify by foreground area fraction (<1%, 1–2%, 2–5%, >5%). For each method, report DSC, **missed-structure rate** (GT contains structure but prediction is empty), and **false-alarm rate** (GT is empty but prediction is nonempty).
- Mechanistic explanation: per-class soft Dice with macro averaging makes each class contribute irrespective of its area rather than allowing background to dominate purely due to area; CE provides per-pixel calibration. Use the missed-structure rate as empirical evidence regarding collapse, **reporting the actual result without assuming the conclusion**.

**WP1d — ACDC Clinical Metric Errors** (E-4, R2-2; high value)
- For test patients, calculate LV/RV volumes separately at ED and ES (pixel counts × voxel volume using original NIfTI spacing), derive LVEF, RVEF, and myocardial mass (Myo volume × 1.05 g/mL).
- For each method versus GT, report mean absolute error, Bland–Altman bias, and limits of agreement; perform paired comparisons by patient (n=20).
- If the test set lacks paired ED/ES data, report only volume errors and explain why.
- This directly connects “boundary error” with clinically meaningful quantities and belongs in main-text Results.

**WP1e — Dataset Structural-Complexity Descriptors** (R3-I1; supports the “structural-complexity gradient”)
- For each dataset, compute: number of foreground classes; mean number of connected components per image; fraction of boundary pixels; perimeter-to-area ratio (L/A); distribution of foreground area fractions; proportion of slices containing empty classes.
- Produce a compact main-text table (extend Table 1 or add a new one), turning the “complexity gradient” from a qualitative assertion into a measurable statement.
- Use L/A in the theoretical section's first-order error decomposition (§5.4).

**WP1f — Error Decomposition Analysis** (R3-R14, “error analysis”)
- Decompose FP/FN pixels into errors inside the boundary band (≤2 px) versus distant errors (>2 px).
- Analyze predicted connected components: number of spurious components (predicted components with no intersection with GT).
- Plot histograms or CDFs of surface distances by method and dataset to determine whether worsening HD95 reflects a few distant errors or a global offset.
- Prediction: Synapse HD95/ASSD deterioration is mostly due to distant FP components. **Follow the data; do not predetermine the conclusion.**

**WP1g — Complete GSL Predictions**
- If GSL checkpoints exist, run inference only to generate prediction files. If missing, retrain under the conditional Tier A item. Include GSL in failure-case analyses and qualitative figures thereafter.

**WP1h — Additional Deployment Efficiency Evidence** (R3-S, F8)
- Measure GPU latency for batch sizes 1/8/16 and laptop CPU latency (single-threaded and multithreaded); optionally measure latency after TorchScript/ONNX export.
- Include parameter counts, FLOPs, and latency for all new comparators in the same resource table.
- Use `torch.profiler` to investigate the +24% batch-1 latency: kernel count and launch overhead, supported by evidence.
- Correct wording: the training-only auxiliary branch is removable, but actual latency differences must be acknowledged; do not claim “zero overhead.”

**WP1i — Gate-Collapse Evaluation on Matched Checkpoints** (F10)
- Force g = 0.5 at inference (modulation factor fixed at 1), evaluate all metrics on matched checkpoints using the unified statistical pipeline. Replace or supplement the old “different checkpoint family” analysis.

**Acceptance:** Each subtask has a script, output CSV, and summary in `RESULTS_LOG.md`.

---

### WP2 — Component Ablations (GPU, Tiers A/B) (R1-4, R3-M9, R3-R14, E-1)

| ID | Configuration | Description |
|---|---|---|
| M0 | U-Net | Existing |
| M1 | U-Net + SDF surface loss (λ_sdf=1) | No auxiliary branch, P, or gate. **Critical ablation already promised by the manuscript** |
| M2 | U-Net + auxiliary boundary decoder + `L_boundary` | Pure multitask setup; no distillation, gate, or SDF (Tier B) |
| M3 | BDS-Lite − SDF | Auxiliary branch + distillation + gate |
| M4 | BDS-Lite − distillation (λ_distill=0) | P trained only through the gate |
| M5 | BDS-Lite − gate (α=0, gate identically 1) | Keep distillation |
| M6 | BDS-Lite full | Existing |

Analysis design:
- Addition pathway: M0 → M1 (SDF contribution) → M6 (added value of boundary-distillation path beyond SDF).
- Leave-one-component-out: M6 versus M3/M4/M5.
- Compare against theoretical predictions (§5.4): SDF should primarily affect distant FP and thus HD95/ASSD; boundary BCE/Dice concentrates on the boundary band and should primarily influence Boundary F1. **Report whether results support or contradict these predictions.**
- Statistics: each ablation versus M6 (ablation family) and versus M0 (baseline family), with separate Holm correction within each dataset (see WP6).
- Main text: one ablation table (3 datasets × 5 metrics × variants, mean ± seed SD) and a Δ heatmap (same style as Fig. 2).

---

### WP3 — Hyperparameter Sensitivity (GPU, Tier B) (R2-5)
- α ∈ {0 (=M5), 0.1, 0.25 (existing), 0.5, 1.0}; datasets ISIC + Synapse (complexity extremes), seeds 1–3.
- λ_b = λ_distill ∈ {0.01, 0.05 (existing), 0.2}; dataset ISIC, seeds 1–3.
- Output: line plots of metrics versus α and λ, with error bars across seeds.
- Explanation: α controls **bounded multiplicative modulation**, with modulation factor in [1−α/2, 1+α/2]. α=0.25 corresponds to [0.875, 1.125]. It was specified **a priori**, based on keeping modulation small and near the baseline, **not** tuned on test. Sensitivity analysis tests robustness of conclusions to α.
- **Important:** Sensitivity studies describe robustness; do not retrospectively select α using test metrics. Any new selection must use validation data only.

---

### WP4 — SOTA / Comparator Evaluation (GPU, Tiers A/B/C) (R1-3, R3-M9, R3-R15, E-1)

**Use two distinct levels and clearly separate them in the manuscript:**

**(1) Matched-protocol comparators** (same split, seeds, training schedule, input, no augmentation, and evaluation code)

| ID | Method | Category | Selection rationale | Tier |
|---|---|---|---|---|
| C1 | U-Net + Boundary Loss (Kervadec 2021, including region/boundary weight schedule) | Loss-level | Classical distance-map boundary loss | A |
| C2 | U-Net + BoundaryDoU (Sun 2023) | Loss-level | Recent boundary loss already cited | B |
| GSL | U-Net + GSL | Loss-level | Existing | — |
| C3 | Attention U-Net (same compact U-Net width c=32) | Architecture | Explicitly requested by R3; similar parameter scale | A |
| C4 | LB-UNet (Xu & Tong 2024) | Lightweight boundary-assisted architecture | Most directly related: boundary components **remain at inference** versus removable training-only components here | A |
| C5 | Swin-Unet (ImageNet pretrained, 224) | Transformer | R3 requests a transformer. Swin UNETR is natively 3D and mismatched to this 2D setting; use its 2D counterpart and explain in response letter | B |
| C6 | nnU-Net 2D (nnUNetv2, `splits_final.json` aligned to locked split, shortened-epoch trainer, 1 seed) | Self-configuring framework | Explicit R3 request; serves as an **own-protocol reference upper bound**, not matched | C |
| C7 | EGE-UNet (optional) | Ultra-lightweight | Lightweight skin-lesion SOTA | C |

Implementation notes:
- Prefer official implementations and record commit hashes and licenses in `revision/comparators.md`. LB-UNet is originally binary; change its output head to K classes for multiclass tasks. If changes are too extensive, evaluate only ISIC and explain.
- Swin-Unet requires three-channel input; repeat grayscale images across three channels. Flag pretraining as a protocol difference in tables.
- Handle 8 GB OOM according to §0.2.
- Include all comparators in WP1 (ISIC official test, volume-level metrics, resource table) and WP6 (statistics).

**(2) Literature reference table** (no reproduction; background context only)
- Under standard protocols (ISIC 2018 official test; Synapse TransUNet protocol; ACDC TransUNet or Swin-Unet protocol), show published numbers for representative methods: TransUNet, Swin-Unet, nnU-Net, UNet++, Attention U-Net, LB-UNet, EGE-UNet, etc.
- Annotate protocol differences per row: training case count, resolution, augmentation, pretraining, and 3D versus 2D. Table note: “not directly comparable; shown for context.”
- Present this study's WP1a/1b standard-protocol results at the end of the same table.
- **Verify the source of every number** (§0.1-3).

**Discussion framing:** This study does not aim to top SOTA leaderboards; it characterizes the conditional effects of boundary supervision on a fixed compact backbone. Comparators answer two questions: how BDS-Lite's effect compares in magnitude with other boundary losses/architectures, and how far a compact U-Net is from heavier models.

---

### WP5 — Mechanism Analysis and Theoretical Support (R1-2, E-2, R3-R12, R3-I2)

**WP5a — Gradient Conflict Measurement** (GPU; reuse three BDS-Lite seed-1 reproducibility runs)
- Every 50 steps, separately compute cosine similarity between ∇L_seg (CE + Dice) and ∇L_aux (λ_sdf·L_sdf, λ_b·L_b, λ_d·L_distill, calculated separately and then summed) on shared encoder parameters. Also log ratios of gradient norms.
- Preregistered hypothesis: higher cosine similarity/fewer conflicts on ISIC; lower or negative values on ACDC/Synapse, corresponding to the “capacity competition” interpretation.
- Output: cosine-similarity trajectories over training for all three datasets and a mean table. **Report results even if they refute the hypothesis**, and revise Discussion accordingly.
- Cite gradient-conflict literature, e.g., PCGrad (Yu et al. 2020), after verification.

**WP5b — First-Order Error Decomposition** (analytical derivation + WP1e L/A data)
- Let the predicted contour have mean normal displacement δ, foreground area A, and perimeter L. Then |P△G| ≈ L·|δ|, giving DSC ≈ 1 − L|δ|/(2A). HD95 and ASSD respond directly to δ and distant errors.
- Implication: for a large, compact, single target (ISIC lesion), a 1-px boundary improvement barely changes DSC but can directly change HD95/ASSD and Boundary F1. This explains ISIC's observed “unchanged DSC but improved boundary/distance metrics” pattern.
- Use dataset L/A from WP1e to estimate the magnitude of DSC change for a 1-px change in δ, and report in the manuscript.

**WP5c — Write a “Theoretical Rationale” subsection** (see §5.4).

---

### WP6 — Expanded Statistics (R3-S, R3-R14, E-1)
1. **Preregistration:** Before inspecting any new results, commit `analysis/revision/prereg_revision.md` containing comparison families, Holm correction scope, decision rules, and practical-relevance thresholds. The Git commit timestamp provides evidence.
2. Comparison families:
   - Family P (primary hypothesis, unchanged): BDS-Lite vs U-Net; Holm correction across five metrics within each dataset.
   - Family A (ablations): each variant versus full BDS-Lite; Holm correction within dataset.
   - Family C (comparators): each comparator versus U-Net (descriptive + inferential); Holm correction within dataset.
   - Omnibus across methods: Friedman test for each dataset × metric, followed by Holm-adjusted paired Wilcoxon post-hoc tests (Demšar 2006 approach); optional critical-difference diagram.
3. Statistical units unchanged: ISIC image (new official test n=1,000), ACDC patient (n=20), Synapse case (n=12). Average over seeds within each unit first.
4. **Practical relevance:** Define a smallest effect size of interest (SESOI) a priori; suggested DSC 0.01, Boundary F1 0.01, HD95 2 px or 1 mm, ASSD 0.5 px. Compare bootstrap 95% CIs against ±SESOI, classifying outcomes as superior / practically equivalent / inconclusive / inferior. This directly addresses statistical significance versus practical importance.
   - Since earlier results are already known, transparently state in preregistration that thresholds were chosen at the revision stage.
5. Tables: keep Table 3-style BDS-Lite vs U-Net comparisons in main text, adding SESOI judgments; place all comparisons in Supplementary Tables S1–Sx.
6. Discuss limited statistical power with ACDC n=20 and Synapse n=12; add an approximate minimum detectable effect based on cluster-level SD.

---

### WP7 — Figures (R3-R13, R3-I8)

**WP7a — License Verification [HUMAN Final Approval]**
- ISIC 2018 Task 1: CC-0 (ISIC data page); can be used directly; attribute source in captions.
- ACDC: read official Creatis data terms. Synapse (syn3193805): read its conditions of use. Document original URLs and conclusions in `revision/licensing.md`.
- If ACDC/Synapse terms are ambiguous: **[HUMAN]** email PeerJ staff to ask if attributed derivative image slices from these datasets may appear in a CC BY 4.0 article (agent drafts email).
- Fallback: for ACDC/Synapse show **label maps only** (GT and method-predicted labels, without original intensity images) and error maps; show full ISIC panels.

**WP7b — Dataset Example Figure (New Figure)**
- One row per dataset: original image | GT overlay | boundary target y_b | SDF target. Visually explain how boundary and SDF targets are generated, keeping terminology consistent with F7.

**WP7c — Qualitative Comparison Figure (New Main-Text Figure)**
- Columns: Image | GT | U-Net | BDS-Lite | U-Net+GSL | M1 (SDF only) | strongest comparator. Overlay contours (GT green, predictions in other colors).
- Rows: three examples per dataset, **chosen using deterministic rules** (largest positive ΔHD95, median, largest negative ΔHD95) to avoid cherry-picking. State selection rules in caption and record examples in a manifest.
- Add error-map row: FP in red, FN in blue.
- Revise previous failure-case descriptions (ISIC_0013140, ISIC_0016060, etc.) to refer to this figure.

**WP7d — Other Figures:** Ablation Δ heatmap; α/λ sensitivity curves; gradient cosine curves; surface-distance CDFs; ACDC clinical-metric Bland–Altman plots (supplement).

Figure requirements: vector PDF or ≥300 dpi; colorblind-friendly palette; typography consistent with PeerJ template. **Do not generate images using AI** (PeerJ policy).

---

### WP8 — Optional Extensions (Tier C) [HUMAN Decides]
- **Additional dataset** (experimental part of R3-I3; tests whether a single-contour task benefits again). Choose one:
  - 2D brain-tumor MRI (e.g., LGG FLAIR slice dataset, Buda et al.) → directly addresses “why no brain tumors.”
  - Kvasir-SEG (polyps, binary, single contour) → potentially more permissive image licensing.
  - Agent checks license and size for both, records in `revision/extra_dataset.md`, and **[HUMAN]** selects.
  - Run only U-Net / BDS-Lite / GSL / M1 × 3 seeds; image- or patient-level statistics as appropriate.
- **2.5D variant** (experimental part of R3-M10): three adjacent slices as input channels; evaluate U-Net and BDS-Lite on ACDC/Synapse. If time is short, use textual rationale and WP1b's 3D metrics only.
- **Seeds 4 and 5:** strengthen descriptive reproducibility.

---

### WP9 — Manuscript Rewrite (See §5)
Execute after experimental results are available. Result-independent sections (Introduction, Related Work, Methods formulas, theoretical derivation, clinical background) may be drafted earlier.

### WP10 — Response Letter (See §6)

### WP11 — Release and Quality Checks (See §8)

---
## 5. Section-by-Section Manuscript Revision Instructions

### 5.1 Title
Keep the title unchanged (the question-form title aligns with the paper's conditional-conclusion framing).

### 5.2 Abstract (R3-A1, R3-A2)
- Structure: background/gap (1–2 sentences) → **one sentence stating novelty** (below) → method and protocol → **quantitative results** → conclusion.
- Check the PeerJ CS author guidelines first for the abstract word limit and any structured-abstract requirement.
- Draft novelty statement (revise after final results):
  > “The contribution is twofold: (i) a training-only boundary-distillation recipe—an auxiliary boundary decoder whose features are transferred to the segmentation path through stop-gradient feature distillation and a bounded gate—that is removed at deployment, adding 1,072 parameters (0.056%) to the deployed U-Net; and (ii) a matched-protocol, component-ablated, multi-metric characterization showing when such boundary supervision helps, is neutral, or trades overlap against surface distance.”
- Results must include at least: ISIC official-test ΔHD95 and ΔBoundary F1 with 95% CIs; lack of stable advantage on ACDC; overlap–distance tradeoff on Synapse; a one-sentence ablation finding specifying which component drives which effect; deployment overhead.

### 5.3 Introduction (R3-I4, I5, I1, I2, I3, E-3, E-4, R2-1, R2-2)
Restructure into five paragraphs in the order requested by R3:
1. **Motivation:** Clinical consequences of boundary errors. ISIC: lesion-border irregularity is the “B” of the ABCD dermoscopy rule, used in melanoma screening and excision-margin planning. ACDC: EF and myocardial mass derive from boundary-dependent volume estimates, so contour offsets systematically bias clinical quantities. Synapse: organ volume measurements and radiotherapy organ-at-risk contouring; HD95 and surface Dice are contour-quality metrics. Verify and cite sources for each point.
2. **Background:** Differences between overlap and boundary/distance metrics; role of compact U-Net.
3. **Literature:** Short summary; move details to Related Work.
4. **Research gap:** Existing boundary approaches (a) are rarely compared on a fixed backbone under a common protocol, (b) rarely report all three metric families with cluster-aware statistics, (c) rarely attribute gains to individual components, and (d) often retain lightweight boundary modules at inference. Therefore the question **“when does boundary supervision help?”** remains inadequately answered.
5. **Contributions:** Update to include ablations, comparators, clinical metrics, and mechanism analysis, distinguishing methodological from empirical contributions.

Add **“Scope: Why Compact 2D U-Nets”** (R2-1, E-3). Draft rationale:
- (a) Deployment: point-of-care, edge-device, and CPU inference settings (dermoscopy apps, bedside devices). Removable training-time components matter particularly with tight deployment budgets.
- (b) Experimental control: high-capacity models may conceal/confound auxiliary-supervision effects. Capacity competition is easier to observe in compact networks and is especially relevant where design choices matter.
- (c) Reproducibility and accessibility: research conducted on a single laptop GPU; code already available publicly.
- (d) Lightweight U-Net research remains active (EGE-UNet, LV-UNet, LB-UNet, etc.).
- Explain that WP4 comparators establish how compact U-Net compares with heavier networks.

**Dataset selection rationale** (R3-I1, I3): The three tasks represent a complexity gradient from “one closed contour → three nested structures (including ring-shaped myocardium) → eight organs with strongly varying scales.” Quantify using WP1e descriptors. Explain why LiTS/BraTS were excluded: 3D, multimodal, large-volume tasks beyond compact 2D scope and single-GPU compute. Under the study hypothesis, single-contour tumor segmentation might resemble ISIC; discuss as future work (or report WP8 results if completed).

### 5.4 New Related Work Subsection (R1-1)
- Subsections: (i) boundary/surface losses (Boundary Loss, HD loss, Active Boundary Loss, BoundaryDoU, GSL); (ii) multitask boundary branches and shape constraints (Psi-Net, Gated-SCNN, etc.); (iii) knowledge distillation and auxiliary supervision (Hinton, FitNets, Structured KD); (iv) lightweight medical segmentation networks (EGE-UNet, LV-UNet, LB-UNet, UNeXt, etc.); (v) evaluation guidance (Metrics Reloaded, Taha & Hanbury).
- **Comparison table** (R1 explicitly requests “contributions, limitations, problems”): Method | Category | Mechanism | Additional inference cost | Evaluation datasets | Reported metric families | Limitations.
- Add relevant recent 2024–2026 work. Handle R3-I7's suggested paper according to §0.1-5.

### 5.5 Methods
1. **Design Rationale subsection** (R3-I6): Map each component → targeted error mode → expected metric family.

   | Component | Mechanism | Expected effect |
   |---|---|---|
   | SDF surface loss | Distance-to-GT-boundary-weighted penalty on predicted probability mass | Distant FP → HD95/ASSD |
   | Auxiliary boundary decoder + L_b | Concentrates gradients near boundary band | Local contour → Boundary F1 |
   | Feature distillation | One-way transfer of boundary features into segmentation path | Boundary information in segmentation features |
   | Bounded gate | Bounded modulation of segmentation features using boundary-aware features | Small, stable modulation |

2. **Complete equations** (F3, F4, F6): Specify `L_sdf-surface`, `L_GSL` (including alpha schedule), `L_boundary`, and `L_distill`; include all tensor dimensions (`F_s ∈ ℝ^{c×H×W}`, `q_s, q_b ∈ ℝ^{c_b×H×W}`, `W_g ∈ ℝ^{c×c_b}`), **verified against the actual implementation**.
3. **Define stopgrad** (R2-4), draft:
   > “stopgrad(·) denotes the stop-gradient operator, which is the identity in the forward pass and has zero Jacobian in the backward pass (implemented as `tensor.detach()` in PyTorch). Consequently, L_distill updates the segmentation-side projection P and, through it, the segmentation decoder, but provides no gradient to the boundary decoder; the boundary branch is trained only by L_boundary. This makes the transfer one-directional: boundary features act as a fixed target for the segmentation path at each step, preventing the segmentation path from pulling the boundary representation toward itself.”
4. **Explain α** (R2-5), draft:
   > “Because g ∈ (0,1), the modulation factor 1 + α(g − 0.5) is bounded in [1 − α/2, 1 + α/2]. α therefore sets the maximum relative rescaling of each decoder feature; α = 0 recovers an unmodulated decoder. We fixed α = 0.25 a priori (factor range [0.875, 1.125]) so that the gate could re-weight features by at most ±12.5% and the trained network stayed close to the baseline operating point; its sensitivity is reported in Fig. X.”
   Also justify λ_b=λ_distill=0.05 based on actual code or historical records (e.g., whether initial auxiliary-loss magnitudes were around x% of main loss), and refer to WP3.
5. **Training pseudocode (Algorithm 1):** single-step forward pass, each loss, backward pass; deployment conversion steps.
6. **Hyperparameter table:** all methods, including comparators, in one table.
7. **Evaluation subsection:** document DSC/IoU empty-mask conventions (F9), volume-level protocol (WP1b), physical-unit metrics, NSD, clinical metrics, and SESOI decision rules.
8. **Datasets subsection:** add official ISIC validation/test, remove “test labels not publicly available,” and update Table 1 (structural descriptors, or a separate table).

### 5.6 New “Theoretical Rationale” Subsection (R1-2, E-2, R3-I2)
Place at the end of Methods or as a standalone subsection:
1. First-order error decomposition (WP5b): explain why overlap may be insensitive to boundary improvements, with dataset-specific magnitudes.
2. Gradient locality: CE/Dice gradients span the regions; boundary BCE/Dice focuses on the boundary band; SDF-weighted gradient magnitude increases with distance to boundary and thus primarily penalizes distant errors. Derive expected metric families for each component.
3. Multitask gradient interactions in shared encoders and capacity competition in compact models: auxiliary objectives may align with main error modes for a single contour, but heterogeneous boundaries can cause conflicts.
4. Bounded-gate stability (α bounds).
5. State **testable predictions P1–P4** and assess each in Results against WP2 ablations and WP5a gradient conflict measurements, stating whether supported or unsupported. This constitutes the “theoretical framework” requested by R1.

### 5.7 Results (New/Reordered Subsections)
1. Main matched comparison (retain; add official ISIC test and SESOI judgment).
2. **Component ablations** (new).
3. **Comparisons with boundary-loss and architecture comparators** (new), plus literature reference table.
4. **Standard-protocol (volume-level) and clinical metrics** (new): WP1b, WP1d.
5. **Small structures and error decomposition** (new): WP1c, WP1f.
6. **Qualitative results** (new): WP7c.
7. Mechanism analysis: gradient conflict (new).
8. Sensitivity analysis (new).
9. Resource analysis (expanded: WP1h, WP1i).

### 5.8 Discussion
- Add **“When and Why BDS-Lite Does Not Help”** (R3-R12): explain mechanisms using ablations, gradient conflict, structural descriptors, and per-class results—not just “task dependence.”
- Discuss positioning against comparators and literature (R3-O).
- Clinical implications: use WP1d to discuss whether Δ produces clinically meaningful changes; report honestly, including possible nonsignificance.
- Discuss 2D versus 3D (R3-M10).

### 5.9 Limitations / Conclusions
- Delete or revise limitations that have been resolved: no independent ISIC test, no ablations, and distance metrics reported only in pixels (if WP1b adds mm metrics).
- Retain genuine limitations: sample sizes, single backbone, no-augmentation protocol, and possible suboptimal performance of comparators under this protocol (especially differences for Swin-Unet and nnU-Net).
- Update Conclusion to reflect ablation findings. Remove “Future work: SDF-only ablation” once completed.

### 5.10 Concision Rules (R2-3)
- Sentences ≤30 words; one principal idea per paragraph.
- State each caveat fully only once (in Limitations), and reference it briefly elsewhere. Particularly reduce repeated “recipe-level contrast” language (currently appears approximately six times).
- Remove piled-up hedges such as “essentially,” “descriptive,” and “slightly.”
- Move implementation details (conversion pathways, audit details, bit-for-bit checks, etc.) to Supplementary Methods.
- Because new material is substantial, trim at least 20% of the old main-text content. Put detailed tables in the supplement.
- After every rewrite, a human must read the manuscript end to end; agent editing may introduce inaccuracies.

### 5.11 Acknowledgments
- Update AI-tool disclosure to cover every tool and purpose used during revision (code and writing).

---

## 6. Response Letter Template and Requirements

File: `revision/response_letter.tex` (or `.docx` if PeerJ requires).

Structure:
1. **Opening to the Editor** (no more than half a page): thank the editor; summarize major changes in 5–7 bullets, including ablations, comparators, official ISIC test, volume-level/clinical metrics, qualitative figures, theoretical subsection, expanded statistics, and concise writing.
2. **“Summary of Major Changes” table:** Change | Addressed comment IDs | Location (revised manuscript page/line).
3. **Point-by-point responses:** for every ID in the §1 matrix:
   - **Comment:** quote the reviewer's original wording, with review-PDF line numbers.
   - **Response:** polite, specific, evidence-based.
   - **Changes:** location in the revised manuscript (tracked-changes PDF line number) plus excerpts of key new text.
4. Points requiring polite clarification:
   - R3-M9 “attention gates, multi-scale decoders” are **not** components of BDS-Lite. Clarify the four actual components, explain their ablations, and add Attention U-Net as an architecture comparator.
   - R3-I3 liver/brain tumors: explain scope and compute limitations (plus WP8 results if available).
   - R3-M10 3D: explain the 2D setup and new volume-level 3D metrics (plus 2.5D results if available).
   - R3-I7 suggested paper: explain whether it was cited, and why.
   - R3-R14 statistical table: politely point out that original Table 3 already included Wilcoxon + Holm + bootstrap, and explain the expanded comparison scope and practical-relevance decisions.
   - Explain substituting Swin-Unet for Swin UNETR.
5. **Tone:** thank reviewers for each comment. Avoid defensive phrasing; prioritize “we did X, and obtained Y.” Never exaggerate new results.

---

## 7. Timeline (Starting November 2, 2026)

| Period | Tasks |
|---|---|
| **Week 1 (Nov 2–8)** | Complete WP0; commit preregistration; start Tier A queue after reproducibility passes; run WP1a–1i on CPU in parallel; WP7a license checks; **[HUMAN]** email PeerJ if necessary |
| **Week 2 (Nov 9–15)** | Complete Tier A → begin Tier B; WP5a gradient analysis; WP6 statistical pipeline (validate initially on existing results); initial WP7 figures; start result-independent WP9 writing (Introduction, Related Work, Methods, theory) |
| **Week 3 (Nov 16–22)** | Finish Tier B; consolidate all results in `RESULTS_LOG.md`; **[HUMAN]** review findings and approve narrative; write WP9 Results/Discussion; Tier C if capacity permits |
| **Week 4 (Nov 23–30)** | WP10 response letter; WP11 QA; generate tracked changes with `latexdiff`; **[HUMAN]** review manuscript, check submission-system information, and submit |

- **[HUMAN]:** First confirm the date of the editorial decision email and calculate the 30-day deadline. If the work is likely to exceed it, request an extension politely in Week 1 or 2, noting that substantial experiments are necessary (agent may draft the email).

---

## 8. Completion Criteria and Final Checklist (WP11)

**Experiments and data**
- [ ] Every Must item in the §1 matrix has a deliverable (script + result file + manuscript location).
- [ ] All new runs contain config, log, `best.pt`, prediction files, and `final_metrics.json`; `DEVIATIONS.md` records every protocol deviation.
- [ ] Every manuscript number can be regenerated by `make_tables.py`; spot-check 10 numbers against raw JSON.
- [ ] Every row in the literature-number table has a traceable source in `lit_numbers.csv`.
- [ ] Preregistration commit predates generation of the corresponding new results.

**Manuscript**
- [ ] Abstract states novelty and quantitative results; word count satisfies PeerJ requirements.
- [ ] All F1–F10 are corrected.
- [ ] All equations are complete; notation consistent; stopgrad, α, and λ are defined and justified.
- [ ] Main text adds Related Work comparison table, Theoretical Rationale, ablations, comparators, standard-protocol/clinical metrics, qualitative results, and dataset example figure.
- [ ] Figure/table numbering, cross-references, and citations are correct; verify author/year/venue/DOI for each new reference.
- [ ] AI-use disclosure updated; Data Availability identifies official ISIC validation/test sources and license.
- [ ] Limitations agree with new results and no longer list resolved issues.

**Response letter and submission materials**
- [ ] Response letter addresses every row of §1; every claimed revision can be found in manuscript (S-1).
- [ ] Both tracked-changes (`latexdiff`) and clean versions are ready.
- [ ] Repository includes new configs, scripts, results; README revision section; tag `v2-revision`; updated Zenodo version DOI in manuscript.
- [ ] **[HUMAN]** Verify authors, affiliations, funding, and conflicts of interest in submission system (cannot be changed after acceptance).
- [ ] **[HUMAN]** Read entire manuscript before submission.

---

## 9. Risks and Contingencies

| Risk | Contingency |
|---|---|
| Reproducibility check fails (drift beyond tolerance) | Investigate dataset versions and determinism first. If original results cannot be reproduced exactly, rerun all seeds for the original three methods under one environment (+27 runs), preserving comparability, and disclose |
| Ablation shows nearly all gain is from SDF loss (M1 ≈ M6) | Report honestly; an important finding in itself: “the distillation pathway adds no benefit in this setup.” Revise title/abstract wording to emphasize conditional conclusions |
| BDS-Lite advantage disappears on official ISIC test | Report honestly; downgrade the 519-image validation results to selection-split results; change conclusion to “not reproduced on independent test” |
| Comparators substantially outperform BDS-Lite | Consistent with study's non-SOTA framing; discuss applicability limits of compact models and training-only auxiliary supervision |
| Swin-Unet or nnU-Net OOM/too slow on 8 GB | Gradient accumulation; reduce to seed 1; use shorter-epoch nnU-Net trainer with clear labeling; if still infeasible, retain literature reference table only |
| ACDC/Synapse licenses prohibit image publication | Substitute label maps and error maps (WP7a fallback); retain full ISIC figures |
| Insufficient time | Priorities: Tier A experiments > WP1 evaluation > qualitative figures > writing > Tier B > Tier C; request extension early |
| Laptop overheating causes abnormal step times | Repeat resource profiling separately from a cold start; do not use training duration as part of study conclusions |
