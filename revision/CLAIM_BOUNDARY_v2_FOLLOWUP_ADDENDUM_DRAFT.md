# Claim boundary v2: follow-up addendum (DRAFT for the author's approval; not in force)

**Status:** draft, 2026-11-10. `revision/CLAIM_BOUNDARY_v2.md` is unchanged and remains the
governing claim boundary. Until the author approves this addendum, the follow-up wording in the
manuscript stays within v2. The checks in the last section show where it does.

**Source of every number:** `revision/followup/results/contrasts.csv` (registered analysis,
`analysis/followup/analyze.py`), the official ACDC patients 101–150, 17 runs per arm, and the
Stage 3 run-level estimator. Δ is the first arm minus the second.

## A6 (proposed). Follow-up controls: secondary, exploratory

| # | Comparison | Boundary F1 Δ [95% CI] | Holm p (C1–C4) | Registered category |
|---|---|---|---|---|
| A6.1 | C2 D+KD-full − D | +0.0106 [+0.0034, +0.0177] | 0.0196 | P |
| A6.2 | C3 D+KD-band(compact teacher) − D | +0.0118 [+0.0037, +0.0200] | 0.0196 | P |
| A6.3 | C1 D+KD-band − D+KD-full | +0.0073 [−0.0005, +0.0150] | 0.0945 | U |
| A6.4 | C4 D+KD-band − D+KD-band(compact) | +0.0060 [+0.0001, +0.0119] | 0.0945 | U |

Registered mapping: band case **B3**, teacher case **T2**.

**Proposed allowed wording:**
- "In exploratory controls on the same held-out patients, full-image distillation and band
  distillation from a teacher without pretraining or a capacity advantage also raised Boundary
  F1 over D."
- "The replicated gain is not shown to depend on the boundary band or on the teacher's advantage."
- "What the band and the pretrained teacher added beyond these controls was not resolved."

**Required labels:**
- secondary and exploratory;
- the same patients as Stage 3, whose results had been read;
- the registration was committed before training, but has no externally verified timestamp
  (Amendment 1).

## Proposed changes to the forbidden claims

| # | Change | Reason |
|---|---|---|
| F1 | Keep the forbidden claim. Update its reason: the D+KD-full control now exists. C1 is unresolved (+0.0073, Holm 0.095), and full-image KD alone gave a gain (C2). | The control is exploratory and did not establish a band contribution. |
| F3 | Keep. Add: "nor that the teacher's advantage is unnecessary in general". | One compact teacher and one external teacher, each trained once. Pretraining and capacity remain confounded. |
| F18 (new) | Forbid: "the band restriction / the pretrained teacher has no effect" or "is irrelevant". | C1 and C4 have positive point estimates (41% and 34% of the Stage 3 Δ) with intervals reaching about 0.015 and 0.012. Unresolved is not absence. |
| F19 (new) | Forbid describing any follow-up result as confirmatory, or as replicating Stage 3. | The follow-up reuses the examined test patients. |
| F20 (new) | Forbid any external-timestamp claim for the follow-up registration. | Amendment 1. The `.ots` files of `affab9e` remain pending and were never verified. |
| F21 (new) | Forbid causal claims from the post hoc error-location analysis, including "KD helps where errors are near the contour". | Post hoc. The association was null or negative (Supplementary S22). |

## Manuscript sentences the author should check against v2

1. **Discussion, "In-network transfer as a control".** It places the compact-teacher result
   (C3) beside the in-network result (A1). It says explicitly that the two "are not compared with
   each other". It infers only that "the absence of an informational advantage alone does not
   explain" the in-network null result. This is meant to stay within §0.3 / F15 (side by side,
   no difference of effects). The author should confirm it does.
2. **Results, follow-up secondary metrics.** C1 HD95 and ASSD are reported with Holm-within-metric
   p values (0.043 and 0.034) as "descriptive". Rule A4 governs only the Stage 3 secondary
   wording. This addendum proposes the same practice for the follow-up: descriptive, never
   leading, never replacing the primary result (F16).
3. **Abstract, last sentence.** "The gain is better described as an effect of distilling a
   separately trained network's predictions than as boundary-specific distillation." This is
   the registered T2/B3 narrative, worded more cautiously than the registered phrase "general
   soft-label KD effect, neither boundary nor teacher specific".
4. **The A3 literal Stage 3 wording** is unchanged in the Results ("without a D+KD-full control it
   does not show that the gain is specific to boundaries"). It is the frozen registered sentence.
   The follow-up paragraph that comes after it reports the control.
