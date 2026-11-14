# Scientific-narrative assessment (for author review)

Status: analysis only. `main.tex`, `supplement.tex`, and the response letter are unchanged. New analyses below are exploratory and post hoc; none is registered. Sources: submitted manuscript (`recovered/peerj-submission/main.tex`), decision letter, current manuscript, raw per-case results, checkpoints.

## 1. Bottom line

**Recommendation: Narrative 2, in a bounded form.** BDS-Lite is presented as a modular, training-only boundary-supervision framework whose value lies in what its component evaluation shows, not in an accuracy gain.

**Your concern is justified.** My last rewrite is too negative, and not only in tone. It turned several "not established" results into "does not help", lumped a measured harm together with unresolved effects, and missed the most reproducible pattern and the most informative mechanism. Those are scientific errors, listed in Section 7.

**It would still be wrong to go back.** Some fixed-model differences are Holm-significant, but no accuracy gain for BDS-Lite holds across both model sets and run-to-run variation on any dataset. Narrative 1 is not defensible.

## 2. What the raw evidence supports

Component effects below come from one additive model over the 21 ablation runs per dataset, with run and test-unit uncertainty combined (`analysis/revision/narrative_component_effects.py`). Pooled rows are fixed-effect across datasets. Thresholds are the registered SESOI: 0.01 for DSC and Boundary F1 (BF1), 2 px for HD95.

### Three different kinds of result

| Status | Finding | Evidence |
|---|---|---|
| **Best-supported effect** | The surface loss lowers BF1 (size unclear) | Pooled −0.011 [−0.020, −0.002]; negative on all three datasets; removing it hurt in 1 of 15 leave-one-out cells. Not corrected across the 20 pooled tests |
| **Established mechanism** | Distillation transfers boundary-readable signal to the deployed projection | Section 3 |
| **Evidence of no relevant effect** | No component changes DSC or HD95 by a relevant amount, pooled | All eight pooled intervals lie inside the threshold. Example: distillation DSC −0.001 [−0.005, +0.002], HD95 +0.01 [−0.56, +0.57] px |
| **Evidence of no relevant effect** | ISIC HD95: no benefit from BDS-Lite | Within threshold in both model sets |
| **Failure to establish** | Decoder effect on BF1 | Pooled +0.011 [−0.002, +0.023]; ISIC +0.022 [−0.005, +0.049] |
| **Failure to establish** | Distillation and gate effect on BF1 | +0.003 [−0.007, +0.013] and +0.004 [−0.006, +0.015]. Gains above about 0.013–0.015 are excluded; smaller ones are not |
| **Failure to establish** | BDS-Lite vs U-Net on ISIC BF1 | Two-set mean +0.017 [−0.009, +0.044]; positive in 5 of 6 seed-matched pairs |
| **Failure to establish** | Most individual dataset-metric effects | Of 15 cells per component, 12 (decoder), 9 (distillation), 9 (gate), and 7 (surface) are undetermined, mainly on ACDC and Synapse |
| **Not estimable** | Component interactions | No decoder + surface arm exists |
| **Not established either way** | Dependence on task structure | Three datasets, confounded by modality and sample size |

In this additive analysis no component shows an established gain in any dataset-metric cell. Equally, the evidence does not show that boundary distillation has no effect.

### Findings that my rewrite missed or misstated

1. **"Did not reproduce" was too strong.** The two ISIC BF1 estimates (+0.031, +0.004) differ by +0.027 ± 0.053, so they are statistically compatible at run level.
2. **The original paper's core idea survives.** BDS-Lite versus U-Net across six dataset-by-set pairs: BF1 favorable in 5 of 6, HD95 unfavorable in all 6 (sign test p = 0.03, descriptive), spurious components up in all 6. BF1 moves through both precision and recall (decoder alone: +0.017 and +0.016 on ISIC). Boundary supervision shifts local contour agreement without improving the far tail. **Caveat:** once missed structures are scored, HD95 is favorable in 3 of 6, so this pattern applies to primary scoring only.
3. **The surface loss should be separated from distillation.** It is an add-on to boundary distillation, not part of it, and it is the one component with a measured harm. Lumping them made distillation look worse than the evidence shows.
4. **The cleanest test of distillation was skipped.** Distillation + gate on top of the decoder, without the surface loss, gives BF1 −0.005 [−0.009, −0.001] on ISIC (within threshold), +0.006 [−0.003, +0.014] on ACDC, and −0.004 [−0.023, +0.013] on Synapse. Bounded and neutral, neither good nor bad.

### Findings that limit the positive case

- **GSL remains at least as good.** All 15 contrasts favor it, 8 after Holm, none beyond the run-level margin. It also has the highest validation DSC on all three datasets among arms with a validation record, apart from LB-UNet on ISIC2018.
- **"BDS-Lite without the surface loss" must not become the method.** It looks strong on test data (ACDC first in DSC), but validation does not select it, so promoting it would be choosing on test results. Its Synapse gain also depends on aggregation: +0.015 registered, +0.035 unit-weighted, +0.042 sample-weighted.
- **Retraining noise is large.** The same U-Net recipe gives ACDC DSC 0.742 and 0.764. The retrained set selected earlier checkpoints, and the cause cannot be established.

## 3. Is there a contribution independent of accuracy?

| Candidate | Verdict | Evidence |
|---|---|---|
| Low parameter overhead | **Not evidence of effectiveness, and not distinctive** | 1,072 parameters follows from deleting a decoder. Decoder-only and GSL deploy with zero extra parameters, so BDS-Lite is the worse option here |
| Train/deploy decoupling | Real, not new | Verified, but a training-only boundary head discarded at inference, trained with BCE + Dice, already exists in STDC (Fan et al.). BiSeNet V2 and medical self-distillation methods share the pattern |
| Efficiency | Mixed | Inference +8% (batch 1), +15% (batch 16). Training +44% step time, +46% memory. "Lite" applies to deployment only |
| Training stability | No | Run SDs are similar across arms |
| Faster convergence | No | Epochs to 95% of best validation DSC are similar |
| **Boundary information transfer** | **Yes, with a sharp limit** | See below |
| **Evaluation knowledge** | **Yes** | Correction, run-to-run variance, undefined-case scoring. Valuable regardless of the method's accuracy |

**Transfer probe** (`analysis/revision/boundary_transfer_probe.py`; inference only, no fitting except a closed-form linear readout):

| Measure | ISIC2018 | ACDC | Synapse |
|---|---|---|---|
| R² of deployed `q_s` for teacher `q_b`, distilled arms | 0.70 to 0.79 | 0.28 to 0.73 | −0.08 to 0.61 |
| Same, undistilled arms | strongly negative | strongly negative | strongly negative |
| Frozen teacher head on deployed `q_s`: average precision, distilled vs undistilled | 0.123 vs 0.035 (×3.5) | 0.191 vs 0.077 (×2.5) | 0.104 vs 0.012 (×8.7) |
| Teacher decoder's own average precision | 0.168 | 0.547 | 0.569 |
| Linear readout of decoder feature `F_s`, distilled minus undistilled | +0.009 ± 0.015 | 0.000 ± 0.040 | −0.006 ± 0.054 |
| Same readout, arms with no decoder | 0.126 | 0.366 | 0.320 |

Interpretation:
- **Distillation works mechanically.** It places boundary-readable signal in the deployed projection, and the difference is large (Welch test over runs, p ≤ 0.003).
- **It did not detectably add boundary information to the features that make the prediction.** A plain U-Net already holds as much linearly readable boundary signal in `F_s`. The likely reason is that a 1×1 projection can satisfy the loss by reading what is already there.
- **This explains the null accuracy result** and suggests a design lesson: a learnable projection may absorb a feature-distillation loss.

**Gate.** It runs at 56–76% of its bound, so it is not near identity. Almost all of that is a constant per-channel scale (ISIC 0.072 of 0.084; ACDC 0.069 of 0.069; Synapse 0.095 of 0.096), and it is not boundary-selective (band/off-band ratio 0.75, 1.08, 1.11). Removing it changes under 0.1% of pixel labels (BDS-Lite, 96 test images per dataset). It behaved as a static rescale, not spatial modulation.

**Answer:** yes, but the contribution is knowledge, not a capability. It is what these components actually do and why they do not change accuracy, plus how reliably such effects can be measured. No engineering property distinguishes BDS-Lite favorably from the simpler alternatives.

Limits: linear readouts cannot see nonlinear information, the images are subsets (300 test images per dataset), and the analysis was designed after results were known.

## 4. The three narratives compared

| Criterion | N1: novel method | N2: framework + component evaluation | N3: limitations of boundary distillation |
|---|---|---|---|
| Evidential support | **Fails.** No established gain; GSL at least as good | **Strong** if scoped to what components do | **Partial.** Fits this recipe, not "boundary distillation" |
| Consistency with original submission | **Low.** It never claimed this ("controlled study") | **High.** Keeps recipe, decoupling, metric families, adds the attribution it deferred | Moderate. Keeps conditional framing, drops the method |
| Editor | Partly; wants method clarity but also evidence | **Best fit**: ablation, rigor, method, rationale | Good on rigor, weak on "proposed method" |
| Reviewer 1 | Poor; read it as a study | **Strong**: ablation was the main weakness | Good |
| Reviewer 2 | **Contradicts** "no notable methodological advance" | **Strong**: fits the "AI Application" reading | Good |
| Reviewer 3 | Raises novelty and SOTA demands we cannot meet | **Best available**: defines novelty, answers component effectiveness | Weak: novelty and method questions look evaded |
| Novelty | Overclaimed given STDC, BPKD | Modest and honest | Low; value rests on the evidence alone |
| Practical value | None beyond simpler options | **Clear design guidance** | Warnings only |
| Risk | **Overstates** | Could overstate if "framework" implies efficacy | **Understates and overgeneralizes** |
| PeerJ CS suitability | Poor: soundness is judged, and conclusions would exceed results | **Best** | Acceptable |

PeerJ's published criteria say decisions do not rest on novelty or impact and that null findings are considered, so a novelty claim buys little and can only hurt. I could not read the full page, so please check it. A PeerJ blog also describes newer pre-review screening.

### Strongest skeptical objections

**N1.** "GSL is simpler and did at least as well, and training-only boundary heads are not new. What is the method for?" I see no defensible answer.

**N2.**
- *"This is a negative paper with a positive label."* A real risk. The answer is to state the results in the abstract, including that no accuracy gain is established.
- *"Why publish a framework you would not recommend?"* The contribution is the decomposition and the design rules, and the paper should say plainly that simpler options suffice.
- *"The additive model and probe are post hoc."* True. Registered Holm families must stay primary, and these must be labelled exploratory.
- *"Three seeds cannot support design rules."* Partly true. Rules should be worded by strength of evidence.

**N3.**
- *"One recipe, one backbone, one weight setting, no augmentation: you cannot speak about boundary distillation."* Correct, and hard to answer. λ was checked with a single seed, and that check did not favor 0.05.
- *"Non-significance on 12–20 cases is not a limitation of the method."* Correct. The minimum detectable DSC effect on Synapse (0.023) exceeds the threshold (0.01).

### Risk of selective presentation

- **N1** would have to foreground +0.031 and hide +0.004, GSL, and prior art.
- **N2** could slide into emphasizing the transfer probe while downplaying that accuracy did not change, or into recasting the surface-free variant as the method.
- **N3** would have to foreground nulls while hiding the BF1 direction and the mechanism's success, and would read underpowered results as evidence of absence. My rewrite did this.

## 5. Why Narrative 2

It is the only narrative where every headline statement matches a result: it keeps the method the editor and Reviewer 3 ask about, delivers the attribution Reviewers 1 and 3 made central, and fits Reviewer 2's reading. It also reports the negative findings in full, as results about components, not as a verdict on the paper.

It needs one condition. **"Framework" must describe what BDS-Lite is, never what it achieves**, and the abstract must state that no accuracy gain over U-Net or GSL was established. Without that, it becomes a disguised Narrative 1.

## 6. Proposed framework for the paper

**1. Central claim.** In a compact U-Net, the components of removable boundary supervision have separable and mostly small effects: distillation transfers boundary information to a deployed projection without relevantly changing overlap or HD95, the signed-distance term lowers boundary agreement, and the auxiliary decoder's Boundary F1 gain is favorable but unconfirmed against run-to-run variation.

**2. Three contributions.**
1. *A modular training-only framework.* BDS-Lite separates four boundary signals and defines the exact deployed graph, with deployment and training costs reported. Evidence: verified equations and code; 1,072 parameters; collapse to the exact U-Net graph; +44% step time.
2. *Component-level evidence and mechanism.* A matched ablation with four comparators on three datasets, classifying each effect as harm, no relevant effect, or undetermined, plus direct measurement of what distillation and the gate do. Evidence: Sections 2 and 3; GSL at least as good.
3. *Error geometry and measurement reliability.* Boundary supervision shifts local contour agreement without improving the far tail, and conclusions about effects this small change with independent test data, repeated training, and scoring of missed structures. Evidence: 5/6 and 0/6 sign patterns; spurious components; corrected ISIC evaluation; 0.022 same-recipe gap; HD95 sign reversal.

**3. Central argument of the Introduction.** Training-only boundary supervision is attractive for compact models because nothing need remain at inference, and prior work, including STDC-style detail heads and boundary-aware distillation, has established that pattern. But composite recipes bundle several signals, so it is rarely known which part matters, or whether boundary information reaches the deployed features at all. BDS-Lite is built to answer that: its signals can be removed one at a time under a fixed protocol.

**4. Results structure.**
1. Corrected benchmark: BDS-Lite vs U-Net in two model sets, with run-to-run variation.
2. Component effects: registered contrasts first, then the additive summary with the three-way classification.
3. Mechanism: does boundary information reach the deployed features, and what does the gate learn?
4. Comparators and published context, with GSL given full weight.
5. How errors change: BF1 and HD95, spurious components, missed structures, all-case scoring, qualitative examples.
6. Deployment and training cost, last, so it is not read as evidence of effectiveness.

**5. Key message of the Discussion.** Boundary supervision in a compact U-Net acted on local contour agreement rather than overlap or the distance tail, and any such effect is small relative to retraining noise. The distinctive pathway does what it was built to do mechanically, but it relocates boundary signal instead of enriching the features that make the prediction, which explains why accuracy did not change. Simple options are therefore the sensible default, and a future transfer mechanism should act on those features directly. Claims stay limited to this backbone and protocol.

**6. Conclusion.** BDS-Lite provides a modular way to add and remove boundary signals in a compact U-Net at training time. No accuracy gain over U-Net or a single surface loss was established: the signed-distance term lowered boundary agreement, distillation and the gate did not relevantly change overlap or HD95, and the decoder's Boundary F1 effect remains unconfirmed. Practitioners should start with a removable decoder or a single surface loss, and confirm small effects with repeated training on independent test data.

**7. Reviewer concerns that remain unresolved.**

| Concern | Status | Could it be closed? |
|---|---|---|
| R3 novelty and framework justification | **Highest risk.** Novelty is modest, and prior art must be added | Partly, by honest positioning |
| R3-O practical significance | Open: no accuracy gain | No; answered by reframing |
| R3-M9, R3-R15 nnU-Net, Swin UNETR | Not run | nnU-Net possible, at a cost to protocol matching |
| E-4, R2-2 clinical metrics | Not computed (geometry unverified) | Only with verified native data |
| R3-M10 2.5D or 3D | Volume metrics only | With new experiments |
| R3-I8 ACDC and Synapse images | Label maps only | With permission |
| R3-S, R3-O statistical reliability | Improved, but BF1 undetermined | **Yes, with more seeds** |
| R2-5 weight justification | λ single-seed; no support for 0.05 | With a multi-seed λ study |
| R1-4 component causality | Main effects only | **Partly, with 9 runs** (below) |

## 7. Corrections needed whichever narrative you choose

| Location | Problem | Fix |
|---|---|---|
| `main.tex:34`, `:53`, `:256` | "no measurable benefit" lumps a harm with unresolved effects | State each component's status separately |
| `main.tex:232` | "it does not" converts non-detection into absence | Report bounds and what is undetermined |
| `main.tex:156` | "did not reproduce" overstates | Say the two estimates differ but are compatible |
| `main.tex:49` | "a third option" overclaims; STDC and BPKD are uncited | Credit prior art; narrow the novelty claim |
| `main.tex:234` | Gate-bound speculation is contradicted | Replace with the measured static-rescale finding |
| `main.tex:160` and letter line 45 | "withdrawn" and "not reliable" treat a non-significant pattern as resolved | Say not established |
| Table 3, Supp S5 | Synapse aggregation sensitivity is not disclosed in the main text | State it |
| Letter R2-6, R3-O | Inherit the same overstatements | Rewrite with the manuscript |

## 8. Experiments that would change the paper (need your authorization)

No training was run. `AGENTS.md` forbids it unless the plan lists it.

1. **Decoder + surface arm, 9 runs, roughly 7 GPU hours.** Fills the missing cell, so the decoder–surface interaction becomes estimable and contribution 2 gets stronger.
2. **More seeds on ISIC, about 28 runs, roughly 21 GPU hours.** U-Net, decoder-only, BDS-Lite, and GSL at 10 seeds each would cut the run-level margin from about ±0.038 to about ±0.014–0.021. That could confirm or exclude a decoder effect of the observed size, which is the largest open question.

Time estimates assume about 45 minutes per run, as in `REVISION_STRATEGY.md`.

## 9. Guardrails for the rewrite

- Keep the ISIC correction in the abstract and in its own section.
- Keep GSL's result prominent.
- Do not lead with +0.031, and do not select the surface-free variant as the method.
- Do not cite parameter count as evidence of effectiveness; describe "lightweight" as deployment-only.
- Keep registered analyses primary and label the additive model and probes as post hoc.
- Word each design recommendation by its evidence status.
- State the HD95 pattern together with its all-case caveat.

## 10. Sources and new artifacts

New, uncommitted: `analysis/revision/narrative_component_effects.py`, `analysis/revision/boundary_transfer_probe.py`, `revision/tables/families/component_effects.csv`, `revision/tables/families/boundary_transfer_probe.csv`. The direct contrasts, sign counts, precision/recall split, and gate structure check were run from scratch scripts and would be promoted if used.

External, from search results only; read each before citing:
- PeerJ: [Editorial criteria](https://peerj.com/about/editorial-criteria), [Aims and scope](https://peerj.com/about/aims-and-scope), [Rethinking Sound Science](https://peerj.com/blog/post/115284884570)
- [STDC: Rethinking BiSeNet (arXiv 2104.13188)](https://arxiv.org/pdf/2104.13188.pdf)
- [BPKD (WACV 2024)](https://openaccess.thecvf.com/content/WACV2024/html/Liu_BPKD_Boundary_Privileged_Knowledge_Distillation_for_Semantic_Segmentation_WACV_2024_paper.html)
- [MISSU (TMI 2023)](https://doi.org/10.1109/tmi.2023.3264433), [Dual self-distillation](https://pubmed.ncbi.nlm.nih.gov/39717673/), [ACW-SKD](https://www.nature.com/articles/s41598-025-98116-7), [MedCAGD (arXiv 2607.00409)](https://arxiv.org/pdf/2607.00409)
