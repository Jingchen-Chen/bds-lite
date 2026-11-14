# Response to the Editor and Reviewers

**Manuscript:** PeerJ Computer Science CS-2026:06:142937, *When does boundary distillation help lightweight medical image segmentation?* (major revision)

We thank the Editor and the three reviewers. Their requests for an ablation, for statistical testing, for comparisons and for qualitative results led us to retrain every model several times, and that work changed what the study shows. We describe the changes openly below, including an error in our submitted evaluation.

**How to read this letter.** Comments are numbered E.1–E.6 (Editor), R1.1–R1.6, R2.1–R2.7 and R3.1–R3.20. Each entry gives a summary of the comment, our response, and the location of the change. Line numbers are the margin line numbers of the revised manuscript PDF (`main.pdf`); tables, figures and Supplementary sections (S1–S24, `supplement.pdf`) are cited by number. Each entry carries one status: **Done** (addressed in the revised manuscript), **Partly done** (addressed within stated limits), or **Not done** (with the reason).

## 1. What the revised paper shows

The reviewers asked which component of BDS-Lite is responsible for its effect. Answering that required a reference that already contains the boundary decoder, repeated training runs, and controls. The revised paper is an evaluation study of training-only boundary signals in a compact 2D U-Net, and it reports four findings ({{L:abstract}}), which the Discussion interprets ({{L:disc_answer}}):

1. **Run-to-run variation is as large as the effects.** In the arms with ten or more runs, the standard deviation of Boundary F1 between retrainings of one recipe was {{R:RV.sd.acdc.min}} to {{R:RV.sd.acdc.max}} on ACDC and {{R:RV.sd.isic2018.min}} to {{R:RV.sd.isic2018.max}} on ISIC2018.
2. **In-network distillation added no detectable gain.** The feature distillation and gate of BDS-Lite changed Boundary F1 by {{R:IN.nosurf.pooled.boundary_f1.est}} (95% CI {{R:IN.nosurf.pooled.boundary_f1.text}}; three runs per arm on each dataset) beyond a training-only boundary decoder, and the complete recipe scored lower in Boundary F1 than the decoder alone on ISIC2018 ({{R:TEN.full.decoder.boundary_f1.delta}}, {{R:TEN.full.decoder.boundary_f1.text}}; confirmed on fresh seeds).
3. **External-teacher distillation helped, in a pattern consistent with regularization.** It raised Boundary F1 by {{R:FIN.cf.abs}} ({{R:FIN.cf.text}}) on {{R:OT.patients}} held-out ACDC patients with {{R:CF.n}} runs per arm, in a confirmatory study whose analysis plan was fixed and timestamped before the test patients were used; under the prespecified rule this is a gain of unclear size, because the unrounded lower bound, {{R:RAW.FIN.cf.low}}, lies just below the preset smallest effect of interest, 0.01 (Supplementary S18). Gains also appeared in exploratory controls with whole-image distillation ({{R:FU.C2.boundary_f1.delta}}) and with a compact teacher that scored below the reference ({{R:FU.C3.boundary_f1.delta}}). The band and the pretrained teacher were not shown to add to the gain, but their point estimates ({{R:FU.C1.boundary_f1.delta}} and {{R:FU.C4.boundary_f1.delta}}) are a third to two fifths of it, so we do not read the controls as showing that they contribute nothing ({{L:disc_teacher}}).
4. **On ACDC, augmentation mattered more than any boundary signal.** On the same {{R:OT.patients}} patients, flip-and-rotation augmentation of the reference raised Boundary F1 by {{R:E3.E3a.boundary_f1.delta}} ({{R:E3.E3a.boundary_f1.text}}; exploratory, not prespecified), and with augmentation in both arms a distillation gain was not detected ({{R:E3.E3.boundary_f1.delta}}, {{R:E3.E3.boundary_f1.text}}).

The conclusion is that, on ACDC, the augmentation regime mattered more than the choice of boundary signal, and that effects of this size can be resolved only over repeated training runs. The qualifications are collected once, in the Limitations ({{L:lim}}): the confirmed gain, of unclear size under the prespecified rule (Supplementary S18), covers one dataset; our references are below published results because they were trained without pretraining and mostly without augmentation; and the boundary-decoder gain ({{R:IN.dec.pooled.boundary_f1.est}}, {{R:IN.dec.pooled.boundary_f1.text}}) is exploratory.

*Presentation.* The main text classifies each result as confirmed, detected or not detected, defined once in the Statistics ({{L:vocab}}) and in the evidence column of Table 2, and rounds to three decimals; the supplement keeps four decimals and the prespecified readings against the preset 0.01 threshold. Analyses added after the results were known are marked post hoc (the interaction between augmentation and distillation, {{L:res_interaction}}; Supplementary S24).

## 2. Why the numbers differ from the submitted version

The absolute scores in the revised manuscript are much lower than in the submitted version. There are four reasons, and the first is an error of ours. The revised manuscript summarizes them in its Study design ({{L:corrections}}); the details are here and in Supplementary S4 and S5.

1. **The submitted ISIC2018 evaluation overlapped the training images.** The 519 images on which the submitted ISIC2018 results of U-Net and BDS-Lite were computed were not held out from those models: 358 were in the models' training set, 84 in their checkpoint-selection set, and 77 were held out. The favourable differences lay in the training images. We withdraw the submitted ISIC2018 results and the conclusion drawn from them, and we apologize for the error. The audit, with the split manifests, checkpoint hashes and a table of the submitted contrast divided by image membership, is in Supplementary S4. All models are now evaluated on the 1,000 official ISIC2018 test images.
2. **Classes absent from both prediction and reference were scored as perfect.** In the submitted means, a foreground class that was absent from both masks of a slice counted as 1. This inflated the ACDC and especially the Synapse means, where many slices contain few of the organs. The revised analysis excludes such pairs ({{L:empty}}) and reports missed structures and false alarms separately (Supplementary S8).
3. **The models were retrained.** The training script of the submitted models is not preserved. The submitted version stated that code, configurations and split manifests were available in a public repository; the checkpoints of that version are preserved (byte-identical to the archive manifest of that release) and its tables can be recomputed from them (Supplementary S4), but the script that trained them is not, so the submitted training cannot be repeated and that statement was too broad. Every arm of the revised manuscript was retrained with the released code on documented partitions; the main-text tables therefore describe different trained models from those of the submitted version.
4. **Recipe contrasts average within patient or case.** Absolute scores (Table 4) still average over images or slices within a run, but every contrast between recipes first averages within each patient (ACDC) or case (Synapse), and the training run is the unit of replication ({{L:stats}}).

The table separates these steps for the two models of the submitted version. Column A is what we reported. Column B re-scores the same checkpoints with the revised metric definition (reason 2) and, for ISIC2018, on the official test set (reason 1). Column C is the retrained models of the revised manuscript (reason 3), scored as in B.

| Dataset | Model | A: submitted version | B: same checkpoints, revised scoring | C: retrained models (revised manuscript) |
|---|---|---|---|---|
| | *Trained models* | submitted checkpoints, seeds 1–3 | submitted checkpoints, seeds 1–3 | retrained with the released code, seeds 1–3 |
| | *Test set* | ISIC2018: 519 images, 358 of them training images; ACDC: 20 internal test patients; Synapse: 12 test cases | ISIC2018: 1,000 official test images; ACDC and Synapse as A | as B |
| | *Empty class pairs* | scored as 1 | excluded | excluded |
| | *Aggregation* | mean over images or slices, then over seeds | mean over images or slices, then over seeds | mean over images or slices, then over seeds |
| ISIC2018 | U-Net | DSC 0.883, BF1 0.552 | DSC 0.835, BF1 0.362 | DSC 0.841, BF1 0.363 |
| ISIC2018 | BDS-Lite | DSC 0.883, BF1 0.560 | DSC 0.838, BF1 0.392 | DSC 0.837, BF1 0.367 |
| ACDC | U-Net | DSC 0.789, BF1 0.831 | DSC 0.764, BF1 0.808 | DSC 0.742, BF1 0.783 |
| ACDC | BDS-Lite | DSC 0.785, BF1 0.827 | DSC 0.764, BF1 0.808 | DSC 0.758, BF1 0.801 |
| Synapse | U-Net | DSC 0.854, BF1 0.861 | DSC 0.516, BF1 0.537 | DSC 0.487, BF1 0.506 |
| Synapse | BDS-Lite | DSC 0.857, BF1 0.863 | DSC 0.517, BF1 0.538 | DSC 0.496, BF1 0.516 |

BF1: Boundary F1. Sources: column A, Table 2 of the submitted version (reproduced in Supplementary S4); columns B and C, Supplementary S5 ("Two model sets"). Column C equals Table 4 of the revised manuscript. On ACDC and Synapse the step from A to B is the empty-class correction alone, on the same checkpoints and test data; on ISIC2018 it also includes the change of test set. The step from B to C is retraining; the ACDC and Synapse U-Nets of the retrained set score lower than those of the submitted checkpoints, and because the original training script is lost we cannot establish why (Supplementary S5). We therefore compare arms only within the retrained set.

## 3. Why the direction of the conclusion changed

The submitted version concluded that BDS-Lite was most useful on ISIC2018, the single-contour task. The revised manuscript reports that on ISIC2018 the complete BDS-Lite recipe scored below the decoder-only reference. Two different things changed, and we keep them apart.

*What the retrained models show.* With ten runs per arm on the official test set, BDS-Lite minus the plain U-Net was {{R:TEN.full.unet.boundary_f1.delta}} in Boundary F1 (95% CI {{R:TEN.full.unet.boundary_f1.text}}), and BDS-Lite minus the decoder-only reference was {{R:TEN.full.decoder.boundary_f1.delta}} ({{R:TEN.full.decoder.boundary_f1.text}}), confirmed on seven fresh seeds (Holm p = {{R:TEN.full.decoder.boundary_f1.holm}}). The decoder-only arm did not exist in the submitted version; it was added because Reviewers 1 and 3 asked for an ablation. It is this comparison, which the ablation made possible, that reverses the reading: in the retrained models the gain over the plain U-Net is associated with the boundary decoder, and a further gain from the components added to it was not established ({{L:res_decoder}}; {{L:res_full}}; Table 5).

*What re-scoring the submitted checkpoints shows.* On the official test set, the submitted BDS-Lite checkpoints still scored above the submitted U-Net checkpoints in Boundary F1 (0.392 against 0.362; image-level difference +0.031, 95% CI +0.026 to +0.036; Supplementary S5). That is a difference between two fixed sets of three models, and it was not reproduced by the retrained models, whose three-run difference is within run-to-run variation. Checkpoints of the earlier pipeline that include a decoder-only arm were also re-scored, with the revised evaluator and test sets (column B scoring), so they differ from the retrained models in the training pipeline only, whose script is not preserved: there, BDS-Lite minus the decoder on ISIC2018 was {{R:H.H5.isic2018.boundary_f1.delta}} {{R:H.H5.isic2018.boundary_f1.ci}} and the pooled decoder gain over the U-Net was {{R:H.H1.pooled.boundary_f1.delta}} {{R:H.H1.pooled.boundary_f1.ci}}, with three runs per arm (Supplementary S15). The earlier checkpoints therefore do not show the decoder gain, and with three runs per arm their interval for the deficit of BDS-Lite is too wide to confirm or contradict the ten-run result, as the Limitations of the revised manuscript state ({{L:lim_precision}}).

## 4. Other corrections

- **Statistical unit.** The submitted analysis averaged seeds within images and tested images. The revision treats the training run as the unit of replication; image-level tests of fixed models are summarized in the supplement (Supplementary S5, S14), and their full tables are in the evidence archive (Supplementary S16).
- **Inference cost.** The submitted Table 4 reported a batch-1 latency of 1.93 ms for U-Net and 2.40 ms for BDS-Lite (10 warm-up and 50 measured passes). The revised Table 9 reports medians of a saved profile, {{R:RF.gpu.b1.u}} ms and {{R:RF.gpu.b1.g}} ms (FP32, batch 1). The two were measured with different protocols and are not combined. Parameter counts and GFLOPs are identical in both versions. The submitted training-step times, peak memory and U-Net+GSL timings cannot be traced to a preserved result file and are not carried into the revision. The submitted statement that the deployed network was "essentially identical in size to U-Net" is replaced by the measured cost of the retained projection and gate ({{L:res_cost}}).

## 5. PeerJ requirements for AI Application articles

| Requirement | Where it is addressed | Status |
|---|---|---|
| Code used in the study is available | The code, configuration files, split manifests and analysis scripts are in the GitHub repository given in the submission's data statement (https://github.com/Jingchen-Chen/bds-lite), together with the two evidence archives with SHA-256 manifests described in Supplementary S16. The statement also gives the source and access terms of each dataset. | Done |
| Introduction: motivation for the study and for the technical choices | {{L:intro_motivation}} (boundary error), {{L:intro_compact}} (compact U-Net), {{L:routes}} (why these routes are compared) | Done |
| Third-party data: source, version, split | {{L:data}}; licences in Supplementary S16 | Done |
| Preprocessing actually performed | {{L:preprocessing}} | Done |
| Rationale for models, losses, teacher and reference methods | {{L:comparators}}, {{L:segloss}}, {{L:teacher}}, {{L:controls_method}} | Done |
| Computing infrastructure | {{L:training}}: GPU, Python, PyTorch, CUDA, cuDNN and NumPy versions as recorded with each run. The operating system was not recorded with the runs and is not stated. | Partly done |
| Evaluation method, metric definitions, rationale and limitations | {{L:metrics}}; statistics {{L:stats}} | Done |
| Results reported with tables and figures; discussion uses the defined metrics | Tables 4–9, Figures 2–3; Discussion | Done |
| Conclusions: model type and purpose, limitations, future directions | {{L:conclusions}} | Done |
| Acknowledgments and references | {{L:ack}}; reference list checked against the citations | Done |

## Editor

### E.1
**Comment (summary).** Address every point raised by each reviewer.

**Response.** Done. All 39 points are answered below, and each change is located in the revised manuscript.

### E.2
**Comment (summary).** Prioritise the experimental evaluation: comparisons with additional methods, ablation studies, statistical rigour, qualitative results.

**Response.** Done, with one limit on the qualitative results.

- *Ablation.* An eight-cell component ablation on all three datasets, and ten runs per arm for four arms on ISIC2018 with a decision on fresh seeds (Table 2, studies 1–2; Table 5).
- *Comparisons.* Four reference methods under the same protocol: U-Net+GSL, U-Net with Boundary Loss and Attention U-Net (Table 4), and an adapted LB-UNet (Supplementary S7); published results with their protocol differences, and volume-level scores of our U-Net for comparison with them (Table 8).
- *Statistical rigour.* The training run is the unit of replication; every contrast has a 95% interval and, where several contrasts are tested together, a Holm-adjusted p ({{L:stats}}). One design was replicated on held-out patients under a plan fixed in advance (Table 6; a gain of unclear size under the prespecified rule, Supplementary S18). Post hoc analyses are labelled as such.
- *Qualitative results.* Figure 3 shows three official ACDC test slices with the image, the reference mask, the predictions of D and D+KD-band and a change map. Contour maps for ISIC2018 test images are in Supplementary S12. See R3.11 and R3.16 for the limits.
- *Error analysis.* A new Results subsection and Table 7 report missed structures and false alarms ({{L:res_errors}}).

**Location.** Tables 2 and 4–9; Figures 2–3; {{L:stats}}.

### E.3
**Comment (summary).** Clarify and complete the description of the method and its theoretical foundations.

**Response.** Done. A new Methods subsection, "Framework: contour error, overlap, and the routes of a boundary signal", gives the first-order relation between contour displacement and DSC and the four routes by which a boundary signal can reach the output; each arm is then defined by the routes it opens. Eq. 1 uses the mean absolute normal offset, so that inward and outward displacements do not cancel. The stop-gradient operator, the gate and its scale α are now defined in the text (R2.5, R2.6).

**Location.** {{L:framework}}; {{L:eq_dsc}}; {{L:stopgrad}}; {{L:alpha}}.

### E.4
**Comment (summary).** Give the rationale for methodological choices, for example the exclusive focus on compact U-Net architectures.

**Response.** Done. The Introduction now has a paragraph on this choice with two reasons. First, a compact network has little spare capacity, so training-only signals that leave the inference network unchanged are particularly attractive there. Second, the effects in question are of the order of the variation between retrainings, and resolving them took 10 to {{R:CF.n}} training runs per arm; we therefore spent the compute budget on repeated runs rather than on model size, and holding the architecture fixed lets a difference between arms be assigned to the training signal. The consequence is a limit on scope, which the Limitations state. The rationale for the datasets, the segmentation loss, the teacher, the training protocol without augmentation and the reference methods is stated where each is introduced.

**Location.** {{L:intro_compact}}; {{L:data_choice}}; {{L:aug_rationale}}; {{L:comparators}}; {{L:segloss}}; {{L:teacher}}.

### E.5
**Comment (summary).** Discuss the clinical relevance and potential impact of the work.

**Response.** Partly done. The Introduction now explains, with references, why contour error matters in the three applications (border irregularity in melanoma assessment, ventricular volumes and myocardial mass, radiotherapy contours), and why overlap scores register it unevenly. We state explicitly that the study measures contour agreement on public benchmarks and not a clinical outcome; distances are in pixels of the resized 224 × 224 grid, and we did not compute physical distances, volumes or functional indices, so we cannot translate a Boundary F1 difference of 0.018 into millimetres or millilitres. We state this as a limitation and do not claim clinical relevance for the effect sizes. The practical implications are stated for model developers, not for clinical use.

**Location.** {{L:intro_motivation}}; {{L:practice_scope}}; {{L:lim_scope}}.

### E.6
**Comment (summary).** Staff note: all comments must be addressed in the response letter, and edits mentioned in the letter must be inserted in the manuscript.

**Response.** Done. Every response below names the location of the corresponding text. Where we did not make a change, the entry says so and gives the reason.

## Reviewer 1

### R1.1
**Comment (summary).** The study is interesting and well described; some adjustments are needed.

**Response.** We thank the reviewer. The adjustments are answered in R1.2–R1.6. We note that the findings of the revised manuscript differ from those the reviewer read (Sections 1–3 above).

### R1.2
**Comment (summary).** Add a literature review covering other works with their contributions, limitations and associated problems.

**Response.** Done. The Related work section is organised by type of boundary signal (losses, branches, distillation) and by the measurement problem, and a new Table 1 lists each method with its type, what it keeps at inference and its limitation relevant to this study. The longer table with the metric families each source reports is in Supplementary S1. Three studies published in 2025 were added where they bear on our design, each in one sentence of the Related work and one row of the table in Supplementary S1: a comparison in which the benefit of a Boundary DoU term differed between a convolutional U-Net and the transformer-based SwinUNETR (Sutassananon et al., 2025); a network that guides its decoder with boundary masks from edge-detection operators (CTO; Lin et al., 2025); and distillation from a teacher of the student's own architecture (Dong et al., 2025), which relates to our compact-teacher control. The review concentrates on the methods that define our arms and reference methods; it is not a systematic survey of boundary-aware segmentation.

**Location.** {{L:related}}; Table 1; Supplementary S1.

### R1.3
**Comment (summary).** The paper lacks a theoretical framework; discuss the relevant concepts in greater depth.

**Response.** Done. See E.3. The framework has two parts: why overlap is an uneven measure of contour error (Eq. 1, with the per-dataset consequence: a one-pixel shift costs about {{R:S.isic2018.1.drop}} DSC on an average ISIC2018 lesion and about {{R:S.acdc.2.drop}} on the ACDC myocardium), and the routes by which a boundary signal reaches the output, which define the arms and the reference. For the interpretation of the external-teacher results, the Related work and the Discussion now cite the literature that analyses distillation as regularization (gains from a teacher no larger than the student, Furlanello et al., 2018; soft targets as learned label smoothing, Yuan et al., 2020), and the Discussion proposes a mechanism: the teacher's cached probabilities were computed on its own training images, where they are likely close to the reference labels except near the contours, so distillation acts like label smoothing near the contours ({{L:disc_mechanism}}). We did not develop a formal theory; the mechanism is stated as an interpretation, and a soft-label control derived from the reference masks is named as future work.

**Location.** {{L:framework}}; {{L:related}}; {{L:disc_teacher}}; {{L:future}}.

### R1.4
**Comment (summary).** Present a comparison with the state of the art, even by reproducing some reference works.

**Response.** Partly done. We trained four reference methods under our protocol (three in Table 4, an adaptation of LB-UNet in Supplementary S7), and Table 8 lists published results. We did not reproduce the published transformer models or nnU-Net. The published values use other splits, pretraining, augmentation and volume-level scoring, so Table 8 is context and not a ranking, and the manuscript says that our matched-protocol DSC is lower than the published values on every dataset. To make the gap interpretable, Table 8 now also gives the volume-level DSC of our plain U-Net, computed from the existing predictions: {{R:RV.vol.acdc.U}} on ACDC and {{R:RV.vol.synapse.U}} on Synapse, against per-slice values of {{R:RV.sub.acdc.C.dsc}} and {{R:RV.sub.synapse.C.dsc}}. The remaining distance to the published values (0.90 on ACDC, 0.77 to 0.79 on Synapse) reflects our weaker training protocol, and the manuscript states that the references are weak in absolute terms and that the effects are effects on such references.

**Location.** Table 4; Table 8; {{L:res_published}}; Supplementary S7.

### R1.5
**Comment (summary).** The absence of an ablation study is the only weakness of the experimental design.

**Response.** Done. The ablation covers eight of the sixteen combinations of the four components (Supplementary S3) on three datasets with three runs per arm, and four arms were extended to ten runs on ISIC2018. Two limits remain: Table 5 adds feature distillation and the gate as a pair; the cells that add them one at a time (D+G+S, D+F+S) exist only together with the surface loss and are reported in the supplement (Supplementary S7, S13; stated in the Limitations), and with three runs per arm an effect was detected in only one of the five pooled contrasts (Table 5). Its result is the main change of the paper: the boundary decoder raised pooled Boundary F1 by {{R:IN.dec.pooled.boundary_f1.est}} (95% CI {{R:IN.dec.pooled.boundary_f1.text}}), and adding feature distillation and the gate to it changed Boundary F1 by {{R:IN.nosurf.pooled.boundary_f1.est}} ({{R:IN.nosurf.pooled.boundary_f1.text}}).

**Location.** Table 5; {{L:res_decoder}}; {{L:res_innetwork}}.

### R1.6
**Comment (summary).** Without ablation, results cannot be attributed to individual components, which makes the title question hard to answer.

**Response.** Partly done. The ablation and the controls now allow a conditional answer, which the Discussion and Conclusions state directly: in the studied setting, distillation from an external teacher helped (on ACDC, a gain of unclear size under the prespecified rule) when the network was trained without augmentation, and in exploratory controls a gain remained with whole-image distillation (the band restriction removed, boundary pixels still in the loss) and with a compact teacher, a pattern consistent with regularization. The limits are also stated: the ablation has three runs per arm, it does not assign the ISIC2018 deficit of the complete recipe to one component, and single-component estimates for feature distillation and the gate exist only from an exploratory additive model (Supplementary S13).

**Location.** {{L:disc_answer}}; {{L:disc_bds}}; {{L:conclusions}}; {{L:lim_precision}}.

## Reviewer 2

### R2.1
**Comment (summary).** The focus on compact U-Nets is not clearly justified.

**Response.** Done. See E.4. The justification has two parts, both now in the Introduction: training-only signals matter most where capacity is limited, and effects of this size need 10 to {{R:CF.n}} runs per arm, so the compute budget went to repeated runs of a fixed compact backbone rather than to larger models.

**Location.** {{L:intro_compact}}.

### R2.2
**Comment (summary).** The clinical relevance of boundary errors is not discussed in detail and should be expanded.

**Response.** Partly done. See E.5. The motivation paragraph now explains what a displaced contour changes in each of the three applications, with references, and Eq. 1 quantifies why overlap hides it for some structures. We did not add clinical validation, and the text says that none was done.

**Location.** {{L:intro_motivation}}; {{L:dsc_relation}}.

### R2.3
**Comment (summary).** The writing is unclear in places and needlessly verbose.

**Response.** Partly done. The manuscript was rewritten, and its sentences are shorter, but the study itself became larger (five studies, 24 supplementary sections), so the article is not shorter. To keep it readable, the evidence level of each result is defined once (Statistics and Table 2) and results are then stated plainly; the four findings are listed in the Abstract only, the Discussion interprets them, and the Conclusions have five sentences; limitations are collected in one Limitations subsection. In the supplement, process descriptions were condensed and most internal file names were removed (run identifiers remain where a record must be matched to the archive, Supplementary S15–S17), and the supplement was shortened from 67 to {{PAGES:supplement}} pages: image-level tests of fixed models, tables that repeated other tables, and long profile and per-class listings are no longer printed. Each place where a table was taken out says so, and its values are in the evidence archive described in Supplementary S16 or in another table; no analysis was removed. The main text has {{PAGES:main}} pages.

**Location.** Whole manuscript; for example {{L:abstract}}, {{L:design}}, {{L:lim}}.

### R2.4
**Comment (summary).** The contributions are an evaluation of BDS-Lite on three tasks, a comparison of two geometry-aware approaches, and findings on the complementary value of metrics; there is no methodological advance, but the work suits an AI Application paper.

**Response.** We agree with this positioning and have written the revision as an evaluation study. It does not claim a methodological advance for BDS-Lite. Its contributions are the component ablation, the replication with controls, the comparison under augmentation, and the run-to-run noise baseline.

**Location.** {{L:intro_contrib}}.

### R2.5
**Comment (summary).** Some methodology is poorly reported; stopgrad(q_b) is used but never defined.

**Response.** Done. The operator is now defined (identity in the forward pass, zero Jacobian in the backward pass; `detach()` in PyTorch), together with its consequence: the distillation loss updates the projection, the segmentation decoder and the encoder, and sends no gradient into the boundary decoder.

**Location.** {{L:stopgrad}}.

### R2.6
**Comment (summary).** The parameter α of Eq. 1 is not explained, and its value of 0.25 is not justified.

**Response.** Done. The text now states that α bounds how much the gate can rescale a feature (the factor lies in [1 − α/2, 1 + α/2], so α = 0.25 allows ±12.5%, and α = 0 is the identity). The value was set in single-seed Synapse validation runs during development, in which α = 0.5 worsened HD95 for several organs, and was frozen before the matched experiments without test data. A sensitivity analysis over five values is in Supplementary S11; it does not single out 0.25.

**Location.** {{L:alpha}}; Supplementary S11.

### R2.7
**Comment (summary).** The conclusions are consistent with the results, which indicate that boundary-aware objectives benefit only single-contour datasets.

**Response.** This conclusion of the submitted version is withdrawn. It rested on the ISIC2018 evaluation that overlapped the training images (Section 2, reason 1). On the official test set and with retrained models, the complete BDS-Lite recipe scored below the decoder-only reference on ISIC2018, and the gain that replicated, of unclear size under the prespecified rule, was on ACDC, a multi-structure task (Section 3).

**Location.** {{L:res_full}}; {{L:res_replication}}; Supplementary S4.

## Reviewer 3

### R3.1
**Comment (summary).** The improvements are modest and inconsistent; clarify their statistical significance and practical relevance, justify the framework against existing methods, and give quantitative evidence for deployment efficiency.

**Response.** Partly done; the three parts are answered in turn. The second part is answered by reframing the paper rather than by a new argument for BDS-Lite.

- *Significance and relevance.* Every contrast now has a 95% interval, and each result is stated as confirmed, detected or not detected. We agree that the effects are modest: the boundary effects are of the size of the run-to-run standard deviation on the same dataset ({{R:RV.sd.acdc.min}} to {{R:RV.sd.acdc.max}} on ACDC, {{R:RV.sd.isic2018.min}} to {{R:RV.sd.isic2018.max}} on ISIC2018, in the arms with ten or more runs), which is now a stated finding. A difference of 0.01 was preset as the smallest effect of interest; it is an analysis threshold without clinical validation. The Statistics also state that the recipe-level standard error counts one variance term twice, so the intervals are somewhat conservative ({{L:se_conservative}}; Supplementary S20).
- *Justification against existing methods.* We accept the reviewer's point. The revision does not argue that BDS-Lite should be preferred over existing methods; under the matched protocol, the reference methods scored as well as or higher than BDS-Lite on most cells of Table 4. The paper is now an evaluation study, and its value lies in the controlled ablation, the confirmatory replication with controls and the noise baseline rather than in BDS-Lite itself.
- *Deployment efficiency.* Table 9 gives parameters, GFLOPs and latency. BDS-Lite keeps a projection and a gate at inference, which cost {{R:RF.params.extra}} parameters and {{R:RF.gpu.b1.pct}}% and {{R:RF.gpu.b16.pct}}% GPU latency at batch sizes 1 and 16 in our profile; the manuscript no longer describes BDS-Lite as training-only.

**Location.** {{L:res_variation}}; {{L:vocab}}; {{L:se_conservative}}; Table 4; Table 9; {{L:res_cost}}.

### R3.2
**Comment (summary; reviewer's item Abstract 1).** Abstract: the novelty of BDS-Lite is not defined.

**Response.** Done. The Abstract now names BDS-Lite in full and says what it is (in-network feature distillation and a gate, built on an auxiliary boundary decoder). The Introduction now states where it comes from: BDS-Lite is a recipe that we designed for compact U-Nets, and each of its components follows a published pattern (a training-only boundary head, a distance-weighted surface loss, hint-style feature distillation), which the Related work cites; the combination is ours. We do not claim it as a methodological advance: the revised results do not support BDS-Lite as an addition to the decoder, and the paper's contribution is the evaluation.

**Location.** {{L:abstract}}; {{L:intro_background}}.

### R3.3
**Comment (summary; reviewer's item Abstract 2).** Abstract: quantitative results are not reported.

**Response.** Done. The Abstract states four findings, each with its main estimate and 95% interval.

**Location.** {{L:abstract}}.

### R3.4
**Comment (summary; reviewer's item Introduction 1).** Why were skin lesions, cardiac structures and abdominal organs chosen?

**Response.** Done. The Data subsection now states that the three public tasks were chosen because their structures differ in number, size and perimeter-to-area ratio, the quantities that determine how strongly overlap responds to a contour shift (Eq. 1): one lesion per image, three nested cardiac structures and eight abdominal organs of very different size. It also states that the tasks were not selected to form a scale of difficulty. Why no further task was added is answered in R3.6.

**Location.** {{L:data_choice}}; {{L:dsc_relation}}.

### R3.5
**Comment (summary; reviewer's item Introduction 2).** How do boundaries affect these segmentation results?

**Response.** Done. See E.5 and R1.3: the motivation paragraph gives the role of the contour in each application, and Eq. 1 gives its effect on overlap per structure.

**Location.** {{L:intro_motivation}}; {{L:dsc_relation}}.

### R3.6
**Comment (summary; reviewer's item Introduction 3).** Why was liver or brain tumour segmentation not chosen?

**Response.** Not done; no tumour dataset was added. *Brain tumours:* BraTS is multimodal, outside the single-channel and RGB inputs of our pipeline. *Liver tumours:* LiTS is single-channel CT and would fit the pipeline; the reason it was not added is compute. Each further task multiplies a training programme that already comprised {{R:FIN.runs.total}} runs, and we judged that repeating each recipe 10 to {{R:CF.n}} times was worth more to the question than a fourth task. The consequence is a limit on scope, which the manuscript states: the results do not extend to other tasks, tumour segmentation among them. Small, irregular lesions are a setting in which boundary signals could behave differently, and we name additional datasets as future work.

**Location.** {{L:intro_compact}}; {{L:lim_scope}}; {{L:future}}; Supplementary S3.

### R3.7
**Comment (summary; reviewer's item Introduction 4).** Arrange the Introduction into motivation, background, literature, research gaps and contributions.

**Response.** Done. The Introduction now has labelled paragraphs: Motivation, Why a compact U-Net, Background, Research gaps and Contributions. The literature is reviewed in the Related work section that follows it.

**Location.** {{L:intro}}.

### R3.8
**Comment (summary; reviewer's item Introduction 5).** What is the research gap, and how do the methods address it?

**Response.** Done. The gap is how boundary distillation has been tested: against which reference it is credited, with which unit of replication, and with which controls. Each point is matched by a design element: the decoder-only reference, repeated training runs with the run as the unit, and controls that remove the band and the teacher's advantage.

**Location.** {{L:intro_gaps}}; {{L:intro_contrib}}.

### R3.9
**Comment (summary; reviewer's item Introduction 6).** Discuss the selection of the BDS-Lite components in detail.

**Response.** Done. Each component is described with its purpose, its equation and how its hyperparameters were set, and the framework subsection assigns each to a route of the boundary signal. A new Discussion subsection reports what the diagnostics of the trained networks show about the complete recipe: distillation shaped the projected feature but left the predicting feature unchanged, and the gate learned a nearly constant scale; it does not assign the ISIC2018 deficit to one component.

**Location.** {{L:routes}}; {{L:arms}}; {{L:disc_bds}}.

### R3.10
**Comment (summary; reviewer's item Introduction 7).** Cite recent references on medical image segmentation, e.g. doi:10.1002/ima.70424.

**Response.** Done. We cite the suggested article (Raj et al., 2026) in the Introduction as an example of the larger transformer-based models used for tumour segmentation: for liver tumours in CT, it pairs a vision-transformer classifier that flags slices containing tumour with a Swin-transformer U-Net that segments it. We checked this description against the article's abstract. The next sentence uses this trend to motivate the study of compact networks. We judged the article relevant for that purpose and did not use it as a comparison method. (See also R3.20.)

**Location.** {{L:raj}}.

### R3.11
**Comment (summary; reviewer's item Introduction 8).** Show a few images from the datasets, with annotated boundaries.

**Response.** Partly done. Figure 3 now shows three official ACDC test slices with the MR image, the annotated reference, the predictions of D and D+KD-band and a change map; the slices were chosen by a rule fixed before rendering. ACDC is distributed under CC BY-NC-SA 4.0, and the figure is marked with that licence. ISIC2018 is shown as reference and predicted contours without the photographs (Supplementary S12), because the ISIC2018 Task 1 annotations are released under CC-0 whereas each photograph carries the licence chosen by its contributor, which we have not verified image by image. Synapse images are not shown, because we could not establish from its access terms that images may be republished (Supplementary S16). If the Editor can confirm that reproduction of these benchmark images is acceptable to the journal, we will add them.

**Location.** Figure 3; {{L:res_qual}}; Supplementary S12, S16.

### R3.12
**Comment (summary; reviewer's item 9).** What gain is attributable to each component over standard backbones such as nnU-Net, Swin UNETR or Attention U-Net?

**Response.** Partly done. *Per component, on our backbone:* Table 5 gives the contrasts. The boundary decoder: {{R:IN.dec.pooled.boundary_f1.est}} in pooled Boundary F1 (95% CI {{R:IN.dec.pooled.boundary_f1.text}}). Feature distillation and gate added to it: {{R:IN.nosurf.pooled.boundary_f1.est}} ({{R:IN.nosurf.pooled.boundary_f1.text}}). The surface loss: {{R:IN.s.nodec.pooled.boundary_f1.est}} ({{R:IN.s.nodec.pooled.boundary_f1.text}}) without the decoder and {{R:IN.s.dec.pooled.boundary_f1.est}} ({{R:IN.s.dec.pooled.boundary_f1.text}}) with it. Single-component estimates from an exploratory additive model over the eight cells are in Supplementary S13: pooled +0.001 (−0.008 to 0.010) for feature distillation and +0.002 (−0.007 to 0.012) for the gate. *Against other backbones:* Attention U-Net was trained under our protocol and is in Table 4. We did not train nnU-Net or Swin UNETR, and the reason is compute. An nnU-Net 2D configuration with an added auxiliary boundary decoder would isolate the boundary signal in the same way as our design, and we agree that it is a natural next step. The effects in question, however, are of the size of the run-to-run variation and needed 10 to {{R:CF.n}} runs per arm to resolve; repeating a full nnU-Net configuration that often for each arm was beyond our budget, and one run per arm would not resolve effects of this size. Swin UNETR is in addition a 3D model. We therefore cannot say what these components would add to those models, and the manuscript states that the results do not extend to them.

**Location.** Table 5; Table 4; {{L:intro_compact}}; {{L:lim_scope}}.

### R3.13
**Comment (summary; reviewer's item 10).** For 2D slices from 3D modalities, how is out-of-plane continuity preserved, and how does slice-based performance compare with 3D or 2.5D approaches?

**Response.** Not done as an experiment; answered directly here and stated in the manuscript. Out-of-plane continuity is not preserved: ACDC and Synapse volumes are processed as independent 2D slices, no component of any arm uses neighbouring slices, and scores are computed per slice and averaged within patient or case. We did not compare with native 3D or 2.5D models. Two pieces of existing evidence are in the supplement, neither a matched comparison: volume-level scores of stacked slice predictions (Supplementary S9), and a development run on a different preparation of ACDC in which three adjacent slices as input raised patient-level DSC by {{R:CTX01.dsc.low}} to {{R:CTX01.dsc.high}} (Supplementary S15). All arms share the slice-wise design, so the contrasts between arms are not affected by it, but whether boundary signals behave the same way in 3D models is open, and we list it as future work.

**Location.** {{L:preprocessing}}; {{L:lim_scope}}; {{L:future}}; Supplementary S9, S15.

### R3.14
**Comment (summary; reviewer's item 11).** Where targets occupy less than 1–2% of the volume, how does the loss prevent collapse to background?

**Response.** Answered directly; no new loss was added. Every arm uses cross-entropy plus a soft Dice term computed per foreground class and averaged over classes, so a small class contributes its own ratio regardless of its area; that is the only mechanism against collapse. We did not use focal Tversky or other reweighting losses, so that all arms share one segmentation loss; the boundary loss of Kervadec et al. is one of our reference methods. We do not claim that this prevents missed small structures. It does not: a missed structure scores approximately zero on DSC and Boundary F1, has no defined distance, and is excluded from HD95, which the Methods now state, and the Results now report missed-structure and false-alarm rates in the main text (Table 7): on Synapse about one structure in ten was missed in each arm of Table 7, and these errors account for much of the low per-slice scores on that dataset. Supplementary S8 gives the rates by structure size and an all-case analysis that penalises them. Most targets in our data are in this regime: of the reference structures in the test sets, 370 of 1,013 on the internal ACDC patients and 1,887 of 3,370 on Synapse cover less than 1% of the slice, and a further 386 and 675 cover 1–2% (Supplementary S8, Table S19; totals in S24). No arm collapsed to background. Among reference structures below 1% of the slice, the plain U-Net missed 13.8% on ACDC and 17.1% on Synapse, and the decoder-only arm 8.8% and 16.8% (Supplementary S8, rates by reference area). These are shares of a slice, not of a volume.

**Location.** {{L:segloss}}; {{L:empty}}; {{L:res_errors}}; Table 7; Supplementary S8.

### R3.15
**Comment (summary; reviewer's item 12).** BDS-Lite underperforms in most cases; justify the limitations.

**Response.** Done. We agree, and the revision reports it as a finding instead of defending the recipe. The Results give the estimates (Table 5), and the new Discussion subsection reports what the diagnostics of the trained networks show: the distillation loss was satisfied by the projection without a measurable change in the feature that makes the prediction; the gate learned a nearly constant per-channel scale; and gradient probes do not indicate a conflict between objectives. We also state what the data do not show: the ablation does not assign the ISIC2018 deficit to one component, and the probes do not establish a cause.

**Location.** {{L:res_innetwork}}; {{L:res_full}}; {{L:disc_bds}}; {{L:lim_precision}}.

### R3.16
**Comment (summary; reviewer's item 13).** No qualitative segmentation is compared with the ground truth.

**Response.** Partly done. See R3.11. Figure 3 compares the predictions of D and D+KD-band with the ground truth on three official ACDC test slices, selected at the 90th, 50th and 10th percentiles of the paired slice-level DSC difference, so that the third row shows a case in which distillation did worse. The figure is limited to the two arms of the replication, because BDS-Lite and the reference methods were not trained in that study and have no predictions for these patients. For ISIC2018, Supplementary S12 compares U, D, BDS-Lite and U-Net+GSL, and U, D and D+KD-band, with the reference contour, without photographs. Qualitative results for Synapse are missing for the licensing reason given in R3.11; structure-level error plots are in Supplementary S12.

**Location.** Figure 3; {{L:res_qual}}; Supplementary S12.

### R3.17
**Comment (summary; reviewer's item 14).** Statistical test results in tables, an error analysis and ablation studies are missing.

**Response.** Done. *Tests in tables:* Table 5 (ablation contrasts with intervals, fresh-seed estimates and Holm-adjusted p) and Table 6 (replication, controls and augmentation comparison). *Ablation:* see R1.5. *Error analysis:* the Results have a subsection on missed structures and false alarms with Table 7; Supplementary S8 reports errors by distance from the reference contour, spurious components, and missed structures and false alarms by structure size, and Supplementary S12 plots per-class DSC; the Results relate the missed-structure rates to the higher Boundary F1 and higher HD95 of the decoder-only arm on ACDC. The failure-case subsection of the submitted version described the withdrawn ISIC2018 models and was not carried over. The error analysis is descriptive and post hoc: the rates of missed structures and false alarms are reported per arm but were not tested between arms.

**Location.** Tables 5–7; {{L:res_errors}}; Supplementary S8.

### R3.18
**Comment (summary; reviewer's item 15).** Comparison with state-of-the-art methods on each dataset is mandatory.

**Response.** Partly done. See R1.4. Table 8 lists published results for each dataset with the protocol of each source, four reference methods were trained under our protocol, and Table 8 now includes the volume-level DSC of our U-Net so that part of the gap can be assigned to the scoring. We did not reproduce the published models under their own protocols. Our absolute scores are lower than the published ones, and the manuscript says so and gives the reasons (no pretraining, no augmentation in the matched protocol, per-slice scoring, fewer Synapse training cases).

**Location.** Table 8; {{L:res_published}}; Supplementary S7.

### R3.19
**Comment (summary).** Overall: the novelty and contributions of BDS-Lite are not established; improvements are inconsistent and modest; statistical robustness, component effectiveness and the choice of comparators need justification through an ablation.

**Response.** Done. We accept this assessment of the submitted version, and the ablation the reviewer asked for bears it out. The revised manuscript does not claim novelty or a consistent benefit for BDS-Lite. It reports an exploratory gain of the boundary decoder, what in-network distillation adds to it (an estimate whose interval includes zero), what an external teacher adds (a gain confirmed without augmentation, on one dataset, of unclear size under the prespecified rule), and how large all of these are relative to run-to-run variation. The choice of reference methods is justified in the Arms subsection.

**Location.** {{L:abstract}}; {{L:comparators}}; Tables 5–6; {{L:conclusions}}.

### R3.20
**Comment (summary).** Staff note: suggested references should be included only if the authors agree they are relevant.

**Response.** We included the one suggested reference (R3.10) because it illustrates the larger transformer-based models for tumour segmentation against which our Introduction positions compact networks. The three 2025 references added under R1.2 were chosen by us; no other reference was added at a reviewer's suggestion.

**Location.** {{L:raj}}.
