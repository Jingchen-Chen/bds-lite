<!-- Generated from letter_to_editor.template.md by analysis/revision/render_editor_letter.py; do not edit. -->
# Letter to the editor

**Manuscript:** PeerJ Computer Science CS-2026:06:142937, *When does boundary distillation help lightweight medical image segmentation?* (major revision)

Dear Dr. Saccomanno,

Thank you for your decision and for the reviewers' comments. We submit a major revision. The response letter (`response.md`) summarises every comment, numbered E.1–E.6 and R1.1–R3.20, and gives the line numbers of each change in the revised manuscript PDF. **[AUTHOR ITEM: regenerate the tracked-changes PDF from the final text before stating that it is included.]**

**What the revised paper shows**

We kept the title and the BDS-Lite study, and we now answer the title question from corrected and extended evidence. The revised paper is an evaluation study with four findings for a compact 2D U-Net:

- Run-to-run variation is as large as the effects under study: in the arms with ten or more runs, the standard deviation of Boundary F1 between retrainings of one recipe was 0.006 to 0.009 on ACDC and 0.019 to 0.026 on ISIC2018.
- BDS-Lite's in-network distillation added no detectable gain beyond a training-only boundary decoder (+0.009, 95% CI −0.001 to 0.019; three runs per arm on each dataset), and the complete recipe scored 0.032 lower in Boundary F1 than the decoder alone on ISIC2018, confirmed on fresh seeds.
- Distillation from a frozen external teacher raised Boundary F1 by 0.018 (95% CI 0.010 to 0.026) in a confirmatory replication on 50 held-out ACDC patients with 17 runs per arm, under an analysis plan fixed and timestamped before the test patients were used; under the prespecified rule this is a gain of unclear size, because the unrounded lower bound, 0.0098, lies just below the preset smallest effect of interest, 0.01 (Supplementary S18). In exploratory controls, whole-image distillation and a compact teacher that scored below the reference also gave gains, a pattern consistent with a regularizing effect.
- On the same 50 ACDC patients, flip-and-rotation augmentation of the reference raised Boundary F1 by +0.067 (0.050 to 0.084; exploratory, not prespecified), several times the distillation gain, and with augmentation in both arms a distillation gain was not detected (−0.004, −0.012 to 0.004).

**Why the conclusions changed**

The submission concluded that BDS-Lite helped most on the single-contour ISIC2018 task. That conclusion rested on an ISIC2018 evaluation that largely used the models' own training images. It is withdrawn in the manuscript (Methods, Study design, "Relation to the first version") and in the response letter, with the audit in Supplementary S4, and all models are now evaluated on the official ISIC2018 test set. The ablation requested by the reviewers then made the training-only boundary decoder of BDS-Lite the reference against which its distillation, and an external-teacher variant, were tested.

**Major changes**

1. Corrected ISIC2018 evaluation and a retrained pipeline for every arm.
2. An eight-cell ablation on all three datasets; ten runs per arm for the central ISIC2018 arms, with confirmation on fresh seeds; four reference methods under one protocol.
3. An external-teacher distillation design, explored on three datasets and confirmed, as a gain of unclear size under the prespecified rule, in a replication with a prespecified analysis plan on held-out ACDC patients, with controls and an augmentation comparison on the same patients.
4. The training run as the unit of replication, with intervals, multiplicity status and a prespecified threshold for every contrast; absolute scores, missed-structure and false-alarm rates, qualitative examples and the resource profile in the main text. The main text has 6,505 words of prose against 5,761 in the submission.

**Disclosures**

- **Selection of ACDC.** ACDC was chosen for the replication after the exploratory three-dataset results (Holm-adjusted p: ACDC 0.088, ISIC2018 0.496, Synapse 0.500). The held-out patients remove test-set reuse, but not this selection.
- **Amendment.** The prespecified integrity check could not pass because of a defect in how sources were paired (74 files listed in the plan, 61 recorded by the trainer). Before any test data were accessed, we adopted a stricter replacement check; hypotheses, endpoint, estimator and decision rules are unchanged (Supplementary S18).
- **Timestamp.** The analysis plan was fixed in a local commit and timestamped with OpenTimestamps; the Bitcoin attestation shows that it existed before the first test-set access, not that it predates the start of training.
- **Controls.** The three controls reuse the held-out ACDC patients after the replication had been analysed, so none of their comparisons is confirmatory. Their analysis plan was fixed before training; it has no external timestamp.
- **Analysis plan.** The replication's analysis plan is an internal, timestamped document, not an entry in a public registry.
- **Figure 3.** It reproduces three ACDC test slices under the dataset's CC BY-NC-SA 4.0 licence.
- **Reproducibility.** The training script of the submitted version is not preserved, so its training cannot be repeated; every result in the revised manuscript comes from retrained models. The per-case results, analysis scripts, protocols and analysis plans are provided, and the reported statistics can be recomputed from them; the datasets, model checkpoints, teacher caches and prediction maps are not included.

The code, configuration files, split manifests, analysis scripts and the evidence archives are available at https://github.com/Jingchen-Chen/bds-lite, as given in the data statement of the submission.

Sincerely,

Jingchen Chen, Chenxiang Hong, Yao Wu, and Tao Yang (corresponding author)

