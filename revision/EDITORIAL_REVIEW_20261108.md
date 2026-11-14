# PeerJ editorial review, 8 November 2026

The revised answer is specific and positive: external-teacher boundary-band distillation added to the decoder-only reference on held-out ACDC patients. The pre-registered replication found Boundary F1 +0.0178, 95% CI [+0.0098, +0.0258]. The interval establishes a positive effect and crosses the 0.01 SESOI. Success and the registered size qualification are both preserved. BDS-Lite's in-network transfer did not establish an added gain; the two designs were not compared head to head.

The source of the reviews is `revision/reviewer.md`. `peerj-reviewing-142937-v0.pdf` is the original submission, rather than a PDF of the decision letter. The active response is `response_to_reviewers.tex` / `.pdf`; `response_letter_v2.*` is historical.

## Narrative and scientific corrections

- The abstract, contribution list, Discussion, Conclusions, response and editor letter state the positive ACDC finding directly. Repeated “only condition” wording is removed. The Results retain claim-boundary A3's literal registered wording and explain what its size qualification means.
- Positive direction is distinct from practical magnitude. The lower bound is printed to four decimals; neither secondary results nor the paired sensitivity changes the primary decision. Descriptive DSC, IoU and ASSD results remain accompanied by the non-established HD95 result.
- ACDC's post hoc selection, overlapping arm distributions, preprocessing reconstruction, integrity amendment and reporting deviations, timestamp scope, and limits on teacher and dataset inference remain disclosed. No full-image KD control or new evidence is implied.
- The Methods now distinguish pooled within-cell run variance/residual degrees of freedom in the three-run ablations from Welch variance/degrees of freedom in the ISIC2018 extension and stages 2–3. This corrects an inaccurate description; neither estimator nor any frozen output changed.
- The boundary map's BCE/Dice objective acts across the map. A boundary-shaped target does not restrict that objective to a band. The mechanistic rationale now makes that distinction and identifies the feature probes as post hoc observations, not a causal explanation of an unresolved accuracy contrast.
- The decoder-only architecture caption now says “no active gate,” consistent with the inert gate in its configuration. Availability and AI-use wording refer to a prepared revision release and accompanying scripts, without implying that unreleased changes are already public. The AI tool list and responsibility statement remain intact.

## Decisions on every original main figure and table

Old numbers refer to the task's starting manuscript. New main numbers follow the compiled manuscript; supplementary table numbers are obtained from LaTeX, including longtable counters.

| Original item | Decision and new location | Contribution and reason |
|---|---|---|
| Figure 1, architecture | Retain, Figure 1 | Necessary to understand the train/inference paths, gate, and decoder-only reference; answers E-2, R2-4/5 and R3-I6. |
| Figure 2, synthetic targets | Move to S3, Figure S1 | Explains target construction but does not answer the empirical question. Retained for R3-I8; the response now points to it explicitly. |
| Figure 3, central forest plot | Move to S13, Figure S6 | The main pooled estimates are already in Table 2. Retain dataset uncertainty and ten-run contrasts together in the supplementary vector plot. |
| Figure 4, KD confirmation | Retain, Figure 3 | Central positive evidence: run spread, overlap and exploratory versus held-out intervals. Does not imply a cross-dataset confirmation. |
| Figure 5, qualitative contours | Retain, Figure 2 | Explicitly requested by R3-R13. Shows deterministic examples against the reference, including distant false positives and negative examples. |
| Table 1, datasets | Move to S3, Table S2 | Supports task rationale, structural descriptors and the first-order argument; main text retains explicit S3 pointers. |
| Table 2, studies | Retain, Table 1 | Three stages, arms, seed counts and test populations; the ten-run extension is a caption note within stage 1. Methods retains the chronology and registration disclosures. |
| Table 3, ten-run contrasts | Move to S13, Table S34 | Full fresh-seed Holm evidence remains accessible and cited, with the central deficit still stated quantitatively in Results. |
| Table 4, central contrasts | Compact, Table 2; full version S13 | Pooled Boundary F1, DSC and HD95 intervals remain in the main table; interpretation is in Results and full readings remain in S13. Per-dataset detail moves to S13; Boundary F1 detail is also in Figure S6. |
| Table 5, KD contrasts | Move to S18 | Figure 3 and Results retain the central comparisons; S18 supplies exact intervals, run counts, raw p-values and exploratory Holm correction without a redundant main table. |
| Table 6, all arms | Move to S7, Table S17 | Dense supporting comparator/ablation evidence; no test or arm removed. Main Results retain the comparator findings and explicit S7 pointer. |
| Table 7, published context | Rebuild as Table 3; adaptations/history in S7 | The same six core matched models on all datasets; every adapted LB-UNet result remains in S7 irrespective of performance. Favorable ISIC2018 and unfavorable multiclass results remain explicit in Results. Nine mean-only numerical winners; published LB-UNet and source HD nomenclature remain in panel B. All values/protocols audited; no cross-panel ranking. |
| Table 8, deployment cost | Simplify, Table 4; full profile S19 | Main table retains four models (U-Net, deployed/gate-collapsed BDS-Lite, Attention U-Net), parameters, FLOPs and GPU batch-1/16 latency to quantify gate overhead. S19 retains the complete unchanged profile, including all adapted LB-UNet measurements, CPU timings, batch-8 GPU latency and kernel counts. |
| Table 9, answer | Remove redundant summary | Discussion now answers the title in prose, with the ACDC interval and unresolved in-network contrast. Results and S13/S18 retain the full evidence; Methods and S18 retain the limits on cross-study comparison. |

The main manuscript therefore contains three figures and four tables. Separate submission exports and the response's float names follow this numbering. The source figures and complete numerical tables are preserved. Caption-based pointers read the actual compiled labels rather than assuming table counters, since supplementary longtables also advance those counters.

## Evidence and validation scope

`analysis/revision/verify_editorial_numbers.py` independently aggregates the existing per-case CSVs and reimplements the relevant variance, interval and Holm calculations using pandas, NumPy and SciPy. Its only use of the production statistics module is path resolution. It verifies 572 quantities at absolute tolerance 1e-10: ACDC primary/secondary metrics and per-structure contrasts; all ISIC2018 ten-run/fresh-seed contrasts and metric-wise Holm correction; central study-1 dataset/pooled contrasts; exploratory KD dataset contrasts and Holm correction. It found zero mismatches (maximum absolute difference 4.27e-14). This is central-evidence recalculation, not a new analysis or re-evaluation of predictions.

The output is `revision/validation/editorial_numerical_verification.csv`. Additional key/reference, reviewer-quotation, artifact-preservation and PDF checks are recorded in `revision/validation/editorial_consistency.json` and the required final audit report. No Experiment C test result was read, and no training, inference, release, push, message to the journal or submission was performed.

## Author decisions left open

1. Resolve the visible affiliation/present-address marker and decide whether the public earlier PDF is treated as a preprint requiring the correction note already proposed in the manuscript.
2. Approve the full revision, response and AI tool inventory with all co-authors; verify submission-system metadata.
3. Commit the author-owned Experiment C stop/deviation/queue records. They were present before this task and are not staged by this revision.
4. Decide and perform the release actions in `RELEASE_GUIDE.md`; mint a revision DOI and finalize availability. The earlier public repository's image/history issue remains the author's release decision. No release action is authorized here.
5. The claim boundary's header still says DRAFT. `AGENTS.md` explicitly records its approval and authority; this editorial pass does not amend that author-controlled decision record.

Research limitations remain substantive: a post hoc selected dataset, one compact 2D backbone without augmentation, one external teacher and no full-image KD control, reconstructed official-set preprocessing, and no verified physical-distance or cardiac-index interpretation. These limit practical and mechanistic conclusions; they do not negate the positive primary interval.

## Table follow-up, 9 November 2026

Table 1 now uses Ablation and comparisons, Seed confirmation, Exploratory KD, and ACDC confirmation; its test set is labeled Official ACDC test (50 patients). Pre-registration and post hoc ACDC selection remain explicit. Table 2 retains Metric, Contrast and Pooled Δ [95% CI] only. Its nine numerical rows are unchanged; Results explains the SESOI interpretation once, and S13 retains the full table and frozen readings. The caption and response references were synchronized. Only rendering and document checks were run; no research evaluation or frozen analysis was rerun.

## Figure 2 decision, 9 November 2026

| Option | Narrative benefit | Cost | Decision |
|---|---|---|---|
| Keep Table 2; move the full forest to S13 | Main text retains precise pooled estimates for Boundary F1, DSC and HD95; the full graphic supplies dataset uncertainty and ten-run context where detailed evidence is inspected | Main text no longer gives a graphical view of the in-network intervals; it points to S13 | Selected |
| Add a minimalist three-row pooled Boundary F1 plot | Makes position relative to zero and ±0.01 immediately visible | Repeats all three Boundary F1 rows of Table 2 and their Results interpretation, while still requiring the full supplementary plot for dataset and ten-run evidence | Not selected |

The first option removes a duplicate main-text presentation and gives the ACDC confirmation graphic the main visual emphasis its held-out primary result warrants. This does not remove or weaken the in-network evidence: all three pooled estimates remain in Table 2, the scoped ten-run deficit remains quantitative in Results, and the complete forest plot is Figure S6 in S13. The plot's PDF is byte-identical to the previous version, contains vector marks/text rather than raster images, and retains all intervals, sample-size distinctions and the ±0.01 band. The final main and supplementary pages were visually inspected. Main figure numbering and submission exports now run 1–3. No analysis or model evaluation was rerun.


## External-teacher presentation, 9 November 2026

The former main Table 3 is now in S18. Its generated numeric content is unchanged, including all cross-dataset Holm results and the independent primary endpoint. Main Figure 2 retains the ACDC run distribution and the four contrast intervals. Its vector layout uses a dedicated label column, aligned groups, consistent typography and a single pre-specified ±0.01 band. The qualitative figure remains Figure 3.

Results now develops “Cross-dataset exploration” and “Independent ACDC confirmation” as two consecutive findings. Methods explains dataset selection, registration, new patients and seeds, and conservative power planning. The abstract and literal primary Results passage retain the essential selection disclosure; repeated audit labels were removed from contributions, figure caption, discussion and summary-table evidence labels. The positive +0.0178 interval remains separate from uncertainty about exceeding the 0.01 SESOI. Holm correction, descriptive secondary endpoints, absent full-image control, one teacher, preprocessing limits, amendment and timestamp qualifications remain explicit in the submission package. No statistical estimate, frozen analysis or model evaluation changed.


## Three-stage structure, 9 November 2026

The Results now has exactly three main experimental headings: Stage 1: Ablation and Comparisons; Stage 2: Exploratory Knowledge Distillation; Stage 3: ACDC Confirmation. Stage 1 begins with the pooled three-run ablations, followed by comparators, errors/qualitative evidence, run-to-run uncertainty and resource cost. Its ISIC2018 ten-run findings retain the decoder's supported/unconfirmed status and the scoped confirmed deficits of full BDS-Lite and GSL relative to the decoder. Full seed means, fresh-seed intervals, Holm corrections and pipeline/tolerance sensitivities remain in S13/S15. A concise main uncertainty paragraph replaces a longer historical example; its underlying fixed-model and pipeline results remain in S5/S14. External-teacher training cost is reported with stage 3.

Table 1 has three rows and four purpose-oriented columns; protocol timing is explained in Methods. The four-arm ISIC2018 extension is explicitly attached to stage 1, while stage 2 retains three students against ten ISIC2018 decoder runs. No count is relabeled. The Discussion summarizes the stage-specific findings in prose. Figure order follows the narrative: architecture 1, qualitative 2, external-teacher contrasts 3. Separate exports and response references follow compiled labels.

Methods explicitly states that the stages group work performed under separate protocols, not a single prospectively planned sequence. It retains the earlier exploratory information available to the matched-ablation plan, the frozen expansion protocol, post hoc selection of ACDC, pre-registration, and independently verified timestamp scope. The supplement maps historical studies 1–2 to stage 1, study 3 to stage 2 and study 4 to stage 3; immutable protocols keep their original identifiers. The literal ACDC primary Results paragraph, numerical conclusions, SESOI framework and scientific limitations are unchanged.


## Discussion summary-table removal, 9 November 2026

Removed the former main Table 5 (“When did boundary distillation help?”) rather than relocating another duplicate. The opening Discussion paragraph states the +0.0178 [95% CI +0.0098, +0.0258] ACDC external-teacher gain directly, distinguishes positive direction from exceeding the 0.01 SESOI, and contrasts this with the unresolved in-network addition. It retains the confirmed full-recipe deficit on ISIC2018 in the retrained set and the reason it cannot be attributed to distillation. Results preserves the exact contrasts, fresh-seed Holm decisions and pipeline sensitivity; S13/S18 retains all statistics. S18 now describes separate contrasts in prose instead of referring to deleted table rows. Response pointers now target Discussion and the primary Results. Three figures and four tables remain.


## Matched comparison and balanced KD, 9 November 2026

Table 3 now reports mean ± sample SD over seeds 1–3 for U-Net, decoder-only, full BDS-Lite, GSL, Boundary Loss, Attention U-Net and LB-UNet across all three datasets. Raw class-level files reproduce 189 per-run metric means and 126 aggregate mean/SD values, maximum reconciliation error 3.55e-15. Positive HD95 is an absolute distance, not a sign-reversed effect. The image/slice class-macro aggregation matches the existing descriptive supplementary table; the run-level patient/case contrasts remain separate. Published rows have DSC on a 0–1 scale and unavailable metrics as dashes. Their original PDF values and hashes are recorded; ISIC random splits/pixel aggregation and ACDC/Synapse volume aggregation prevent a matched ranking. Removed the prior unsupported explanation of an accuracy gap as expected from backbone size. Swin-Unet (2D) is explicitly distinguished from Swin UNETR (3D, not run). Earlier-pipeline/volume and challenge-Jaccard context remain in S7.

The author then explicitly requested a balanced three-versus-three exploratory presentation. The original seed-matched ISIC contrast was already planned as a descriptive sensitivity; promoting it and recomputing a balanced three-dataset Holm family are post hoc. The new independent calculation is isolated in revision/kd_posthoc_balanced/: ISIC +0.0523 [−0.0377, +0.1423], p=0.1467, Holm p=0.2934; ACDC/Synapse estimates and decisions are unchanged. Both balanced and original families establish no gain after Holm. The original frozen 3-versus-10 analysis remains intact and is reported as sensitivity in S17; it was the analysis available when ACDC was selected. New KDB bindings are separate from all frozen KD/CF bindings.

Equal counts are acceptable for exploratory presentation but are not required by Welch inference, omit seven reference runs and leave only three runs for variance estimation. The retained unpaired estimator does not model within-seed covariance; Holm adjustment does not make a post hoc display confirmatory. The plot's upper limit was extended to +0.16 so the new +0.1423 CI endpoint remains visible. All 17+17 confirmation run means and its primary analysis are unchanged. The author request authorizes this display change; the approved boundary's original A2 evidence remains a historical frozen result, not a relabeled value.


## Final table refinement and LB-UNet audit, 9 November 2026

Bolding marks only the nine numerical-best means within panel A, based on unrounded results; SDs and every numeric value are unchanged. Boundary Loss has ACDC DSC 0.7620162122878765 versus GSL 0.7619555011412109, so their identical printed 0.7620 means are not a tie. Published rows have no numerical bolding. The main table retains 19 matched rows; binary-domain LB-UNet remains on ISIC2018, while its ACDC/Synapse adaptations are preserved at full printed precision in S7 and their unfavorable contrasts remain in Results. Scope rather than outcome determines placement.

Original papers and pinned released evaluation code were inspected. EGE-UNet/LB-UNet cited tables provide DSC/mIoU, no boundary F1/distance; ACDC TransUNet/Swin-Unet cited results provide DSC only. Synapse papers call their metric HD. Both released evaluators call MedPy HD95 on reconstructed volumes without spacing; Swin-Unet averages test cases and classes. The main and active supplementary published displays therefore retain source HD labels, with the implementation evidence and unit/empty-mask differences in Methods/S7. We do not assert exact historical reproduction or equate these distances with resized 2D pixel HD95.

Static CPU inspection of pinned upstream LB-UNet and strict loading of all nine existing checkpoints verified the graph shapes and recorded source hash. The upstream one-output RGB graph already has 55,865 learnables; the two-output graph has 55,874 and the wrapper adds none. Conv2d/BatchNorm2d-only accounting gives 38,489 but excludes learned spatial tensors (15,872), custom LayerNorm (992) and GroupNorm (512), totaling 17,376 omitted parameters. This can explain a rounded 0.038M under partial accounting, but the paper's actual counting convention is unverified. The text reports that uncertainty and labels the implementation adapted: class logits, loss, morphological targets, resolution/schedule and augmentation differ. No model source, configuration, checkpoint or result changed, and no forward pass or training experiment was run.

## Consistent Table 3 reporting scope, 9 November 2026

The author's latest instruction supersedes the prior 19-row display: Section A now has the same six models on each dataset (18 rows). Adapted LB-UNet is supplementary on every dataset irrespective of performance. The published LB-UNet reference in Section B is byte-identical. Main Results retains the favorable ISIC2018 DSC/HD95 contrasts and the unfavorable multiclass contrasts, and cites S7. Every original S7 metric and contrast table is byte-identical; a compact three-dataset summary preserves all removed values at the original main-table precision. The caption, provenance description, claim-evidence row and responses R1-3/R3-R15 explain this reporting scope.

Only numerical highlighting changes among retained cells: ISIC2018 DSC goes to U-Net+GSL (unrounded 0.8419486263567096), Boundary F1 remains U-Net+GSL (0.38383900426439865), and HD95 goes to U-Net+Boundary Loss (26.929965738173717). ACDC and Synapse highlights are unchanged, including the unrounded ACDC DSC distinction. Bold remains confined to nine means, with SDs unbolded and no significance implication. All experimental values, selections and statistical analyses are unchanged.

## Deployment table scope, 9 November 2026

Table 4 now focuses on BDS-Lite's inference overhead: U-Net, BDS-Lite (deployed), BDS-Lite (gate-collapsed), and Attention U-Net. It retains deployed parameters, FLOPs and actual fp32 GPU latency at batch sizes 1 and 16; all retained values are unchanged. CPU latency moves to S19 for a cleaner main display. The adapted LB-UNet row is supplementary consistently with the main matched-comparison scope, irrespective of its measurements.

S19's full table is byte-identical and still contains LB-UNet's 55,874 parameters, 0.144 GFLOPs, GPU latencies 3.56/3.58/3.91 ms at batches 1/8/16, CPU timings 6.5/5.4 ms, and 462 CUDA kernels. The original source CSV is unchanged. The Results retains BDS-Lite's +8% and +15% GPU latency versus U-Net, 104 to 115 kernels, and the gate-collapsed timings. Thus the main table continues to show measured latency overhead alongside the small FLOP increment. No profiling or new experiment was run. Captions and the reviewer efficiency response distinguish the compact main scope from the complete supplementary evidence.

The unchanged source CSV also contains a Synapse LB-UNet profile and fp32/AMP timing IQRs. Added Table S55 in S19 to render all 20 numeric fields for each of the two recorded LB-UNet profiles, including both parameter counts, both FLOP counters, CPU timings, GPU batch-1/8/16 medians and IQRs for fp32 and AMP, and CUDA counts/times. Original Table S54 remains byte-identical. This addition exposes existing measurements; it does not rerun the profiler or change any result.
