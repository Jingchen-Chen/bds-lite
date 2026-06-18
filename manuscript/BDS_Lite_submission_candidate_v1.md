# When Does Boundary Distillation Help Lightweight Medical Image Segmentation?

## A Controlled Study of Applicability, Failure Modes, and Inference Decoupling

**Draft status:** Phase 17 submission-candidate manuscript, 10 June 2026.
Integrates the Phase 16 source-corrected numbers, the matched three-seed GSL
comparison, the Phase 16 matched Figure 2, and the restricted efficiency
wording. Author metadata, affiliations, funding, and final reference
verification remain human tasks (see
`../human_tasks_before_submission.md`).

## Abstract

Boundary-focused training is often judged mainly by average overlap, although
local contour agreement and surface distance can move in different
directions. We study BDS-Lite as a controlled training-time
boundary-distillation intervention on a compact U-Net. The protocol covers
ISIC2018, ACDC, and Synapse with three seeds, five complementary metrics,
case-aware analysis, a matched-protocol GSL comparison, and resource
profiling. On the locked 519-image ISIC2018 validation partition, mean
Boundary F1 changed from 0.5520 to 0.5597, HD95 from 18.5128 to 17.4435
pixels, and ASSD from 5.9192 to 5.6753 pixels, while DSC remained nearly
unchanged (0.8830 versus 0.8828). ACDC showed no stable advantage. On
Synapse, mean DSC and Boundary F1 were slightly higher, whereas HD95 and ASSD
were worse. No Holm-corrected seed-level superiority statement is supported
at \(n=3\), and cluster-aware analyses did not establish favorable effects on
ACDC or Synapse. In a matched three-seed comparison under the same protocol,
U-Net+GSL means are mixed across datasets in both directions, so GSL remains a
strong geometry-aware comparator and no superiority claim is made in either
direction. The auxiliary boundary decoder can be removed from the studied
inference graph, which retained 1,928,114 parameters versus 1,927,042 for
U-Net. Boundary distillation therefore showed conditional, task-dependent
behavior rather than a uniform ranking advantage.

**Keywords:** medical image segmentation; boundary distillation; lightweight
networks; boundary metrics; surface distance; auxiliary supervision;
statistical evaluation

## 1. Introduction

Medical image segmentation supports quantitative analysis of lesions,
cardiac structures, and abdominal organs. U-Net remains a common compact
encoder-decoder baseline [1], while attention, nested skip pathways,
automated configuration, and transformer encoders have expanded the design
space [2--7]. For small 2D models, however, transparent supervision and
evaluation remain as important as architectural capacity.

Most segmentation reports prioritize DSC or IoU. These region measures are
essential, but they can conceal contour displacement, missed small
structures, and remote false-positive components. Boundary F1 measures local
contour agreement; HD95 and ASSD summarize different aspects of surface
distance. A prediction can therefore preserve overlap while changing its
boundary quality, or gain overlap while worsening a distance metric. Current
evaluation guidance supports reporting complementary metric families and
making the independent statistical unit explicit [8,9].

Boundary losses, surface objectives, and auxiliary contour branches introduce
geometric information during training [10--15]. Distillation and auxiliary
supervision can also transfer intermediate structure without retaining every
training component at inference [16--19]. What is less often examined is the
scope of the resulting effect: whether it is broadly distributed across
cases, stable across tasks, consistent across metric families, and supported
after accounting for repeated slices and multiple comparisons.

We address that gap through a controlled study of BDS-Lite on a compact
U-Net. The study asks: **When does training-time boundary distillation help
lightweight 2D segmentation, when do boundary and distance metrics disagree,
and which auxiliary structure can be removed at inference?** This framing
treats mixed results as evidence about applicability rather than as a basis
for a universal method-ranking claim.

Our contributions are:

1. We study BDS-Lite as a boundary-distillation intervention for lightweight
   2D medical image segmentation.
2. We jointly evaluate overlap, boundary, distance, resource, and statistical
   behavior on ISIC2018, ACDC, and Synapse using three seeds, and we add a
   matched-protocol GSL comparison.
3. We analyze case distributions, target subgroups, organ classes, outlier
   sensitivity, and automatically selected failure cases to identify
   applicability limits.
4. We document train/deploy decoupling: the auxiliary boundary decoder is
   removed from the studied inference graph, which remains close to the
   backbone resource profile.

## 2. Related Work

### 2.1 Lightweight medical image segmentation

U-Net [1], Attention U-Net [3], UNet++ [4], and nnU-Net [2] illustrate the
continuing importance of encoder-decoder designs. Transformer-based variants,
including TransUNet, Swin-Unet, SegFormer, and MISSFormer [5--7,20], broaden
the design space but do not remove the need for compact, reproducible
baselines. EGE-UNet is an example of a compact edge-guided design [21].
Because training schedules and implementations materially affect comparisons,
this study treats protocol alignment as a prerequisite for method ranking.

### 2.2 Boundary-aware and surface-aware learning

Boundary loss uses distance information to address regional imbalance [10].
Hausdorff-oriented objectives target adverse surface errors [11], active
boundary loss promotes contour alignment [12], and GSL provides a strong
geometry-aware baseline [13]. Other approaches combine region, contour, and
shape supervision [14,15]. These methods motivate evaluation beyond DSC:
boundary-local quality and long-range surface errors represent different
failure modes.

### 2.3 Distillation and auxiliary supervision

Knowledge distillation transfers information from an auxiliary predictor or
teacher into a target network [16]. Feature-hint and structured distillation
approaches extend this principle to intermediate representations and dense
prediction [17--19]. BDS-Lite uses a training-time boundary path as the
auxiliary signal rather than claiming a larger external teacher.

### 2.4 Evaluation and statistical testing

Metric definitions can alter conclusions [8,9,22]. HD95 and ASSD require
explicit empty-mask handling, while Boundary F1 depends on a tolerance
definition. Repeated slices from the same volume are correlated; tests at the
slice level therefore cannot be interpreted as independent patient evidence.
Multiple comparisons also require correction, for which Holm's procedure is
used here [23].

### 2.5 Inference-time decoupling

Auxiliary heads can enrich training while being discarded at inference.
The practical question is whether the retained graph and its outputs remain
stable after removal. We evaluate this implementation-specific property with
gate-removal metrics and parameter/FLOP profiling, without extending the
claim to a particular clinical or hardware environment.

## 3. Method

### 3.1 Controlled intervention

Let a U-Net backbone map image \(x\) to segmentation logits
\(z_s=f_\theta(x)\). BDS-Lite retains this encoder-decoder path and adds a
boundary decoder \(h_\phi\) during training. The boundary path receives
multi-scale encoder features and predicts boundary logits \(z_b\) and an
intermediate boundary feature \(q_b\). A projection of the segmentation
feature, \(q_s\), is aligned with the detached boundary feature. This makes
the boundary path a training signal rather than a required deploy component.

### 3.2 Boundary supervision and feature distillation

Boundary targets are derived from the segmentation labels using the saved
dataset configurations. The boundary objective combines binary
cross-entropy and binary Dice. Feature transfer uses a Smooth L1 discrepancy:

\[
\mathcal{L}_{distill}=\operatorname{SmoothL1}(q_s,\operatorname{stopgrad}(q_b)).
\]

The segmentation objective combines cross-entropy and soft Dice. A
signed-distance surface term adds geometric sensitivity. The implemented
training objective is:

\[
\mathcal{L}=\lambda_{ce}\mathcal{L}_{ce}
+\lambda_{dice}\mathcal{L}_{dice}
+\lambda_b\mathcal{L}_{boundary}
+\lambda_d\mathcal{L}_{distill}
+\lambda_s\mathcal{L}_{surface}.
\]

The selected configurations use \(\lambda_{ce}=\lambda_{dice}=\lambda_s=1\)
and \(\lambda_b=\lambda_d=0.05\).
Exact target-generation and empty-mask rules are implementation-specific and
must be read with the archived configuration and metric artifacts.

### 3.3 Bounded boundary gate

The training graph can form a gate from a boundary proxy:

\[
g=\sigma(Wq_b),\qquad
\tilde q_s=q_s\{1+\alpha(g-0.5)\},
\]

where the saved configurations use \(\alpha=0.25\). This bounded residual
modulation limits feature scaling. The gate and boundary path are studied as
training auxiliaries; gate-removal evaluation checks whether retaining them
at inference is necessary in the studied checkpoints.

### 3.4 Deploy graph

At deployment, the boundary decoder and its training outputs are not required.
The retained segmentation graph contains the backbone and a small projection
associated with the trained model. Profiling reports both total training-graph
parameters and retained deploy parameters. This is graph decoupling in the
studied implementation, not evidence for a particular hardware setting.

## 4. Experimental Protocol

### 4.1 Datasets and splits

ISIC2018 is a binary skin-lesion task [24]. The artifact family used for the
locked main analysis contains 2,075 training and 519 validation images in
`data/splits/isic2018/`; no test partition exists in that manifest family.
The re-evaluation script explicitly maps ISIC2018 to `val`, and all 519 case
identifiers in the Phase 12 analysis match the validation manifest. Several
legacy aggregate filenames retain a `_test` suffix; throughout this
manuscript those values are described as validation results.

ACDC is a cardiac MRI task [25] with 70/10/20 train/validation/test patients,
corresponding to 1,350/186/366 processed slices. Synapse is an abdominal
multi-organ CT task [26] with 14/4/12 train/validation/test cases,
corresponding to 1,738/473/1,568 processed slices. The locked manifest family
uses seed 2026 and subject-disjoint checks where subject identifiers exist.
Inputs are processed at 224 x 224 with z-score normalization in the common
configuration.

### 4.2 Methods and training

The principal locked comparison is U-Net versus BDS-Lite across three seeds.
GSL is included as a matched comparator. The Phase 16 matched GSL rerun
trained U-Net+GSL under the same locked seeds, split manifests, 224 x 224
input size, batch sizes, AMP policy, 150-epoch schedule, and
maximum-validation-DSC checkpoint rule used for the U-Net/BDS-Lite family;
test data were not used for model or checkpoint selection. We therefore report
U-Net, BDS-Lite, and U-Net+GSL as a single matched three-seed family rather
than as separate evaluation families.

To make the GSL training and profiling feasible on the available GPU, the
matched rerun and the controlled profile used an exact CPU Euclidean distance
transform in place of the quadratic-memory pairwise-distance fallback. This
preserves the evaluated loss values while keeping the studied protocol
tractable. Three seed means are descriptive; no significance is claimed from
three seed means. Other baselines are excluded from primary ranking when
protocol comparability is incomplete.

### 4.3 Metrics

We report DSC, IoU, HD95, ASSD, and Boundary F1. Higher DSC, IoU, and
Boundary F1 are favorable; lower HD95 and ASSD are favorable. The corrected
Boundary F1 implementation assigns zero when non-overlapping boundaries fall
outside tolerance. Empty-mask and per-class handling follow the saved
evaluation code and artifacts.

### 4.4 Statistical analysis

Three seed-level means are reported as descriptive replication. Exact
two-sided Wilcoxon tests at \(n=3\) have insufficient resolution for a
Holm-corrected superiority statement. We therefore average paired deltas
across seeds per case and use cluster-aware units: images for ISIC2018,
patients for ACDC, and volumes/cases for Synapse. We report cluster bootstrap
95% intervals, paired signed-rank tests with Holm correction, sign effect
sizes, and leave-one-cluster-out sensitivity. Slice-level summaries remain
descriptive for volumetric datasets.

### 4.5 Case, subgroup, and failure analyses

Per-case deltas are oriented so positive values indicate favorable movement.
Subgroups use target area and boundary complexity where derivable. ACDC and
Synapse include class summaries. Failure panels were selected by deterministic
metric criteria before visual review, including favorable cases, unfavorable
cases, similar-DSC/different-boundary cases, and distance failures. They are
illustrative rather than prevalence estimates.

### 4.6 Resource profiling and reproducibility

The project records parameter counts, FLOPs, throughput, latency, and peak
memory in one local environment. A controlled no-update profile on one
RTX 5060 Laptop GPU also records method-specific training-step time and peak
training memory. Throughput and step-time differences are not treated as
hardware-independent effects. Locked splits, configs, seed artifacts,
prediction arrays, table scripts, and audit reports are retained in the
repository. Table 1 summarizes the protocol; source paths accompany every
table row.

## 5. Results

### 5.1 Main three-dataset results

Table 2 and Figure 2 report the matched three-seed U-Net, BDS-Lite, and
U-Net+GSL family. On ISIC2018, BDS-Lite changed mean Boundary F1 from 0.5520
to 0.5597, HD95 from 18.5128 to 17.4435, and ASSD from 5.9192 to 5.6753
pixels. DSC was nearly unchanged (0.8830 for U-Net and 0.8828 for BDS-Lite),
while IoU changed from 0.8091 to 0.8112.

ACDC was unfavorable overall: DSC changed from 0.7893 to 0.7847, HD95 from
6.8762 to 7.6230, ASSD from 2.2080 to 2.4874, and Boundary F1 from 0.8313 to
0.8267. On Synapse, DSC changed from 0.8544 to 0.8573 and Boundary F1 from
0.8608 to 0.8632, but HD95 changed from 10.1985 to 10.3413 and ASSD from
3.7515 to 3.8280. Thus overlap/boundary means and distance means disagree.

In the matched GSL comparison the descriptive directions are mixed. On
ISIC2018 validation, U-Net+GSL had lower mean DSC and Boundary F1
(0.8662/0.5399) and higher HD95/ASSD (19.7954/6.6595) than both comparators.
On ACDC test, its means lay between or near the comparators (DSC 0.7856,
HD95 7.0143, ASSD 2.0902, Boundary F1 0.8277). On Synapse test, U-Net+GSL had
the highest mean DSC and Boundary F1 (0.8640/0.8701) and the lowest mean HD95
(10.1285), but a higher mean ASSD (3.8567) than both comparators. GSL is
therefore a strong geometry-aware comparator, and the matched three-seed means
do not support a superiority statement in either direction.

### 5.2 Per-case distribution analysis

Favorable-case fractions on ISIC2018 were 56.1% for DSC, 56.8% for IoU,
56.9% for HD95, 57.8% for ASSD, and 56.3% for Boundary F1. The shifts were
therefore not confined to one or two cases, although their magnitude remained
small. Trimming 5% of each HD95 tail retained a favorable mean oriented shift
of 0.9505 pixels versus 1.0409 without trimming.

ACDC distributions were unstable and tail-sensitive. Synapse had favorable
fractions of 53.1% for DSC but only 44.8% for HD95 and 48.2% for ASSD. Its
HD95 oriented mean changed from -0.1003 to -0.2106 after 5% trimming, so the
unfavorable distance behavior cannot be dismissed as a few extreme cases.

Metric conflicts were common enough to matter. On ISIC2018, 12.4% of eligible
cases had absolute DSC change at most 0.005 and Boundary F1 gain above 0.01.
On Synapse, about 21% of cases combined favorable DSC with worse HD95 or ASSD.
These observations motivate reporting all three metric families.

### 5.3 Subgroup and organ-level analysis

Table 4 summarizes exploratory subgroups. On ISIC2018, the large-target
subgroup had mean DSC delta 0.0063 and Boundary F1 delta 0.0111, whereas the
small-target subgroup had -0.0121 and -0.0040. High-complexity boundaries had
a Boundary F1 delta of 0.0190 but a slightly unfavorable HD95-oriented delta
(-0.0202), again showing metric disagreement.

Class summaries were heterogeneous. In ACDC, the left ventricle had favorable
DSC and Boundary F1 means but unfavorable distance means; myocardium was
unfavorable across the reported summaries. In Synapse, liver was favorable in
the exploratory table, gallbladder was unfavorable, and stomach combined
slightly favorable DSC/Boundary F1 with worse distance. These slice-derived
class patterns are hypotheses for future validation, not independent
organ-level inference.

### 5.4 Statistical analysis

No seed-level comparison survived Holm correction; all corrected p-values in
the three-seed summary were 1.0. Cluster-aware results are in Table 3.
ISIC2018 showed small signed-rank distributional shifts for the five metrics,
but the bootstrap interval for mean DSC crossed zero. The mean oriented HD95
delta was 1.0409 pixels (95% bootstrap interval 0.2476 to 1.8092), and the
mean Boundary F1 delta was 0.0077 (0.0006 to 0.0146).

For ACDC (20 patients) and Synapse (12 cases), corrected cluster-level tests
did not establish favorable effects. ACDC mean distance deltas were
unfavorable, while Synapse intervals crossed zero with unfavorable mean HD95
and ASSD directions. These cluster-level results take precedence over
slice-level p-values.

### 5.5 Failure cases

Automatically selected panels reveal distinct mechanisms. ISIC case
`ISIC_0013140` shows favorable contour and distance changes, whereas
`ISIC_0016060` has a DSC change of -0.223 and a Boundary F1 change of -0.218,
with large distance degradation. ACDC case
`patient052_frame01_slice_5` combines a favorable DSC change with HD95 and
ASSD worsening, illustrating that added overlap can coexist with a remote or
misplaced contour. Synapse case `case0004_slice134` shows a marked distance
failure. The full set of 24 panels is supplied for transparency.

### 5.6 Resource profile and inference decoupling

The profiled U-Net contains 1,927,042 parameters and 15.970 billion FLOPs.
BDS-Lite contains 2,198,003 total training-graph parameters; after removing
the 269,889-parameter boundary branch, the retained graph contains 1,928,114
parameters and 16.079 billion FLOPs. This is 1,072 deploy parameters above
the profiled U-Net (approximately 0.056%) and about 0.68% more FLOPs.

Controlled no-update profiling on one RTX 5060 Laptop GPU (batch 8, 224 x 224,
AMP, no optimizer step) recorded mean training-step times of 39.62 ms for
U-Net, 57.17 ms for BDS-Lite, and 52.32 ms for U-Net+GSL, with peak training
allocations of 600.29 MB, 873.57 MB, and 600.29 MB respectively. Inference
latency on the same GPU was 1.93 ms for U-Net, 2.40 ms for BDS-Lite, and
1.97 ms for U-Net+GSL. These local measurements do not establish end-to-end
time-to-convergence or portability across hardware; the GSL CPU distance
transform is part of its timed step, so its local timing is implementation-
and hardware-specific.

Gate-removal changes were small in the recorded evaluation: ISIC2018 DSC
changed from 0.8607 to 0.8610, ACDC from 0.7847 to 0.7851, and Synapse from
0.8466 to 0.8449. The evidence supports removability in these checkpoints,
not a general throughput or real-world readiness statement.

## 6. Discussion

### 6.1 Conditional value of boundary distillation

The clearest favorable behavior occurs on ISIC2018. Lesion segmentation is a
binary task with one principal foreground contour, and the method's local
boundary signal aligns with that structure. Even there, the result is a small
boundary/distance shift rather than a DSC gain. This distinction is central:
the intervention changes which errors are emphasized, not necessarily the
average region score.

ACDC and Synapse expose limits. Cardiac slices contain changing anatomy,
small structures, and patient-level dependence. Synapse combines multiple
organs with different scales and boundary ambiguity. A shared boundary
objective can allocate capacity unevenly, and local contour agreement does
not prevent remote false positives or missed components that dominate HD95.
The low-to-moderate correlations between favorable Boundary F1 and distance
changes are consistent with these metrics measuring different error geometry.

### 6.2 GSL as a strong comparator

The matched GSL comparison shows that a simpler geometry-aware loss is
competitive across all three datasets and is stronger than BDS-Lite on several
Synapse means. This prevents any claim that BDS-Lite has the strongest
geometry-aware training objective. It also does not remove the value of the
controlled study. The contribution is mechanistic rather than a ranking:
BDS-Lite supplies an explicit training-only boundary path and an audited
train/deploy separation, while GSL demonstrates that a loss-only geometry
formulation can be equally or more effective on some tasks and weaker on
others. Because the matched three-seed means are mixed in both directions, a
final submission should present this comparison as evidence about when
auxiliary feature transfer is justified relative to loss-only supervision, not
as a winner.

### 6.3 Why distance metrics are essential

On Synapse, slightly favorable DSC and Boundary F1 means coexist with worse
HD95 and ASSD. Failure panels show plausible sources: disconnected
false-positive regions, missing small structures, and local contour changes
that do not address the farthest surfaces. Average DSC alone would hide this
trade-off. Conversely, HD95 can be tail-sensitive, so outlier and clustered
analyses are needed rather than relying on a single aggregate.

### 6.4 Implications and next method steps

The remaining project year should test prespecified, low-cost modifications:
class-weighted or adaptive boundary terms for heterogeneous organs, lower
boundary weights for ACDC, component-aware penalties for remote false
positives, and matched training-time profiling. These experiments should use
validation data for selection and retain all seeds. A change should enter the
paper only if it strengthens cross-seed stability or clarifies the mechanism,
not because one seed is favorable.

## 7. Limitations

The evidence supports a bounded interpretation. Three seeds provide
descriptive replication but do not support a Holm-corrected seed-level
superiority statement. ACDC and Synapse are not consistently favorable, and
the matched GSL comparison is mixed in both directions, so GSL remains a
strong comparator that the study does not claim to beat. The matched family
shares the locked seeds, splits, schedule, and checkpoint rule, but the GSL
runs depend on an exact CPU distance transform substituted for the
quadratic-memory pairwise fallback.

For ACDC and Synapse, slices from the same patient or volume are dependent;
the cluster-level sample sizes are 20 and 12. The subgroup and failure-case
analyses are therefore exploratory and illustrative. They help locate metric
conflicts but do not establish prevalence or causality.

The resource study profiles inference and a controlled no-update training step
in one environment; it does not establish time-to-convergence, total
operational cost, or cross-hardware portability. The experiments use public 2D
benchmarks and do not establish conclusions for prospective use or 3D
architectures. These limits define the scope of the paper rather than
invalidate its central finding: boundary-focused supervision behaves
differently across tasks and metric families.

## 8. Conclusion

Boundary distillation can provide conditional benefits in lightweight 2D
medical image segmentation. In this study, the clearest favorable changes
occur on ISIC2018 boundary and distance measures while DSC remains nearly
unchanged. ACDC does not show a stable advantage, and Synapse demonstrates
that overlap/boundary means can move favorably while distance measures worsen.
A matched three-seed GSL comparison is mixed in both directions, so GSL
remains a strong geometry-aware comparator. The auxiliary boundary structure
can be decoupled from inference in the studied implementation with a deploy
graph close to the U-Net profile. The defensible contribution is therefore a
controlled empirical account of when boundary distillation helps, when metric
families disagree, and where the mechanism fails, rather than a universal
method-ranking claim.

## Data and Code Availability

The project repository retains locked split manifests, configurations,
evaluation artifacts, prediction-derived analyses, table sources, and audit
reports. Public release location and anonymized archive identifier are
`to be supplemented` after author approval.

## Ethics Statement

This study uses public benchmark datasets and does not report prospective
human-subject intervention. Dataset-specific licenses, original approvals,
and required acknowledgments must be checked against the source datasets
before submission.

## Conflict of Interest

To be completed by all authors before submission.

## Funding

X-Talent Program funding details and grant identifier: `to be supplemented`.

## Author Contributions

CRediT roles and final author order: `to be supplemented`.

## References

1. Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for
   Biomedical Image Segmentation. MICCAI. 2015.
2. Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net: a
   self-configuring method for deep learning-based biomedical image
   segmentation. Nature Methods. 2021.
3. Oktay O, et al. Attention U-Net: Learning Where to Look for the Pancreas.
   arXiv:1804.03999. 2018.
4. Zhou Z, Siddiquee MMR, Tajbakhsh N, Liang J. UNet++: Redesigning Skip
   Connections to Exploit Multiscale Features in Image Segmentation. IEEE
   TMI. 2020.
5. Chen J, et al. TransUNet: Transformers Make Strong Encoders for Medical
   Image Segmentation. arXiv:2102.04306. 2021.
6. Cao H, et al. Swin-Unet: Unet-like Pure Transformer for Medical Image
   Segmentation. ECCV Workshops. 2022.
7. Xie E, et al. SegFormer: Simple and Efficient Design for Semantic
   Segmentation with Transformers. NeurIPS. 2021.
8. Taha AA, Hanbury A. Metrics for evaluating 3D medical image segmentation:
   analysis, selection, and tool. BMC Medical Imaging. 2015.
9. Maier-Hein L, et al. Metrics Reloaded: recommendations for image analysis
   validation. Nature Methods. 2024.
10. Kervadec H, et al. Boundary loss for highly unbalanced segmentation.
    MIDL. 2019.
11. Karimi D, Salcudean SE. Reducing the Hausdorff Distance in Medical Image
    Segmentation with Convolutional Neural Networks. IEEE TMI. 2020.
12. Wang C, et al. Active Boundary Loss for Semantic Segmentation. AAAI. 2022.
13. Celaya A, Riviere B, Fuentes D. A Generalized Surface Loss for Reducing
    the Hausdorff Distance in Medical Imaging Segmentation. arXiv:2302.03868.
    2024.
14. Murugesan B, et al. Psi-Net: Shape and boundary aware joint multi-task
    deep network for medical image segmentation. EMBC. 2019.
15. Sun F, Luo Z, Li S. Boundary Difference Over Union Loss for Medical Image
    Segmentation. MICCAI. 2023. doi:10.1007/978-3-031-43901-8_28.
16. Hinton G, Vinyals O, Dean J. Distilling the Knowledge in a Neural Network.
    arXiv:1503.02531. 2015.
17. Romero A, et al. FitNets: Hints for Thin Deep Nets. ICLR. 2015.
18. Liu Y, et al. Structured Knowledge Distillation for Semantic
    Segmentation. CVPR. 2019.
19. Takikawa T, et al. Gated-SCNN: Gated Shape CNNs for Semantic
    Segmentation. ICCV. 2019.
20. Huang X, et al. MISSFormer: An Effective Transformer for 2D Medical Image
    Segmentation. IEEE TMI. 2023.
21. Ruan J, et al. EGE-UNet: an Efficient Group Enhanced UNet for skin lesion
    segmentation. MICCAI Workshops. 2023.
22. Huttenlocher DP, Klanderman GA, Rucklidge WJ. Comparing Images Using the
    Hausdorff Distance. IEEE TPAMI. 1993.
23. Holm S. A Simple Sequentially Rejective Multiple Test Procedure.
    Scandinavian Journal of Statistics. 1979.
24. Codella NCF, et al. Skin Lesion Analysis Toward Melanoma Detection: A
    Challenge at ISBI 2018. arXiv:1902.03368. 2019.
25. Bernard O, et al. Deep Learning Techniques for Automatic MRI Cardiac
    Multi-structures Segmentation and Diagnosis: Is the Problem Solved? IEEE
    TMI. 2018.
26. Landman BA, et al. Multi-Atlas Labeling Beyond the Cranial Vault.
    MICCAI Workshops. 2015.

**Reference note:** Details above were copied from the repository's existing
bibliography. Every entry, including the year/status of the GSL preprint, must
still be checked by a human against the final journal style and authoritative
publisher or index record before submission.
