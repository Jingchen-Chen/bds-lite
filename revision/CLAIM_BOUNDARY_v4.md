# Final governing claim boundary v4 — statistical evidence v1.0.0

**Author-authorized final boundary. Raw-evidence cutoff: 2026-11-10 20:31:26 UTC.**
The author explicitly requested this update and the AGENTS.md pointer change.
This replaces v2/v3 for subsequent manuscript work; those versions and the
provisional reference remain historical, unchanged. The authoritative numerical
reference is `revision/BDS_LITE_EVIDENCE_FREEZE.md`. Registered protocols,
estimators, endpoints and original decisions are not retrospectively amended.

## Supported claims and limits

| ID | Supported conclusion and evidence | Statistical/practical reading and scope |
|---|---|---|
| C01 | In-network feature transfer plus gate did not establish a Boundary F1 gain over D under the registered estimator. Pooled without S: +0.0092 [-0.0010,+0.0194]; with S: +0.0011 [-0.0104,+0.0125]. `expansion_interaction.csv` | Three runs per arm/dataset; conditional fixed-effect summaries across three datasets. Not equivalence or proof of redundancy; not evidence about new datasets |
| C02 | Retrained ISIC full recipe minus D: -0.0321 [-0.0518,-0.0124], fresh-seed Holm p=0.0314. GSL minus D: -0.0308 [-0.0522,-0.0093], fresh Holm p=0.0137. `expansion_contrasts.csv` | Ten-run estimates, seven fresh seeds for the registered confirmation rule. Full recipe contains S/F/G; deficit cannot be assigned to distillation alone. Earlier-pipeline result differs and is uncertain |
| C03 | Original exploratory band KD contrasts were positive on ISIC/ACDC/Synapse; none passed Holm (0.496/0.088/0.500). `kd_unified/contrasts.csv` | Original ISIC3-vs-10 retained; later3-vs-3 display/family is post hoc. No dataset-transfer or confirmed cross-method ranking |
| C04 | Pre-registered ACDC unaugmented band KD minus D: +0.0178204 [+0.0098286,+0.0258122], p=0.00007842, 17 fresh runs/arm on50 official patients. `acdc_confirm/results/results.json` | Registered success **of unclear size** relative to0.01. ACDC selected post hoc; same center/protocol; reconstructed preprocessing. Post hoc lower bounds above0.01 cannot replace the registered reading |
| C05 | Same-patient exploratory full KD minus D: +0.0105641 [+0.0034447,+0.0176834]; compact-band KD minus D: +0.0118292 [+0.0037062,+0.0199522]; both Holm p=0.0196247. `followup/results/contrasts.csv` | Seventeen controls/arm, references reused after test results were examined. Positive specified-recipe contrasts, not independent confirmation or a general KD mechanism |
| C06 | Additional band and teacher-configuration contrasts remain unresolved: band-full +0.0072563 [-0.0005185,+0.0150311]; external-band minus compact-band +0.0059912 [+0.0000766,+0.0119057]; Holm p=0.0945487 for both | Nominal C4 exclusion of zero is not its adjusted decision. No equivalence, absence or necessity conclusion. Loss scale changes with region; teacher capacity/pretraining, realization, calibration, probabilities and band geometry are entangled |
| C07 | Secondary/per-structure scores describe the studied predictions | No secondary endpoint replaces a primary decision. Distances use resized-grid pixels and omit undefined one-sided-empty distances in the primary analysis; penalties are sensitivities, not measured distances |
| C08 | The registered SE duplicates a run-by-unit residual under the additive crossed model; post hoc alternatives can change interval exclusions. `restructure_posthoc/`, `posthoc_reference_estimators.csv` | Preserve primary estimator and original labels. Expected variance inflation is model-dependent; it does not guarantee interval coverage under misspecification |
| C09 | Representation, gate, gradient and error-location probes are diagnostic observations | Shared inputs do not prove redundant learned information. A band-limited loss can affect outputs elsewhere through shared parameters. No causal anatomy/error-location/teacher explanation is established |
| C10 | Historical contaminated ISIC519-image efficacy results are withdrawn; Experiment C is incomplete/blinded and excluded | Original checkpoints re-evaluated on clean official images are different evidence from withdrawn contaminated scores. Do not inspect/use C outcomes without its required amended analysis |
| C11 | External KD uses the same model architecture as D and adds a training loss only; auxiliary boundary decoder is skipped at ordinary inference | This is relative to D. Raw eager D/KD still execute projection, scale-zero gate and another output-head call. Plain-U-Net execution requires explicit pruning; algebraic identity is not measured zero overhead |
| C12 | Matched comparators and published references provide scoped context | Adapted LB-UNet and common-protocol losses are not exact published reproductions. Published splits, pretraining, augmentation, aggregation and distance policies differ. No cross-paper statistical ranking |
| C13 | Registered exploratory E3 under D4 augmentation: band_aug minus D_aug BF1 -0.0037083 [-0.0117766,+0.0043600], p=0.346144, 10 runs/arm,50 already examined patients. `evidence_freeze/final_e3_analysis.csv` | Positive benefit, harm and equivalence are all unestablished. Secondary IoU's nominal negative interval cannot replace BF1. No registered augmentation-by-KD interaction supports claiming augmentation eliminated the earlier effect |
| C14 | Saved profile: gate-retaining ISIC graph has1,072 extra parameters, approximately8.33%/14.99% batch1/16 FP32 GPU latency and104-to115 kernels relative to U-Net. `tables/resource_profile.csv` | Specific measurement hardware/protocol; timings not remeasured here. Pruned gate-collapsed graph differs from unpruned eager scale-zero execution. Original cache duration absent;27.2775s is a later partial-step benchmark only |

## Mandatory disclosures and interpretations to avoid

- Restrict conclusions to the studied compact2D recipes, datasets, partitions,
  teacher realizations and augmentation regime. No unsupported novelty,
  cross-dataset,3D, hardware or use-setting generalization.
- Do not compare the external-teacher confirmation effect head-to-head with
  in-network transfer: patients, seeds, run counts and trainers differ.
- Distinguish statistical significance, nominal intervals, multiplicity-adjusted
  decisions, engineering SESOI, and practical importance. Nonsignificance is
  not equality. A significant result beside a nonsignificant one is not a
  tested difference-of-effects.
- Checkpoint selection includes both-empty classes as1 and slice/image means;
  reported analyses exclude both-empty pairs and recipe contrasts use
  patient/case means. Other selection rankings are unknown.
- The roughly-0.02 official/internal ACDC score shift combines cohort and
  preprocessing differences; it is not a measured preprocessing effect.
- The registration's Bitcoin timestamp postdates training launch but predates
  first test access. Follow-up ordering rests on local records without verified
  external timestamp. Do not strengthen either provenance statement.
- The237 core design slots and540 historical completed-record claims are not
  interchangeable counts of independent models; aliases/reproductions remain.
- Statistical evidence preservation is finalized only within the documented
  scope. Local-only datasets, checkpoints, caches, predictions, external software
  overlays and missing historical artifacts prevent claiming complete portable
  training/inference reproduction. Zenodo deposition is not established.

## Changes from v3 and future revision discipline

V4 adds completed E3 and its unestablished augmented contrast, corrects the raw
inference-graph scope, replaces information/teacher-property causal shortcuts,
distinguishes original from post hoc cache timing, and reconciles experiment
counts and preservation limits. It preserves C01/C04 and every registered decision.

Fable/Opus may revise prose, layout, tables and narrative from the frozen inputs.
They may not silently alter source artifacts, methods, cohorts, seeds, estimators,
exclusions or conclusions. Any new analysis, recovered missing artifact or
scientific correction requires a versioned evidence addendum and appropriate
authorization, retaining this boundary and prior bytes as history. Reported-only
historical/supplementary values cannot become new inferential headlines without
the verification missing from their stated scope.
