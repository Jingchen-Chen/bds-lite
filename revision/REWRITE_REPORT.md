# Manuscript reconstruction report (2026-11-05)

Outputs: `revision/manuscript/{main,supplement,tracked_changes}.pdf`, `revision/response_letter_v2.pdf`.
Coverage: `revision/REVIEWER_COVERAGE.md`. Author items: `revision/HUMAN_DECISIONS.md` (top section).

## 1. What was wrong with the previous version
- **The headline was not reproducible.** The abstract led with Boundary F1 +0.031 from the models trained on the stale partition. The same recipe on the documented partition gives +0.004.
- **Intervals ignored retraining.** Bootstrap CIs over test images were read as evidence about the recipe. Three seeds per arm were averaged away before inference.
- **A low baseline drove the "gains".** ACDC retrained U-Net DSC is 0.742 against 0.764 in the original set, same recipe. It is the lowest retrained arm apart from LB-UNet, and no other retrained arm reaches the original U-Net's 0.765 (unit-weighted). "Practically superior to U-Net" on ACDC largely measured this.
- **A wrong statement about errors.** The Discussion said multi-structure errors are "not imprecise contours". By pixel share 70% of ACDC errors are within 2 px of the contour, while on ISIC only about 20% are.
- **Claims with nothing behind them.** Methods promised an all-case sensitivity analysis, minimum detectable effects, and official-validation results that were never reported. The first two now exist; the third claim is removed.
- **A distance conclusion that was not stable.** Primary HD95 skips missed structures. With them scored, retrained ACDC HD95 changes from -1.45 to +1.99 px.
- **Structure.** "Characterization of BDS-Lite" with P1-P4 presented post-hoc expectations as testable predictions, in labelled paragraphs that read like a checklist of reviewer requests. Three model families with overlapping identities were hard to follow.

## 2. Central contribution now
A method paper that tests its own design assumption. BDS-Lite is defined precisely, its one distinctive claim is stated (distillation adds something beyond an auxiliary decoder), and that claim is tested and not supported. Value lies in that component-level evidence, the transparent ISIC correction, and three evaluation lessons: independent test data, repeated training, and how undefined cases are scored. I rejected both alternatives: keeping "controlled characterization" undersells a real design, and selling a method overstates evidence that does not support it. Title kept (plan 5.1): the paper now answers its question.

## 3. Structural changes
| Before | After | Why |
|---|---|---|
| Labelled Introduction, 7 paragraph headings | 5 prose paragraphs, opens on a concrete number | Coherent; still covers R3-I4 order |
| Theoretical rationale + P1-P4 | Short Design rationale, labelled post hoc | Not preregistered, so not presented as if it were |
| "Primary / matched / replication" | Two sets: original, retrained | Replication was the same models as matched on ISIC/Synapse |
| 9 Results subsections | 5 | One claim each |
| 10 main tables | 5 (one table holds all arms) | Fewer, stronger |
| No view of run spread | Fig. 3 (new) shows every run against a noise band | Carries the central statistical point |

## 4. Claims weakened, corrected, or removed
- Removed: HD95 benefit on ISIC; "benefit on single-contour tasks"; Synapse "overlap-distance conflict" as central finding; "structural-complexity gradient"; "capacity competition observable"; "attributed to the auxiliary decoder" (now "suggestive"; inside run-level margin); gradient ordering as support for a prediction.
- Weakened: BDS-Lite worsening ACDC distances (scoring-dependent); M3 "practically superior" on ACDC (low baseline); GSL lead (consistent, 8 of 15 Holm-significant, 0 beyond margin).
- Added against interest: spurious components rise under BDS-Lite in 6 of 6 dataset-set pairs; the surface loss did not reduce HD95 anywhere.

## 5. Where analyses went
Main: two-set comparison, all-arm table, run figure, literature context (condensed, with our own 3D and challenge-metric rows, previously missing), error location, spurious components, missed structures, all-case result, qualitative ISIC, deployment.
Supplement: full statistics, per-arm means, omnibus, GSL reproduction, gradient detail and residual audit, sensitivity detail, volume table, related-methods table, algorithm, comparator equations, ISIC audit detail.
Dropped: official validation-set claim; design-rationale table (now prose).

## 6. Evidence map
| Claim | Evidence | Limitation |
|---|---|---|
| ISIC original results invalid | 226.9 steps/epoch in 6 checkpoints; 358/84/77 from manifests; 0 overlap with official test (all re-derived here) | Repeated lesions exist; not patient-independent |
| BF1 effect not reproduced | family_statistics P, P-matched; run_level_noise.csv | 3 runs per set; partitions differ between sets |
| HD95 benefit gone | same; both sets, CI includes 0 | none material |
| Distillation, gate, surface loss unsupported | family A: 0/45 worse, 6 better (ISIC); gate identity; alpha sweep | ACDC/Synapse intervals wide; absence is not equivalence |
| Decoder alone suggestive | A0 M2 +0.0160 | Inside margin 0.038; no interaction estimate |
| GSL at least as good | C-vs-M6: 15/15, 8 Holm | None beyond margin; GSL also gains from low U-Net on ACDC |
| Run-to-run noise rivals effects | run_level_noise.csv; selected epochs | Post hoc; pooled variance; cause of U-Net gap unknown (trainer lost) |
| Deployment properties hold | tab_resource; gate identity | One machine |

## 7. Statistical and gradient review
- Run-level margin is new and post hoc, labelled so in Methods and Limitations; `run_level_noise.csv` has per-run values.
- A pairing rule was undisclosed: differences use samples defined in both arms. It matters (ACDC HD95 -1.00 paired, -0.61 unpaired) and is now stated.
- Gradient residuals re-read from `gradient_residual_audit.json`: median relative at most 7.4e-5, maximum implied cosine error 6.1e-4, against a smallest dataset gap of 0.139. Immaterial, and exact equivalence is not claimed. The bigger limits are one seed per dataset and a class-weighted ACDC run, so gradients moved to the supplement and are called indicative.

## 8. Validation
| Check | Result |
|---|---|
| main / supplement / letter compile | OK: 16 / 29 / 12 pages (exit code and PDF timestamp checked) |
| Undefined keys, placeholders, unresolved refs, missing citations or figures | 0 |
| All tables and figures referenced | yes |
| Independent re-derivation from raw CSVs and manifests | 12 of 12 match (`revision/validation/rewrite_validation.json`) |
| Letter locations | 86, 0 missing |
| Reviewer quotes verbatim | 40 of 40 |
| ruff format, ruff check (repo) | pass |
| Unit tests | 59 passed |
| Tracked changes | rebuilt, 25 pages; most text is new so it is dense |
| Abstract | 279 words |

Unresolved warnings:
- TeX reports only box and font-shape warnings; none affect content.
- **Build bug found and fixed.** Under tectonic the template's Times fell back to Latin Modern regular, silently removing bold and italics, and crashed xdvipdfmx on the new text. `\usepackage[T1]{fontenc}` fixes both. Earlier PDFs built here had that wrong font. Use `revision/build_check.sh`, which fails on crashes and stale PDFs.
- PeerJ's author instructions could not be read (HTTP 403), so limits are unverified.
- `manuscript_v2/ANCHOR_ABSTRACT.md` and `manuscript_v2/claim_evidence_table.md`, named in AGENTS.md, do not exist on this branch. The anchor check could not be performed. Root `PLAN.md` is the historical plan and has no phase for this task, so it was not edited.
