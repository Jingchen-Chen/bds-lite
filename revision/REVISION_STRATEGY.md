# Revision strategy after the GitHub audit (2026-11-02, supersedes earlier choices where they conflict)

Evidence: revision/GITHUB_AUDIT.md. Governing plan: revision/plan.md (= ~/Downloads/plan.md).
Items marked [APPROVAL] need the authors' decision before submission.

## 1. Experimental foundation

| Component | Models | Status |
|---|---|---|
| Primary comparison (Tables 2-3), ACDC and Synapse | submitted checkpoints (`outputs/runs`, "S0/S6") | results reproduce the original Table 3 with the revision evaluator; kept |
| Primary comparison, ISIC2018 | same submitted checkpoints, **official test set (1,000)** | [APPROVAL] original 519-image ISIC results withdrawn: 358/519 were training images of these models (audit C1) |
| Ablations M1-M5, comparators (GSL, Boundary Loss, Attention U-Net, LB-UNet) | trained in the revision with the released code (adapter) on documented partitions | compared with pipeline-matched U-Net/BDS-Lite ("M0/M6": p1b for ISIC/Synapse; adapter-trained unweighted BDS-Lite for ACDC) |
| Replication | p1b follow-up family ("R0/R6") | supplementary; ACDC BDS-Lite class-weighted |

Why pipeline-matched references: two trainings of the same configuration by different trainer
versions diverge from epoch 1, and the submitted ISIC models used a smaller training partition;
comparing revision-trained ablations directly with the submitted models would confound component
effects with these differences.

## 2. Necessity of the registered GPU runs (queue state at writing: 4/96 done)

| Runs | n | Reviewer item | Decision |
|---|---:|---|---|
| ACDC M1/M3/M4/M5 unweighted, Synapse M1/M3, ISIC+Synapse M4/M5 | 30 | R1-4 (major weakness), R3-M9, E-1 | **keep (Tier A)** |
| ACDC BDS-Lite unweighted reference | 3 | needed to make ACDC ablations interpretable | **keep (added)** |
| ISIC M1, M3 | 0 | reused: followup arms S/B (same pipeline, documented partition) | reuse, no training |
| Attention U-Net c=32 | 9 | explicitly requested (R3-M9) | **keep** |
| U-Net + Boundary Loss (official) | 9 | R1-3/R3-R15 | **keep** |
| LB-UNet (official) | 9 | most related boundary-assisted lightweight model (R3-O) | **keep** |
| U-Net+GSL retrain | 9 | original GSL checkpoints unavailable; GSL predictions needed for official ISIC test, volume, error and qualitative analyses (plan WP1g) | **keep, after the three comparators** |
| M2 (aux. boundary only) | 9 | tests prediction P2; Tier B | keep, Tier B |
| alpha sensitivity ISIC seeds 1-3 + Synapse seed 1 | 12 | R2-5 | keep (trimmed from 18) |
| lambda sensitivity ISIC seed 1 | 2 | F5 justification | keep (trimmed from 6) |
| ACDC class-weighted ablations, M0w | 14 | belonged to the p1b reference that is no longer primary | **withdrawn** (queue_withdrawn.yaml) |
| alpha Synapse seeds 2-3, lambda seeds 2-3 | 10 | marginal | **withdrawn** |
| acdc_m1_sdf_only_seed1 (class-weighted) | 1 | already running when withdrawn | kept as p1b-matched sensitivity only |

Estimated remaining GPU time ≈ 92 runs × ~45 min incl. thermal pauses ≈ 3 days. If time is short,
cancel in reverse order: lambda → alpha → M2 → GSL (then report original GSL aggregates only).

## 3. Historical evidence reused without retraining

- Submitted checkpoints S0/S6: official ISIC test, ACDC/Synapse test, stacked-volume metrics,
  gate→identity at inference (F10, now on the matched submitted checkpoints), error
  decomposition, small-structure rates, qualitative figure.
- Followup ISIC arms S (M1) and B (M3); p1b family as replication; WP5 gradient diagnostics.
- Original Phase16 GSL aggregates remain cited as the original comparator values.
- The April family-O ablation grid (`outputs/runs/*_{unet_surface_loss, bds_lite_boundary_only,
  bds_lite_gate_surface,...}`) was trained with the unreleased trainer and, for ISIC, on the same
  stale partition; it is not used as primary evidence. It could be evaluated on the official ISIC
  test and ACDC/Synapse test as a historical cross-check (CPU only) if reviewers ask.

## 4. Remaining reviewer-critical work (priority order)

1. [APPROVAL] Authors approve the ISIC correction and the corrected narrative.
2. Fill ablation/comparator results text as runs complete (auto-refresh regenerates numbers).
3. WP1h deployment profile (needs a 10-min queue pause on an idle GPU; script ready).
4. Clinical relevance (E-4): obtain native ACDC NIfTI with spacing and ED/ES labels to compute
   EF / myocardial-mass errors for S0/S6 (CPU only), or keep as stated limitation.
5. Paste verbatim reviewer text into the response letter; fill final page/line locations.
6. Release fixes for the public repository (audit R1-R5) to be applied before tagging
   `v2-revision` (not on the running experiment code).
