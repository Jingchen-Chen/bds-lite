# When does boundary distillation help? (mechanism notes)

These notes summarize the **conditional, mixed** picture reported in the manuscript.
They are an interpretation of the measured results, not a claim of uniform
improvement. The numbers below are cluster-aware (see `docs/statistics.md`); the
artifacts of record are in `analysis/outputs/` and `results/`.

## Summary of the three datasets

| Dataset | Boundary / distance effect of BDS-Lite vs U-Net | Reading |
|---|---|---|
| **ISIC2018** | Cluster-level boundary/distance metrics shift favorably and reach Holm significance (HD95, Boundary F1, ASSD); DSC essentially unchanged. | The training-time boundary path helps where targets are **single, large, high-contrast** objects with boundary error dominated by a few hard cases. |
| **ACDC** | No stable advantage: Holm-adjusted p ≈ 1.0 for DSC/IoU/Boundary F1; some unfavorable mean distance directions. | For **compact, well-localized** cardiac structures the U-Net boundary is already accurate, leaving little for the auxiliary path to add. |
| **Synapse** | Overlap/boundary means tick up but **distance means get worse**. | For **many small/variable** abdominal organs, gains on average overlap can coincide with worse worst-case distance, i.e. a genuine trade-off, not a uniform win. |

## Why the effect is task-dependent

- The auxiliary boundary decoder `h_φ` and the bounded gate
  `q̃ = q · (1 + α(g − 0.5))`, α = 0.25, only **re-weight** the segmentation
  features near predicted boundaries. They help most when boundary error is the
  binding constraint and the gate has a coherent boundary signal to act on.
- When the backbone already resolves boundaries well (ACDC), the bounded gate's
  small perturbation neither helps nor hurts consistently.
- When there are many small, low-contrast structures (Synapse), improving average
  boundary agreement can still leave (or worsen) large-distance outliers, which HD95
  and ASSD are sensitive to.

## What is explicitly NOT claimed

- No uniform improvement across datasets or metrics.
- No state-of-the-art claim, no clinical-deployment-readiness claim, no claim of
  faster training.
- No Holm-corrected seed-level superiority at n = 3 (all seed-level adjusted
  p ≈ 1.0; the cluster-level analysis is the primary evidence).

See `manuscript/tables/table6_claim_evidence_final.csv` for the claim-by-claim
evidence map and `analysis/notes/` for the longer per-case and subgroup commentary.
