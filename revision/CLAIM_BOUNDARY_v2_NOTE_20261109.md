# Claim boundary v2: disclosure note, 2026-11-09 (author-approved)

This note adds disclosure only. It does not change any allowed claim, forbidden claim, required
wording or decision in `revision/CLAIM_BOUNDARY_v2.md`, and it introduces no new conclusion.

**Approval.** The author approved this note on 2026-11-09 (restructure revision, Option B for
the shared variance component). It is kept as a separate file, so the approved claim boundary
text stays unchanged.

## What is disclosed

The recipe-level SE (`analysis/revision/expansion_statistics.py::contrast` and
`linear_contrast`) counts the run-by-test-unit residual term in both of its components. Its
intervals are therefore conservative. The effect is largest where test units are few, on ACDC
(20 patients) and Synapse (12 cases).

A post hoc sensitivity analysis gives every comparison with that term removed:
- Supplementary S20;
- `analysis/revision/posthoc_shared_variance.py`;
- outputs in `revision/restructure_posthoc/`.

Removing the term moves two headline intervals:
- **Pooled D+F+G − D Boundary F1:** [+0.0009, +0.0182] at k = 1, against the pre-specified
  [−0.0010, +0.0194].
- **Stage-3 primary:** a lower bound of +0.0104, against the pre-specified +0.0098.

## Rules that continue to hold

1. **All readings, labels and decisions use the pre-specified estimator.** This covers SESOI
   readings, confirmation statuses, the stage-3 pre-registered decision and its qualifier.
2. **The corrected intervals are a post hoc sensitivity analysis.**
   - They must not change any reading.
   - They must not be cited as evidence for any claim.
   - They must not appear in the Abstract.
3. **A1 is unaffected.** The allowed statement remains that in-network distillation did not
   establish a gain over D. The manuscript adds "under the pre-specified estimator".
4. **F11 is unaffected.** No wording may state or imply a gain "above the smallest effect of
   interest". The corrected stage-3 lower bound is reported as a number only.
5. **F16 still applies to secondary metrics.** This includes the stage-2 secondary metrics whose
   intervals exclude zero only after the adjustment.
6. **The response letter must disclose the issue and the S20 analysis proactively** (see
   `revision/CHANGELOG_restructure.md`, entry R2-S20).
