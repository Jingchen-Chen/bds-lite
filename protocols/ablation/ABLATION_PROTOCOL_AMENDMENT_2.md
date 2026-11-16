# Amendment 2 to ABLATION_PROTOCOL.md (2026-11-02, before any ablation result was inspected)

## Pipeline-matched ACDC reference

The new ablations are trained by the revision adapter (which reproduces the p1b family exactly).
p1b ACDC BDS-Lite used class weights, so it cannot serve as the matched full-recipe reference for
the unweighted ACDC ablations. Three adapter-trained ACDC BDS-Lite runs without class weights
(`acdc_m6_full_unweighted_seed{1,2,3}`, otherwise identical to the WP5 full recipe, no probes) are
registered as the pipeline-matched M6 reference for ACDC. Family O (submitted) remains the
authoritative main comparison (revision/GITHUB_AUDIT.md).

Tier-B trimming (SENSITIVITY_PROTOCOL.md runs not yet started): alpha sensitivity keeps ISIC2018
seeds 1-3 and Synapse seed 1 only; lambda sensitivity keeps ISIC2018 seed 1 only, explicitly
labelled single-seed (plan section 3.2 allows seed-1-only sensitivity). Withdrawn entries are kept
in revision/queue_withdrawn.yaml.
