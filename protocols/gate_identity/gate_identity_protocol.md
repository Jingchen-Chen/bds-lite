# WP1i matched-checkpoint gate identity intervention

Frozen before generating/inspecting these counterfactual predictions. Historical
gate-removal claims are known, but their family/pipeline ambiguity is the reason
for this matched intervention. No model, seed or threshold is selected on test.

Use all fixed p1b full checkpoints, seeds1/2/3, on the same available val/test
cohorts as their recorded unmodified counterparts. ISIC aliases are proven tensor
identical to the completed October F checkpoints for all three seeds. Other
main-family validation DSC/selected epochs exactly replay the original logs.
Identities, hashes and cohorts are fixed in gate_identity_queue.json.

After strict loading of the original full model, replace only the inference gate
operator with the identity on the segmentation feature. This is exactly the
multiplicative factor obtained by setting g=0.5; encoder, segmentation decoder,
projection and saved checkpoint weights are not trained or modified on disk.
The auxiliary branch remains omitted as in standard inference. Record this
runtime intervention explicitly, not as an unchanged original resolved config.

Use identical FP32 NCHW preprocessing, batch sizes, seeds, argmax and unified
metrics. Save every prediction/per-case metric, including adverse cases and
missing distances. Compare within the same image/patient/case and trained seed.
Report per-seed metric changes and changed-pixel counts descriptively, with
seed-averaged unit effects. Any new corrected significance family requires its
own analysis specification; do not label numerical near-equality as proof of
population equivalence or zero deployment overhead.

This is inference gate collapse (WP1i/F10), **not** the training-time M5 ablation.
It cannot determine whether training with an absent gate would learn the same
weights, and cannot replace a missing M5 experiment. No new training occurs.
