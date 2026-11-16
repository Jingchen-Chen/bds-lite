# Volume metric provenance

Outputs are generated only from completed prediction-recovery cohorts under the
committed volume_evaluation_protocol.md. Source and output hashes are in each
volume_complete.json. Per-volume/per-class values remain distinct by family,
dataset and trained seed. ACDC frames must be averaged within patient before
patient inference; Synapse cases are the independent case unit.

TransUNet columns preserve the pinned evaluator's empty-mask conventions, which
can assign a perfect Dice to false foreground on an empty GT. Robust columns
and explicit empty states are supplied separately. Do not quote reference-policy
empty distances as measured errors. ASSD here pools surface-point distances.

Synapse native index grid uses original H5 GT and scipy order0 upsampling of
existing 224 maps. Its bilinear input preprocessing differs from the official
reference's cubic input zoom. ACDC uses the observed processed slice grid only.
No output is verified in millimetres. Native geometry, clinical volumes/EF/mass,
and true organ-name provenance remain separate acceptance gates.
