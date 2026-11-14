# Historical ablation provenance warnings

Checkpoint tensor hashing identifies 229 distinct complete model states among 275
best.pt paths. Seventy-five paths belong to duplicate-state groups; none are
counted as extra independent experiments merely from duplicated path names.
See validation/checkpoint_states.csv and checkpoint_state_duplicates.csv.

For all three datasets/seeds, archived `bds_lite_boundary_only` and
`unet_boundary_loss` checkpoints have identical complete state dictionaries.
This does not establish a Kervadec boundary-loss comparator: the name may refer
to auxiliary boundary supervision or an alias. Do not count it as C1 without
recovering and validating the actual objective/source.

Other archived ablations contain explicit use_boundary_decoder/use_boundary_gate/
use_surface_loss/use_distillation flags and loss terms. Generic boundary_gate=true
can coexist with use_boundary_gate=false. In inspected surface-only/boundary-only
checkpoints, gate tensor keys are absent, consistent with the explicit use flag.
Legacy projection_head keys rename exactly to current seg_to_boundary_feature;
this is a key mapping, not proof that the original training objective matches.

The old adapt_legacy_* scripts only copy use_boundary_gate when boundary_gate is
absent; they therefore cannot be applied blindly to these contradictory configs.
Strict loading and graph semantics must be checked before any scientific reuse.
No current source or original checkpoint was modified.

Actual ablation logs/configs/environment records exist under outputs/runs,
including 150-epoch histories and historical git commit
5914d35ce63d20b5edd1b7115dc36aefb4606bca. This commit is not in the preserved
76-commit public/followup history, which begins with a public reproducibility
release. Do not claim that the complete pre-release Git history was recovered.
Recover or independently validate the historical objective before replacing
planned missing experiments with these artifacts. Do not unnecessarily retrain
until this investigation is exhausted.

The followup G/C/D controls are not the roadmap M2/M4/M5 arms: all retain the
fixed gate and omit SDF. G is gate/projection only; C adds boundary supervision
without distillation; D adds detached-target matching without boundary supervision.
They must not be relabeled as no-gate or full-minus-distillation ablations.
validation/ablation_candidates.csv maps component metadata without certifying
training-source equivalence or completion. Existing p1b gate-removal artifacts
also require distinguishing training-time removal from inference conversion.
