# Amendment 1 to ABLATION_PROTOCOL.md (2026-11-02, before any ACDC ablation result was inspected)

Verification against the public GitHub release (configs/run_resolved/acdc_bds_lite_full_seed*.yaml)
and the submitted-manuscript checkpoints (outputs/runs/acdc_bds_lite_full_seed*) shows that the
submitted ACDC BDS-Lite recipe uses **no** foreground class weights (`foreground_class_weights: null`).
The class weights [1.5, 1.0, 1.25] belong only to the later p1b follow-up family, which is not the
submitted family. ABLATION_PROTOCOL.md wrongly took p1b as the reference.

Therefore all ACDC ablation arms that had not started (M1 seeds 2-3, M3, M4, M5 seeds 1-3) are
re-registered **without class weights** (`*_unweighted_*` run IDs, otherwise identical), and the
class-weighted ACDC registrations are withdrawn from the queue. The already-running
`acdc_m1_sdf_only_seed1` (class-weighted) is allowed to finish and is retained only as a p1b-family
sensitivity arm; it is not part of the primary ablation family. The class-weighted U-Net control
M0w is withdrawn (its purpose was to deconfound p1b, which is no longer the primary family).
ACDC M1 seed 1 unweighted is added. No ACDC ablation metric had been inspected at this point.
