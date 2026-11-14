# Results log

All entries below describe actual artifacts. Official U/S/F three-seed validation/test evaluations are complete; three WP5 diagnostic runs are registered, with ISIC actually started; no diagnostic run is yet claimed complete.

- 2026-11-02: 12 recovered ISIC U/S/B/F development runs, 519 images each, all 6,228 DSC/IoU pairs independently recomputed with maximum absolute discrepancy 0; validation/followup_overlap_recomputed.csv and analysis/revision/validate_recovered.py.
- 2026-11-02: Exact split membership checks pass ACDC train/val/test, Synapse train/val/test and ISIC val; primary ISIC train includes extra ISIC_0015312(1) outside locked 2,075 IDs; validation/split_compatibility.csv. No file deleted.
- 2026-11-02: 275 best.pt paths CPU-loadable; path and protocol metadata cataloged in audit/checkpoints.csv; duplicates do not constitute independent experiments.
- 2026-11-02: ISIC p1b seed-1 U/F model tensors exactly match recovered October U/F tensors; no independent replication credit assigned to duplicate paths.
- 2026-11-02: Manifest-controlled training masks yielded 20,029 image-class records, with 12 class summaries for processed-grid area, connectivity, inner-perimeter/area and empty classes; structural_descriptors/*.csv. No outcome effect or clinical millimetre interpretation inferred.

Historical preserved followup findings: final_analysis/REPORT.md reports mixed ISIC component results; all saved BF1/HD95/ASSD values and missingness have now been independently recomputed exactly. The ACDC phase-reduction report retains scientific STOP 0/3, and the Oct21 patient analysis explicitly does not reopen it. These are separate development studies, not successes of the PeerJ revision protocol.

- 2026-11-02: All 12 historical U/S/B/F runs passed complete boundary/distance recomputation (18,684 additional metric cells); validation/followup_boundary_distance_recomputed.csv.
- 2026-11-02: Existing U seed1 150-epoch rerun meets WP0 tolerances; selected model tensors equal, maximum trajectory DSC difference 1.5543e-15, DSC difference 1.9243e-7, HD95 difference 0.000337 px; validation/reused_reproduction.json. No new November training claimed.
- 2026-11-02: Fixed U/S/F × seeds1/2/3 evaluated on official val100/test1000; 9,900 saved maps pass independent DSC/ID/shape/dtype/checkpoint catalog validation. Official primary F−U and sensitivity results in tables/official_isic_primary_statistics.csv; generated findings in OFFICIAL_ISIC_FINDINGS.md.
- 2026-11-02: All nine legacy GSL/Phase16 validation trajectories differ; legacy weights cannot regenerate Phase16 outputs. Attention U-Net checkpoints use seed2026, not required seeds1/2/3; validation/gsl_lineage.csv and GSL_LINEAGE_AUDIT.md.
- 2026-11-02:52 recovery cohorts completed,32406 prediction maps and186316 image-class rows independently validated; all12 main ACDC/Synapse selected epochs match150-epoch logs, maximum replay DSC delta7.8e-16; validation/prediction_recovery.csv and main_selection_replay.csv.
- 2026-11-02:20 volume cohorts completed,520 volumes/2160 class rows; original/native-index and processed-index policies remain separate; all output hashes and four fixed real-volume EDT comparisons pass; validation/volume_outputs.json.
- 2026-11-02: FamilyV patient/case bootstrap/Holm analysis generated from existing main models. Both datasets show positive mean volume-Dice differences, with differing surface behavior; see VOLUME_FINDINGS.md and tables/volume_primary_statistics.csv. No mm/EF/mass result is inferred.
- 2026-11-02:202444 image-class error/area rows and995 area groups completed; distance rows joined without inference replay; scope and denominators in error_analysis/README.md and scope.json.
- 2026-11-02: Standard Synapse12 test cases/full z range verified; all1568 processed GT masks exactly match native H5 nearest conversion. Dataset/file attributes contain no physical geometry. Anatomical class-name provenance remains unresolved; no mask labels changed.

- 2026-11-02: 21 matched gate-identity cohorts completed; all73 recovery cohorts now independently validate45,042 maps and245,125 image-class rows. Gate state/config/operator lineage passes for21/21. Primary intervals fall inside engineering SESOI; masks are not identical and rank differences can be measurable. See GATE_IDENTITY_FINDINGS.md. This is not training M5.

-2026-11-02: CDF33 cohorts/149420 image-class rows/52981 valid surfaces complete. All saved HD95/missingness joins and10 independent KDTree checks pass; corrected right-continuous figure inspected. Near-contour and distant-tail changes coexist; spurious-component association is descriptive, not causal. SURFACE_ERROR_FINDINGS.md.
-2026-11-02: Training adapter accepted on exact CPU resume,5163 stored targets, three full-batch GPU probe-neutrality checks and real ISIC epoch2→3 continuation. First10 saved diagnostic validation DSC values exactly equal historical full-model trajectory; no full-run or new-seed credit. Initial probe scalar-identity check failed with residual<=9.62e-6; independent numerical validation remains pending.
-2026-11-02: Temperature85C triggered a safe checkpoint at ISIC epoch24. The registered bounded cooldown supervisor resumed the same run after cooling; all optimization/data settings retained. ACDC/Synapse remain queued.

- 2026-11-05: Run-level analysis from existing per-case CSVs: same-recipe U-Net gap on ACDC DSC +0.022 exceeds margin 0.018; ISIC BF1 margin 0.038 exceeds every boundary-arm effect; all-case scoring reverses retrained ACDC HD95 (-1.45 to +1.99 px). revision/tables/families/run_level_noise.csv, all_case_sensitivity.csv. No new training.
