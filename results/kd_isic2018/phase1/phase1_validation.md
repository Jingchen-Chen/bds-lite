# KD comparator Phase 1 (validation only, exploratory)

Teacher gate: PASSED (teacher val Boundary F1 0.5922 vs 0.5376; DSC 0.9015 vs 0.8632).
Feasibility: {'F1_teacher_gate': True, 'F2_students_stable': True, 'F3_resources': True}. Proceed to Phase 2 under Option A: True.

| Metric | D+KD-band (3) | Decoder-only (10) | Delta [95% CI] | label |
|---|---|---|---|---|
| boundary_f1 | 0.5449 | 0.5376 | +0.0073 [-0.0231, +0.0377] | undetermined |
| dsc | 0.8620 | 0.8632 | -0.0012 [-0.0078, +0.0055] | no relevant effect |
| hd95 | 21.1007 | 20.2952 | -0.8055 [-4.1705, +2.5595] | undetermined |
| iou | 0.7849 | 0.7866 | -0.0017 [-0.0113, +0.0079] | undetermined |
| assd | 6.8751 | 6.7804 | -0.0947 [-1.0741, +0.8846] | undetermined |

Positive favours D+KD-band; HD95/ASSD signs reversed. Three runs against ten: not a test.
