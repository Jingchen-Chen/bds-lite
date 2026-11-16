# Official ISIC evaluation — verified outputs, incomplete revision coverage

Fixed historic U/S/F checkpoints, seeds 1/2/3. Official validation n=100 and test n=1,000 images. No model/checkpoint/threshold selected on these labels. Original-resolution Jaccard uses nearest-resized argmax maps; thresholded Jaccard zeroes scores below 0.65.

## Test descriptive means

| Arm | DSC | Boundary F1 | HD95 (valid pairs) | Native thresholded Jaccard |
|---|---:|---:|---:|---:|
| U | 0.841038 | 0.363353 | 28.265873 | 0.649390 |
| S | 0.834963 | 0.332017 | 28.853812 | 0.630579 |
| F | 0.837359 | 0.366961 | 28.547504 | 0.641006 |

## Registered primary F−U comparison

Positive overlap/F1 differences favor F; positive raw distance differences favor U. Bootstrap resamples images after seed averaging; intervals are conditional on these trained seeds. Paired surface analyses exclude any unit missing any of the six required surface measurements, with an all-case diagonal-penalty sensitivity supplied separately.

| Metric | Paired n | Mean difference | 95% bootstrap interval | Holm p | Revision-stage SESOI classification |
|---|---:|---:|---|---:|---|
| dsc | 1000 | -0.003679 | [-0.005638, -0.001681] | 0.163040 | practically equivalent |
| iou | 1000 | -0.003529 | [-0.005785, -0.001195] | 0.221030 | practically equivalent |
| boundary_f1 | 1000 | 0.003608 | [0.000068, 0.007207] | 0.308742 | practically equivalent |
| hd95 | 997 | 0.352159 | [-0.215430, 0.918673] | 0.825765 | practically equivalent |
| assd | 997 | 0.245672 | [0.103073, 0.385685] | 0.006338 | practically equivalent |
| hd95_penalized | 1000 | 0.282947 | [-0.350238, 0.904949] | not tested (sensitivity) | practically equivalent |
| assd_penalized | 1000 | 0.207486 | [-0.200301, 0.640963] | not tested (sensitivity) | inconclusive |

The previously proposed independent-test HD95 benefit is not established. Full-recipe overlap means are lower, Boundary F1 is numerically higher, and ASSD is worse with a Holm-adjusted significant difference in the primary valid-pair family. All primary intervals remain within the chosen engineering SESOI bands. This does not establish clinical equivalence, patient independence, general superiority or a successful new method.

## Scope and outstanding checks

No exact sample IDs or JPEG byte hashes overlapped the legacy training/development images. Patient/lesion identity independence is not established. All 9,900 official prediction maps passed independent DSC, ID, dtype/shape and preregistered checkpoint-lineage checks (validation/official_outputs.csv). All 12 historical development runs also passed complete BF1/HD95/ASSD recomputation, including missingness.

GSL, other required matched comparators and remaining ablations are not yet in this official-test analysis. WP1a is therefore partial. ACDC/Synapse evaluation, physical-unit provenance, expanded families and full manuscript integration remain outstanding. Figures are deterministic drafts; GSL/comparator panels and human approval remain pending.

Sources: [official ISIC data and Task1 license](https://challenge.isic-archive.com/data/), [official 2018 Task1 evaluation](https://challenge.isic-archive.com/landing/2018/45/). Governing numeric artifacts: tables/official_isic_summary.csv, tables/official_isic_primary_statistics.csv; scripts in analysis/revision/. Prospective commit: a6dc26c.
