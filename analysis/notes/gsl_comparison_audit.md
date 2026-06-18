# GSL Comparison Audit

## Evidence available

Aggregate GSL tables and Holm-corrected comparisons are available:

- `gsl_validation_summary.csv`
- `gsl_test_summary.csv`
- `gsl_holm_comparisons.csv`

Per-image GSL prediction arrays were not found, so case-level GSL deltas,
cluster bootstrap intervals, and failure panels cannot be produced from the
current artifacts.

## Validation comparison

| Dataset | Method | Dice | Boundary F1 | HD95 | ASSD |
|---|---|---:|---:|---:|---:|
| ISIC2018 | U-Net | 0.8647 | 0.5460 | 21.53 | 7.00 |
| ISIC2018 | BDS-Lite, gate removed | 0.8610 | 0.5477 | 20.22 | 6.89 |
| ISIC2018 | U-Net + GSL | 0.8667 | 0.5559 | 19.25 | 6.50 |
| ACDC | U-Net | 0.7867 | 0.8436 | 7.22 | 2.62 |
| ACDC | BDS-Lite, gate removed | 0.7905 | 0.8485 | 7.46 | 2.62 |
| ACDC | U-Net + GSL | 0.7938 | 0.8476 | 6.68 | 2.18 |
| Synapse | U-Net | 0.8409 | 0.8525 | 11.54 | 3.86 |
| Synapse | BDS-Lite, gate removed | 0.8423 | 0.8558 | 11.76 | 4.02 |
| Synapse | U-Net + GSL | 0.8578 | 0.8690 | 11.78 | 3.89 |

The available GSL rows are competitive and frequently stronger on overlap,
boundary, or distance metrics. The comparison does not support a claim that
BDS-Lite exceeds GSL.

## Combined BDS-Lite + GSL rows

Existing test summaries compare U-Net+GSL with BDS-Lite+GSL. The combined model
does not yield a uniform gain: ISIC2018 changes are negligible with lower
Boundary F1 and worse distance means; ACDC overlap/boundary means rise while
distance means worsen; Synapse is unfavorable across the listed means. These
rows are useful evidence that two boundary-oriented objectives do not simply
add.

## Statistical and protocol limits

The Holm table contains 90 adjusted comparisons, but the same n=3 seed
limitation applies. GSL and BDS-Lite schedules must be shown side by side before
any direct ranking is emphasized. Per-case GSL artifacts are missing.

## Resource and complexity comparison

GSL is loss-only and uses the U-Net inference graph. BDS-Lite removes its
auxiliary boundary branch at inference but retains a small gate-related
increment in the deployed graph. Both therefore have inference cost close to
U-Net in the available profiles. BDS-Lite adds an explicit auxiliary decoder
and gate during training; training-time FLOPs, memory, and wall time were not
profiled in a matched experiment, so no quantitative training-cost claim is
supported.

## Direct answers

**Can the paper claim BDS-Lite exceeds GSL?** No.

**How should the paper be rewritten?** Treat GSL as a strong main-table
reference. Position BDS-Lite as an explicit training-time
boundary-representation mechanism whose value is studied through metric
alignment, failure conditions, and removable inference components.

**Does BDS-Lite retain distinct value?** Potentially as a mechanism-analysis
vehicle and for explicit auxiliary-branch removal. The current artifacts do not
show a clear predictive advantage over the simpler GSL alternative.

**Main table or supplementary material?** Main table. Moving GSL out of the
main comparison would hide the strongest reviewer objection. Detailed Holm
rows and combined-objective experiments can be supplementary.
