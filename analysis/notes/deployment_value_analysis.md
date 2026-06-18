# Gate Removal and Inference-Value Analysis

## Structural audit

BDS-Lite contains a compact U-Net segmentation path, a gate-modulated
representation path, and an auxiliary boundary decoder used for training
signals. The auxiliary boundary branch is not required for deployed
segmentation output. Existing checkpoints were evaluated with the gate active
and removed; resource profiles separately distinguish total training
parameters from the deployable segmentation graph.

## Gate-removal results

Three-seed validation means change only slightly when the gate is removed:

| Dataset | Dice full -> removed | Boundary F1 | HD95 | ASSD |
|---|---:|---:|---:|---:|
| ISIC2018 | 0.8607 -> 0.8610 | 0.5487 -> 0.5477 | 20.08 -> 20.22 | 6.87 -> 6.89 |
| ACDC | 0.7847 -> 0.7851 | 0.8330 -> 0.8335 | 7.76 -> 7.80 | 2.344 -> 2.343 |
| Synapse | 0.8466 -> 0.8449 | 0.8595 -> 0.8583 | 10.59 -> 10.49 | 4.028 -> 4.071 |

The gate is active during training (mean absolute gate parameter about 0.183;
mean activation deviation from 0.5 about 0.319), but these results show that it
can be omitted with small aggregate changes in the studied validation runs.
This is decoupling evidence, not proof that every trained model is insensitive
to gate removal.

## Resource profile

CUDA profile at the recorded 256x256 input:

| Model graph | Parameters | FLOPs | FPS | Peak memory |
|---|---:|---:|---:|---:|
| U-Net | 1,927,042 | 15.970 G | 560.9 | 65.45 MB |
| BDS-Lite deploy | 1,928,114 | 16.079 G | 570.0 | 66.50 MB |
| EGE-UNet | 55,253 | 0.193 G | 786.2 | 17.03 MB |

BDS-Lite deploy adds 1,072 parameters (0.056%), about 0.68% FLOPs, and about
1.05 MB measured peak memory relative to U-Net. Total training-time parameters
are 2,198,003; 269,889 auxiliary boundary-branch parameters are excluded from
the deploy graph. The small FPS difference between U-Net and BDS-Lite should be
treated as benchmark variation, not a speed claim.

## Comparison boundary

- GSL is loss-only and has the U-Net inference graph.
- BDS-Lite has a removable auxiliary branch and near-U-Net deploy cost.
- MobileViT-UNet and EGE-UNet resource rows are useful compact-model context,
  but their training schedules and predictive results must be protocol-matched
  before ranking methods.
- No matched training-time wall-clock, energy, or peak-memory profile is
  available for U-Net, GSL, and BDS-Lite.

## Defensible paper wording

Use: “In the studied 2D configurations, the auxiliary boundary branch is used
during training and omitted for segmentation inference. The recorded deploy
graph differs from U-Net by 0.056% parameters and 0.68% FLOPs, while gate
removal causes small aggregate metric changes.”

Do not write that the method is ready for hardware or clinical use, or that it
is faster than U-Net. The deployment contribution is architectural decoupling
and measured near-backbone cost.
