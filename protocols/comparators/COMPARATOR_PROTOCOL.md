# WP4 queue-B protocol: GSL retrain, C1 Boundary Loss, C4 LB-UNet (frozen before training)

Written 2026-11-02 before any queue-B run started. Hashed into every queue-B config.
Locked protocol, splits, seeds 1/2/3, selection rule and evaluation are identical to
ABLATION_PROTOCOL.md / TRAINING_ADAPTER_PROTOCOL.md. Adapter:
scripts/comparators/train_comparator.py (train_run.py main with model/criterion
factory; epoch loop, AMP, clipping, resume, selection and evaluation imported
unchanged). Dispatcher: scripts/comparators/run_queue_b.py (run_queue.py copy;
same GPU lock, thermal guard and completion contract). No class weighting.

## GSL (conditional Tier-A retrain)

The Phase16 checkpoints behind the submitted manuscript's U-Net+GSL rows are missing
(GSL_LINEAGE_AUDIT.md); surviving legacy P2 weights have divergent trajectories and
are not substituted. Retrain with exactly the Phase16 configuration: U-Net c=32,
CE 1 + Dice 1 + lambda_gsl 1 * GSL (alpha step5 schedule), tracked
`bds_lite.losses.gsl` source, distance transform replaced in-process by the exact
SciPy EDT wrapper used by Phase16. Validation: compare each retrained run's 150
validation-DSC trajectory and selected epoch against the preserved Phase16
metrics.jsonl. Agreement is reported as a reproduction check; disagreement is reported
as such and the retrained family (not Phase16 aggregates) is used for every new
analysis that needs predictions. Old Phase16 numbers are never mixed with retrained
predictions.

## C1 U-Net + Boundary Loss (Kervadec et al.)

Official implementation LIVIAETS/boundary-loss @171c32d88a4c (MIT): SurfaceLoss =
mean(softmax[:, idc] * dist[:, idc]); dist = one_hot2dist (edt(neg)*neg -
(edt(pos)-1)*pos, unit pixel resolution, absent classes zero); "gdl_surface_steal"
schedule: weights [region, boundary] = [1, 0.01], after every epoch
[max(0.1, a-0.01), b+0.01]. Matched adaptations (declared): region term = CE + soft
Dice exactly as the U-Net baseline (upstream: generalised Dice); idc = all foreground
classes; dist maps computed on the fly from the stored 224x224 masks.

## C4 LB-UNet (Xu & Tong, MICCAI 2024)

Official lbunet.py @913ee687ecc4 fetched at runtime and SHA-256 verified (upstream has
no license, so it is not redistributed). Official widths c_list=[8,16,24,32,48,64]
(the method is ultra-lightweight by design; no width matching). Multi-class
adaptation: K-channel final conv; pre-sigmoid logits (forward hook + upstream x2
bilinear upsampling) trained with CE + soft Dice; five foreground deep-supervision maps
(weights 0.1-0.5) and three boundary maps (0.1/0.2/0.3) with upstream BCE+Dice
(0.5/1) against the foreground union and this study's union morphological boundary
target (upstream genetic-algorithm key-point maps are ISIC-specific; not reproduced).
Boundary components remain in the deployed graph (as in upstream). Upstream training
settings (256 px, lr 1e-3, 300 epochs, augmentation) are replaced by the locked
protocol; this is a matched-protocol comparator, not a reproduction of published
LB-UNet numbers.

## Analysis

Family C: each comparator vs M0 U-Net, Holm across five metrics within dataset,
plus descriptive comparison with M6; omnibus Friedman per dataset x metric across all
complete matched methods with Holm-adjusted Wilcoxon post-hoc tests, as registered in
analysis/revision/prereg_revision.md. All outcomes reported regardless of direction.
