# Source-verified methods draft — not final manuscript text

These statements describe the active primary implementation. Checkpoint-family parity must be established before attributing them to every historical experiment. Sources: src/bds_lite/{models/bds_lite.py,training/losses.py,losses/gsl.py,data/boundary.py,evaluation/metrics.py}.

## Boundary pathway and stop-gradient (F3/F5/F6/F7)

At width c=32, segmentation features have N×32×H×W and the auxiliary boundary features N×16×H×W. The boundary decoder uses three UpBlocks (256,128→64; 64,64→32; 32,32→16), followed by a 1×1 output convolution. Output channel count is boundary_classes in the resolved config; the default is one union boundary, not automatically K−1 classes. Verify each actual config before describing multiclass targets.

The segmentation-side projection P is Conv2d(32,16,1), producing q_s. The retained gate W_g is Conv2d(16,32,1), applied per sample, per channel, per pixel. It uses q_s, not a boundary decoder output at inference. g=sigmoid(W_g q_s+b_g), F_out=F_s[1+alpha(g−1/2)]. Thus alpha=0.25 bounds modulation between 0.875 and 1.125. This bound does not prove accuracy or latency neutrality.

L_distill=mean SmoothL1(q_s,stopgrad(q_b)), with default beta=1. stopgrad is identity forward and zero derivative backward (detach()). This loss sends no gradient through its target q_b. **Do not claim the boundary decoder is trained only by L_boundary across all variants without checking its other graph paths and detach_boundary_features setting.** The gate uses segmentation proxy features; auxiliary decoder output is returned only when requested.

Boundary target for each nonbackground class is dilation(M) XOR erosion(M), unioned when requested. SciPy's default binary morphology uses the cross-shaped connectivity-1 structuring element in 2D. Calling it an all-ones 3×3 square would be inaccurate. Radius is one iteration by default. Boundary loss is mean BCEWithLogits plus channel-mean sigmoid Dice loss.

## SDF surface objective (F3)

For each sample n and foreground class k, d_nk=EDT(outside)−EDT(inside), negative inside. Empty masks produce all-zero SDF. Let s_nk=max_x|d_nk(x)| clamped at epsilon and v_nk=1 when max|d|>0. With no class weighting,

L_surface = sum_nkx v_nk p_nk(x) [d_nk(x)/s_nk] / [H W sum_nk v_nk].

Return zero if every SDF is invalid. The weighted-class branch uses a separate valid-class weighted mean. This term is signed and can be negative; it is not an absolute distance error. Its logit gradient also passes through softmax. Increased distance weighting alone does not establish a causal mechanism in trained models.

Region Dice aggregates over batch and spatial axes (0,2,3) per foreground class, then averages classes, with epsilon 1e−6. It is not computed independently on each slice then averaged. CE is standard multiclass cross entropy. Historical followup loss-reduction experiments used different reductions and augmentation and cannot be pooled into this protocol.

## GSL (F4)

The implemented region term is CE plus mean_nk [sum_x(y−p)^2 / max(sum_x y²+sum_x p²,epsilon)], including background. The default distance map is distance to pixels belonging to each class (not distance to that class boundary); absent-class distance is the image diagonal.

With z=(1−y)^2 and w_nk=1/[(sum_x y_nk)^2+1],

L_GSLboundary = 1 − mean_n {sum_k w_nk sum_x[d_nk(z_nk−p_nk)]² / max(sum_k w_nk sum_x[d_nk(z_nk−y_nk)]²,epsilon)}.

The GSL combined loss is a_e L_region+(1−a_e)L_GSLboundary. For step5, e0=max(e−1,0), J=max(1,floor((T−5)/5)); a_e=max(0,1−floor(e0/5)/J), set to zero at e0≥T−5. Distinguish this schedule coefficient from the BDS gate alpha.

**Integration caveat:** BDSLiteLoss can add its own CE/Dice on top of lambda_gsl×the combined GSL loss. Resolve ce_weight/dice_weight/use_gsl for each checkpoint before claiming the objective matches any original paper. Original-paper citation verification is outstanding. The default torch.cdist distance implementation can exceed laptop memory; preserved Phase16 SciPy wrapper and followup implementation require exact equivalence validation before reuse.

## Empty classes and distances (F9)

Hard DSC=(2 intersection+eps)/(pred_area+GT_area+eps); IoU=(intersection+eps)/(union+eps). Both empty yield one. One empty yields a small epsilon ratio, not exactly zero. HD95 and ASSD return NaN whenever either surface is empty, including both-empty. Metric means must report valid counts rather than silently interpreting omitted failures favorably. Boundary F1 is one for identical boundaries (including both-empty), zero if only one empty; otherwise morphology-matched precision/recall with tolerance two. Surface extraction uses inner erosion XOR, which differs from the two-sided boundary target band.

## First-order rationale (WP5b; draft)

For a small normal displacement u(s), the symmetric-difference area is approximately integral_boundary |u(s)| ds. If delta is the perimeter-weighted **mean absolute** displacement, this is L delta. A signed mean displacement is insufficient because inward/outward errors can cancel. Since Dice=1−|P△G|/(|P|+|G|), approximately Dice≈1−L delta/(2A) when |P|≈A. Assumptions exclude topology changes, distant components and complete misses. A one-pixel extrapolation can be poor for tiny structures.

Training-only processed-grid descriptors use the discrete inner-boundary-pixel count as a perimeter proxy. It is orientation-dependent, not a physical perimeter estimator. The summary's L/(2A) is a local sensitivity proxy, not an observed accuracy gain. Numeric class-level values must be generated from structural_descriptors/summary.csv rather than manually entered into manuscript prose. Gradient-conflict hypotheses remain unproven.

## Additional interpretation constraints

Boundary BCE is evaluated on both positive band pixels and negative pixels
throughout the image. A thin positive target does not mathematically imply zero
or localized gradients outside the band. Gradient-locality claims need measured
gradients; distance weighting alone does not prove a trained-model mechanism.

Training-mask ISIC connected components and discrete L/A must be inspected
before calling every lesion one closed contour. The one-pixel linearized Dice
proxy is an approximation and can be several percentage points; do not call it
negligible automatically. HD95 is not a mean normal contour displacement and
cannot be substituted for delta in the first-order derivation.
