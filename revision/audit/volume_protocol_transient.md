# Prospective stacked/native volume evaluation

Evaluation-only revision work, frozen before inspecting new stacked/native
results. Existing slice-level outcomes were known. Include all fixed main p1b
U-Net/full seed1/2/3 and separately labeled legacy GSL seed1/2/3 and Attention
seed2026 reference predictions as they become available. No selection based on
new volume outcomes, and no substitution for missing Phase16 checkpoints.

- Synapse test: exact standard TransUNet12 cases, full original z range verified.
  Stack predictions using numeric slice IDs. Upsample the 224 argmax map with
  scipy.ndimage.zoom(order=0) to original H5 in-plane dimensions, following the
  pinned official output-resizing rule. Compare with original H5 labels.
  Original processed input used PIL bilinear, unlike the official cubic input
  zoom; disclose this inference-preprocessing difference, not full pipeline
  identity. Geometry/voxel spacing absent: distances are native-grid index units,
  not verified millimetres. Training set14 versus literature18 cases stays visible.
- ACDC test:20 patients,40 annotated frames, contiguous observed z indices.
  Stack numeric slice IDs on the 224 processed grid. Original native shape,
  verified spacing and ED/ES mapping are absent; do not claim native-resolution
  or clinical EF/mass evaluation. Report each frame and average within patient
  before any patient-level comparison.

For each volume/class report Dice, HD95, pooled bidirectional ASSD, mask volumes,
empty states and 6-connected component counts. Implement exact Euclidean
nearest-surface distances by cKDTree and verify against explicit EDT distances
on synthetic3D masks before execution. Surface extraction is erosion XOR with
6-neighbour connectivity, matching the default reference definition.

Two clearly separated empty policies: (a) pinned TransUNet calculate_metric_percase
semantics, including its unusual (pred nonempty/GT empty)->(Dice1,HD950), and
(pred empty)->(Dice0,HD950); (b) explicit robust reporting with both-empty Dice1,
one-empty Dice0, valid-only HD95/ASSD plus one-empty volume-diagonal sensitivity.
Do not silently turn an empty segmentation into a measured zero distance.

No physical HD95/ASSD/NSD, ejection fraction or myocardial mass until original
physical geometry is trustworthy. A native-index result is not mm merely because
it is compared to a paper table. All cases, seeds and adverse outcomes remain.

Before new volume effects are inspected, define a separate exploratory volume
family V: full-versus-U-Net tests of robust 3D Dice, valid-only HD95 and pooled
ASSD, Holm across these three metrics within each dataset. Average classes/frames
within a patient/case and seeds within that unit first. Use the registered
10,000-draw paired image/patient/case bootstrap implementation with RNG20261102.
Report eligible unit counts. Do not infer patient independence for ISIC.
Dimensionless Dice retains the revision-stage0.01 SESOI; no physical/practical
threshold is asserted for mixed-axis native/index3D distances. Surface penalties
are sensitivity outputs, not measured distances or replacement primary evidence.

Surface eligibility clarification before inspecting the new volume effects:
a primary patient/case surface endpoint is eligible only when every required
class/frame measurement in both methods and all three seeds is finite. Report
the excluded-unit count and retain an all-unit diagonal-penalty sensitivity.
Do not improve an endpoint by silently dropping a method's missed class from
its within-case average. No surface SESOI judgement is assigned on uncalibrated
3D index grids. This clarification changes no stored volume computation.
