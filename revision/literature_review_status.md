# Literature verification status

## Reviewer suggestion R3-I7

DOI10.1002/ima.70424 resolves to Wiley's *ViT-Swin-UNet: A Transformer-Based Framework for Liver Tumor Analysis in CT Images*, first published July29,2026, International Journal of Imaging Systems and Technology36(5), e70424.

Primary publisher source: https://onlinelibrary.wiley.com/doi/10.1002/ima.70424 . Publisher abstract describes a slice-wise ViT classification/Swin-UNet segmentation pipeline on LiTS-derived CT data. This is potentially relevant as context for transformer and liver-tumor scope; the abstract does not establish a directly comparable training-only boundary-distillation study.

Direct DOI fetch initially failed; publisher search indexing exposes metadata/abstract, but a complete original-paper methodological assessment and original PDF table/page verification have not yet been completed. Citation decision remains pending, not falsely marked reviewed/irrelevant. No performance number from this article has been imported into manuscript tables.

## Comparator/reproducibility reference

Official TransUNet source pinned at commit02ef0010b36eb8328b5e689eadaf613602edf9b8 in sources/transunet/receipt.json. The standard Synapse12 test IDs match local data exactly. utils.py calculate_metric_percase calls HD95 without voxel spacing; z_spacing is used only when writing image files. Its unusual empty-class branches must be retained verbatim for reference-compatible reporting and accompanied by a robust explicit empty-policy sensitivity, not silently treated as clinically valid distances.

Its native inference uses cubic in-plane image zoom and order0 prediction zoom. Existing BDS processed images use PIL bilinear resizing. A volume report must disclose this preprocessing difference and not claim full evaluator equivalence merely from matched case IDs.
