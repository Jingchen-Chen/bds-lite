# Synapse class-name provenance gate

All new per-class outputs deliberately retain numeric label IDs. Embedded legacy
configs list [background,aorta,gallbladder,spleen,left_kidney,right_kidney,liver,
stomach,pancreas]. The common literature table order is [aorta,gallbladder,
left_kidney,right_kidney,liver,pancreas,spleen,stomach]. A paper's display order
alone does not establish the actual pixel-label mapping of this local H5 family.

Training-mask area distributions make the embedded names questionable (for
example class5 has the largest foreground area), but that observation is not
sufficient to relabel anatomy. Original H5 files contain no label-name metadata.
Original scalar/figure organ names therefore need a provenance check before
pancreas/liver/spleen-specific interpretation or publication.

Do not modify mask values, locked configs or historical numbers to match an
assumed order. Preserve numeric-ID results and recover the actual preprocessing
mapping or original source dataset metadata. Standard test-case identity does
not by itself prove class-name semantics. New raw aggregate metrics remain
numerically valid irrespective of naming, but named-organ claims are gated.

All1568 local processed test masks exactly match PIL-nearest resizing of the
original H5 numeric labels (validation/native_synapse_labels.csv). This verifies
label-ID consistency through conversion, not anatomical name semantics.
