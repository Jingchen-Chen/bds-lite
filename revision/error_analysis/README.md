# Error and small-structure analysis

Reproduce with error_decomposition.py followed by augment_error_distances.py.
The second stage joins exact validated per-case distance rows; it does not rerun
training or inference. Outputs preserve one method/seed/image/class row and
per-seed/class/area summaries. No seed-image independence is claimed.

Area bins separate empty GT, positive <1%,1-2%,2-5%,>5% masks. Missed structures
are conditional on nonempty GT; false alarms on empty GT. Near/far FP/FN use
distance to the inner GT contour at2 processed pixels. Empty-GT has no contour;
its false foreground enters the explicitly labeled far/no-contour group, not a
measured physical distance. Predicted/spurious components use4-connectivity.

HD95/ASSD bin means retain valid-case and missing-case counts. Full-distance
CDFs are now available in ../surface_cdf, with eligibility, complete ECDF tables
and independent checks. Future comparator refresh/manuscript integration remain.
Synapse anatomy names remain gated; numeric IDs are not relabeled from guesses.
