# Evidence addendum 2026-11-11 — frozen-teacher evaluation (post hoc, descriptive)

Versioned addendum to `revision/CLAIM_BOUNDARY_v4.md` and `revision/BDS_LITE_EVIDENCE_FREEZE.md`,
which require "a versioned evidence addendum and appropriate authorization" before a new analysis
enters the manuscript. Neither file is modified; every registered protocol, estimator, endpoint,
decision and C01–C14 reading is unchanged.

## Authorization

The author's manuscript instruction of 2026-11-11 asks for (i) one sentence at the end of the
external-teacher Results section giving the teachers' scores (external teacher above every
student run; compact teacher not above D), marked post hoc and descriptive, with the full table in
a new supplementary section, and (ii) one sentence in the augmentation Results section stating that
the external teacher still leads D_aug. That instruction is the authorization for this addendum.

## Source

- Protocol and verbatim interpretation rules: `results/teacher_eval/PROTOCOL.md`, committed in
  `0071208` before any new teacher inference. Execution and result: `8e420c8`.
- Values: `results/teacher_eval/summary.json`, `results/teacher_eval/REPORT.md`.
- Manuscript formatting: `analysis/revision/make_final_assets.py::teacher_eval` (keys `TE.*`,
  `generated/tab_s_teacher_{scores,groups}.tex`), checked by `verify_final_assets.py`.

## Added evidence (C15, C16)

| ID | Supported statement and evidence | Statistical/practical reading and scope |
|---|---|---|
| C15 | On official ACDC patients 101–150 the frozen external teacher scored patient-level Boundary F1 0.8778, above all 88 student run means (0 runs exceed it); the compact teacher scored 0.7735, below the D mean (teacher − D −0.0062 [−0.0160, +0.0037]; 14/17 D runs above it). External teacher − D_aug +0.0309 [+0.0158, +0.0454]; pre-specified outcome B: "This explanation is not supported." `results/teacher_eval/summary.json` | Post hoc, descriptive, inference only; one frozen realization per teacher; patients previously examined, and both teachers had earlier one-time evaluations on them (the external record was opened before the rules were written). Intervals are percentile bootstraps over patients conditional on the evaluated runs (10,000 draws), not the run-level estimator; no significance test. Does not explain the KD effect, does not establish teacher quality as a cause (C09), and cannot replace any registered reading |
| C16 | Post hoc proportional-transfer check (definition chosen by the author on 2026-11-11 after the results): registered gain +0.0178204 / (external teacher − D) +0.0980970 = 18% of the teacher's lead closed; 18% of (external teacher − D_aug) +0.0309062 = +0.0056, against the E3 Boundary F1 upper bound +0.0044. Keys `TE.prop.*`, `make_final_assets.py::teacher_eval` | Point estimates only; ignores the uncertainty of the gain, of both teacher leads and of the E3 interval; defined after all results were seen. May be described as narrowly excluding proportional transfer at face value; not a test and not evidence that augmentation removed the effect (C13) |

## Wording limits

Use "post hoc" and "descriptive" with every mention. Do not state that teacher quality causes,
explains or limits the distillation gain, and do not present the teacher–student differences as
tests. The pre-specified outcome sentence may be quoted only for the explanation it was written
for (augmented students outperforming the teacher).
