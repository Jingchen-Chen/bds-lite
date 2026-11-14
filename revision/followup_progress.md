# Follow-up controls: progress log

Registration: `revision/followup_registration.md`. Protocol log (append-only, UTC):
`revision/followup/PROTOCOL_LOG.md`. Plan: `PLAN.md`, Phase 20.

## 2026-11-09: preparation

- Branch `revision-followup` created from `revision-restructure` (`ce62fd1`).
- New code: `scripts/kd_followup/` (hash-registered; not edited after registration),
  `analysis/followup/` (analysis side; not registered), `tests/revision/test_kd_followup.py`.
- CPU smoke tests before registration (scratch outputs only, GPU hidden, `nice 19`):
  - E2 cache step (`cache_peer.py`) on an existing D run as a stand-in teacher: passed
    (cached-vs-online max difference 2.4e-4 < 1e-3; validation consistency < 2e-5).
  - One training epoch each of `kd_full`, `kd_band_peer` (reading the stand-in cache) and
    `kd_band_aug` through the launcher and the frozen `train_mc.py`, including the KD gradient
    probes and the final validation evaluation: all passed, finite losses (probe L_kd at epoch 0:
    1.23 KD-full, 1.47 compact-teacher band, 1.46 augmented band).
  - Bug found and fixed before registration: the augmentation dataset class was defined inside a
    function, which cannot be pickled for Python 3.14's forkserver DataLoader workers; it is now a
    module-level class (regression test added).
- **GPU shared with another project.** At 2026-11-09 ~19:10 CET a separate ROS/Gazebo experiment
  (`rgb_vs_rgbd_ablation`, real-time simulation with a GPU detector node) was running, at
  101/144 single episodes plus 4 multi-class missions remaining. Because that experiment runs in
  real time, sharing the GPU (or heavy CPU load) could change its results. Decision: do not train
  beside it. `analysis/followup/wait_and_launch.sh` launches the supervisor only after no compute
  process has used the GPU for 10 consecutive minutes.
- At ~19:38 CET the other experiment's own status file read `"status": "paused",
  "user_requested": true` (paused 18:31 UTC at 103/144); its controller had exited and the GPU
  had no compute process. Training therefore starts right after the registration commit and its
  timestamp.

## Training groups

(updated after each group)
- 2026-11-09T19:23Z, Amendment 1 (author decision): the hold for a Bitcoin attestation was
  removed before training. The attestation-wait loop was stopped before it reached the launcher;
  no training had started. The earlier sentence "starts right after the registration commit and
  its timestamp" is superseded: training starts without waiting for any blockchain confirmation.
  Pending `.ots` files are kept as historical artifacts (never upgraded or verified).

### E1 (D+KD-full), in progress
- 2026-11-09T22:37Z: 6/17 E1 runs complete (seeds 201–206; selected epochs 23–51, validation
  DSC 0.790–0.802), run 207 at epoch 137; no non-finite loss. Wall time ≈ 30 min per run.
- Abnormality (operational): the laptop GPU runs at 84–85 °C, so the registered thermal guard
  stopped the queue at an epoch boundary 80 times in ~3 h (Stage 3: 54 times over all 34 runs).
  Each stop saves the epoch and the supervisor resumes after cooling, with the frozen config,
  source and data hash checks on resume. No run failed; no intervention.
- **E1 complete 2026-11-10T03:04Z**: 17/17 D+KD-full runs (seeds 201–217), no failure, no
  non-finite loss; selected epochs 19–55; summed epoch time 6.45 h; wall time 19:35Z–03:04Z
  (7.5 h, including thermal pauses). No test data touched.

### E2 (compact teacher), in progress
- E2 teacher `acdc_followup_e2_teacher_seed100` complete (selected epoch 20; validation DSC
  0.8025, Boundary F1 0.8457; the external teacher's recorded validation values are DSC 0.8419,
  Boundary F1 0.9014). Cache step complete 03:31Z, gate `NOT_APPLICABLE` as registered,
  cached-vs-online max difference 2.5e-4 (< 1e-3).
- 2026-11-10T04:38Z: E2 students 201–202 complete, 203 at epoch 74. Thermal-guard resumes so
  far: 246 (all end-of-epoch, hash-checked resumes; no failure).
- **E2 complete 2026-11-10T11:08:50Z**: 17/17 D+KD-band(compact) runs, no failure, no
  non-finite loss; summed epoch time 6.51 h. Whole E1+E2 queue: 36/36 entries, wall time
  19:35Z–11:09Z (15.6 h), 431 guarded thermal resumes.
- Integrity (`analysis/followup/integrity.py --queue e1e2`): PASSED.
- One-time test evaluation: 35 runs, 11:10:34Z–11:19:43Z
  (`results/followup_eval/evaluation_record_e1e2.json`).
- Analysis (`analysis/followup/analyze.py`): C1 U, C2 P, C3 P, C4 U → band case B3, teacher
  case T2 (`revision/followup/results/results.json`).

### E3 (optional augmentation)
- Start condition judged met at 11:30Z (see PROTOCOL_LOG); launched through
  `wait_and_launch.sh e3` with the same guards.
- **E3 stopped 2026-11-10T12:40Z** after 2 of 20 runs (`decoder_aug_seed201`,
  `kd_band_aug_seed201` complete; `decoder_aug_seed202` checkpointed at epoch 74). The stop
  marker `revision/STOP_KD_TRAINING` appeared at 12:40:18Z as an empty file. This agent, its
  scripts and its tests did not create it, and no other agent session was running. It is
  treated as an explicit stop request and left in place. E3 runs were not evaluated on the test
  patients. To resume, remove the marker and run `bash analysis/followup/wait_and_launch.sh e3`;
  the supervisor resumes from the saved checkpoints.
- Correction (2026-11-10): the stop marker was placed deliberately by a separate agent session
  running an evidence-freeze audit (recorded in PLAN.md, Phase 20). That audit is **blocked**
  because the manuscript now describes a D+KD-full control while the governing
  `revision/CLAIM_BOUNDARY_v2.md` (F1) says none exists. The author must approve the
  claim-boundary addendum, or decide otherwise, before the manuscript is governed.
