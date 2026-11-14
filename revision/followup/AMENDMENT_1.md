# Follow-up controls, Amendment 1: external timestamping removed before training

**Date:** 2026-11-09, recorded 19:23:49 UTC (`PROTOCOL_LOG.md`).
**Decided by:** the author.
**Status of the experiment at the time:** no E1, E2 or E3 run had been launched or trained;
no follow-up run folder existed under `results/revision_runs/`; no follow-up model had touched
any test patient.

## What changes

1. The external timestamping step is removed from the active E1+E2 launch workflow. That
   includes OpenTimestamps submission, polling, `ots upgrade`, verification, and the hold of up
   to 90 minutes for a Bitcoin attestation recorded in `PROTOCOL_LOG.md` (entry
   2026-11-09T18:44:58Z).
2. Training does not wait for, and does not depend on, any blockchain confirmation.
3. No new `.ots` proof is generated for the follow-up. E3, if run, follows the same rule.

## What does not change

- The registration `revision/followup_registration.md` (sha256 `30f771eb…a5c3`, registration
  commit `f296023`) is unchanged. It did not itself contain an external-timestamping
  requirement; the requirement and the launch hold existed only in the operational workflow and
  the protocol log.
- The design is unchanged: arms, datasets, splits, seeds, configurations (55 files, hashes in the
  queues), primary and secondary endpoints, estimator, Holm family, decision categories and the
  interpretation mapping.
- The source, config and manifest hash checks of the queue runner, and the supervisor's GPU
  guards (no other compute process, memory below 1 GiB, 10 idle minutes before launch, cool
  launch, thermal stop, stop marker), are unchanged.

## Historical records kept

- `revision/followup/TIMESTAMP_MANIFEST.txt`, `revision/followup/TIMESTAMP_MANIFEST.txt.ots` and
  `revision/followup_registration.md.ots` (commit `affab9e`) stay in the repository as they were
  committed. They hold **pending** attestations from four calendars (stamped
  2026-11-09T18:44:38Z). They were never upgraded to a Bitcoin attestation and were never
  verified, and no such claim is made anywhere.
- The protocol-log entries before this amendment are not edited.

## What establishes the order of events instead

- Git history on branch `revision-followup` (local, never pushed): registration commit `f296023`
  before any follow-up training; commit dates are self-reported.
- UTC timestamps in `PROTOCOL_LOG.md`, the supervisor's `events.jsonl` and the per-run logs.
- Config, source and manifest sha256 values registered in the queues and checked by the queue
  runner before every launch and resume, and recorded in each run's `provenance.json`.

These are local records. They do not give the independent, externally verifiable proof of
existence that a confirmed timestamp would have given, and the manuscript does not claim one for
the follow-up.
