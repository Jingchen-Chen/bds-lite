# Experiment C scope amendment — 2026-11-07T11:52:53.219099+00:00

The author explicitly requested removal of ISIC2018 seeds 11–17 to reduce
training volume. This is an amendment after training began, not the original
frozen registration. No Experiment C test metrics were inspected for this change.

## Operational scope

- Original queue and protocol remain byte-preserved.
- Active queue: `revision/queue_confirmation_acdc_synapse.yaml`.
- Supervisor routing: `revision/queue_plan.yaml`.
- Retained: ACDC and Synapse, U-Net / decoder / BDS-Lite / GSL, seeds 4–10:
  2 datasets × 4 arms × 7 seeds = 56 runs.
- Removed from future scheduling: all 14 ISIC2018 entries, seeds 11–17.
- Already completed ISIC runs retained without deletion: isic2018_m0_unet_seed11, isic2018_m2_aux_boundary_only_seed11.
- Already started ISIC runs observed by directory existence: isic2018_m0_unet_seed11, isic2018_m2_aux_boundary_only_seed11.
- The remaining 12 ISIC tasks had no output directory at amendment time.
- Configs, training sources, checkpoints and predictions are unchanged.
- The running dispatcher caches its queue. The supervisor rereads routing on
  each attempt; activation must be verified at the next normal thermal restart.

## Analysis boundary

The original three-dataset fresh-seed primary test and 13-contrast secondary
family cannot be executed as registered after this cancellation. ISIC2018
seeds 1–10 remain the previously analysed evidence; completed seed-11 artifacts
must be disclosed but do not replace the planned seven-seed confirmation.

Before any Experiment C test-metric inspection, an explicit amended analysis
must be frozen and independently validated for ACDC and Synapse. Do not run
`analysis/revision/confirmation_statistics.py` or interpret its original
three-dataset primary test as fulfilled. Its frozen source remains unchanged
because registered training checks its hash. No revised statistical conclusion
is selected here. Existing manuscript conclusions are unchanged by this
operational amendment.

## Verification and repository limitations

Queue validation checks retained entries equal their original registrations,
source/config/split/protocol hashes, and the 56-run dataset/seed composition.
No training or evaluation command was manually launched.

The historical root plan is preserved. Its active anchor path and local
`.venv/bin/` are absent in this consolidated checkout; the archived anchor was
read at `archive/manuscript_v2_bspc/manuscript_v2/ANCHOR_ABSTRACT.md`.
This is a single operational scope change, not execution of a historical plan
phase. Required commit checks and any resulting blockers are recorded below.

- PASS: all 56 retained entries equal original entries; 114 registered file
  hashes verified, including configs, sources, splits and protocol.
- PASS: scoped `git diff --check`.
- BLOCKED: required `.venv/bin/ruff format --check .` and
  `.venv/bin/ruff check .` both exited 127 with
  `/bin/bash: line 1: .venv/bin/ruff: No such file or directory`.
- BLOCKED: required `.venv/bin/python -m pytest` exited 127 with
  `/bin/bash: line 1: .venv/bin/python: No such file or directory`.
- No commit attempted; required-check failure is not bypassed. Next human
  action: restore the repository-local environment for required commit checks.

## Superseding author decision — 2026-11-07T11:58:35.404671+00:00

The author subsequently requested stopping ALL remaining Experiment C
training, including ACDC and Synapse seeds 4–10. `revision/STOP_TRAINING`
is the persistent operational stop; automatic resume is forbidden. The
56-entry queue remains as provenance, not authorization to resume. Preserve
all completed and partial runs. Original confirmation remains incomplete.
The earlier 56-run continuation is superseded. No Experiment C test metrics
were read. Repository-local required-check blockers above remain; no commit
is attempted. Next action: verify epoch-boundary checkpoint and process exit,
then retain the stopped state pending further explicit author instructions.

Stop verified at 2026-11-07T11:59:06.290562+00:00: training process, dispatcher and supervisor exited; NVIDIA compute-process query empty. Synapse M0 seed 5 saved progress at epoch 112; last.pt exists (31379741 bytes). Explicit STOP exception is expected cancellation, not training failure. No automatic resume is permitted.
