# Revision training adapter and WP5 diagnostic execution

This is a technical execution specification for the authoritative plan, frozen
before new scientific training. Existing U seed1 rerun already passes WP0;
existing main U/F results are retained. Three new full-model seed1 runs are
required only for missing every-50-step gradient trajectories. They are not
additional independent seeds and will not replace historical main effects.
No architecture search or failed followup route is reopened.

Locked: exact recovered split-seed2026 manifests, training seed1, 224 input,
AdamW wd1e-5, cosine150 epochs, clip1, AMP, no augmentation; ISIC batch8/lr3e-4,
ACDC/Synapse batch16/lr1e-4, best validation DSC with strict-greater tie rule.
Read existing stored image/boundary/SDF arrays, without zscore or rewriting data.
All 5,163 training targets match the current morphology/SDF definitions exactly.
ISIC uses development519 for selection and official1000 only after selection.
ACDC/Synapse use frozen val/test. Never interpret spacing as verified mm.

Model width32, union boundary head, retained gate alpha0.25, boundary/distill
weights0.05, SDF weight1; CE/Dice1. ACDC preserves historical foreground weights
[1.5,1,1.25]; other datasets have no class weighting. Target-detached feature
matching follows the actual active loss implementation. New resolved configs
explicitly use detach_boundary_features=false; stale unused legacy flags such as
use_stop_gradient are omitted and their removal is recorded, not interpreted as
an implemented historical switch. This is a well-defined current-code diagnostic;
historical source/ablation provenance uncertainty remains documented separately.

Every50 global steps, before the actual AMP forward/update, compute FP32 gradients
on the same training batch for weighted SDF, boundary BCE+Dice, detached-target
SmoothL1, their sum, and CE+Dice on shared encoder parameters. Record norms,
cosines and norm ratios. Zero-norm cosine is undefined/null, never fabricated as
zero correlation. Snapshot/restore Python/NumPy/CPU/CUDA RNG and all model buffers;
autograd.grad must leave parameter .grad unchanged. Tests compare the following
optimizer update with/without the probe, including dropout/BatchNorm. Real full
GPU batch checks pass exactly for all three datasets (peak allocated <4.1GB).
The first GPU smoke exposed constant zero SDF on an empty Synapse batch; constant
terms now correctly have zero gradients/null cosine. Original failure log kept.
These short engineering checks are not gradient-trajectory evidence.

Resume at completed epoch boundaries. Save atomic last.pt at epoch1, every10,
150 and a requested stop, including model/optimizer/scheduler, best state/epoch,
all RNG, shuffle-generator state, history/probes, config/code/data hashes.
An abrupt crash can replay up to nine unsaved epochs; never combine duplicated
observations. Resume refuses changed config/code/data. GradScaler resets each
epoch, matching the verified released runner, so no mid-epoch scaler resume is
claimed. More than eight consecutive AMP overflows or nonfinite loss stops the
run; standard AMP skipped updates are recorded with sample IDs and scale.

Only run_queue.py launches training. The flock descriptor is inherited by the
child, retaining exclusion if the parent exits. Refuse an occupied GPU. Before
launch require <80C, <1GB used and >=25GiB free. Log temperature/clocks/power/VRAM
every10s; >=85C, <20GiB free or revision/STOP_TRAINING requests SIGTERM and an
end-of-epoch checkpoint. Never signal unrelated processes. Incomplete unknown
outputs are preserved, never overwritten automatically. Final marker exists only
after150 epochs, best.pt, config, val/test maps, all five metrics and hashes.

Engineering acceptance: CPU exact epoch-resume with AdamW/cosine/shuffle and RNG,
probe neutrality/detached targets/constant zero cases, corruption rejection;
full stored-target audit; full-batch AMP probe neutrality on RTX5060. Queue
registration/config hashes and this document are committed before execution.
