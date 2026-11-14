"""Constants, config generation and the strict config check for the follow-up controls.

Registration: revision/followup_registration.md. Every config is the Stage 3 generator output
(scripts/acdc_confirm/common.make_config) with only the keys listed in ALLOWED changed; nothing in
scripts/acdc_confirm/, scripts/kd_mc/, scripts/kd/, scripts/revision/ or src/ is edited.

Arms (run name acdc_followup_<arm>_seed<N>):
- kd_full (E1): D+KD-band with the band replaced by every pixel (same teacher cache, lambda, clamp).
- e2_teacher (E2): the compact teacher; the Stage 3 decoder-only (D) config with seed 100.
- e2_cache (E2): the frozen softmax cache of e2_teacher on the training images (queue kind cache).
- kd_band_peer (E2): D+KD-band reading the e2_teacher cache instead of the external teacher's.
- decoder_aug, kd_band_aug (E3, optional): D and D+KD-band with uniform D4 training augmentation.
"""

from __future__ import annotations

import copy
from pathlib import Path

from scripts.acdc_confirm.common import make_config as stage3_config
from scripts.kd.common import sha

ROOT = Path(__file__).resolve().parents[2]
REGISTRATION = ROOT / "revision/followup_registration.md"
REGISTRATION_REL = "revision/followup_registration.md"
WORKSPACE = ROOT / "revision/followup"
LOG = WORKSPACE / "PROTOCOL_LOG.md"
FAILED = WORKSPACE / "failed_runs.json"
RUNS = ROOT / "results/revision_runs"
EVAL = ROOT / "results/followup_eval"
CONFIGS = ROOT / "configs/revision/followup"
QUEUES = {
    "e1e2": ROOT / "revision/queue_followup_e1e2.yaml",
    "e3": ROOT / "revision/queue_followup_e3.yaml",
}

SEEDS = tuple(range(201, 218))  # the Stage 3 seeds; N = 17 per arm
E3_SEEDS = tuple(range(201, 211))  # 10 per arm
PEER_TEACHER_SEED = 100
ORIGINAL_CACHE_RUN = "acdc_kd_teacher_cache"
PEER_TEACHER_RUN = f"acdc_followup_e2_teacher_seed{PEER_TEACHER_SEED}"
PEER_CACHE_RUN = "acdc_followup_e2_teacher_cache"

# arm -> (Stage 3 arm it is generated from, KD region, cache run, training augmentation)
ARMS = {
    "kd_full": ("kd_band", "full", ORIGINAL_CACHE_RUN, None),
    "e2_teacher": ("decoder", "band", ORIGINAL_CACHE_RUN, None),
    "kd_band_peer": ("kd_band", "band", PEER_CACHE_RUN, None),
    "decoder_aug": ("decoder", "band", ORIGINAL_CACHE_RUN, "uniform_D4"),
    "kd_band_aug": ("kd_band", "band", ORIGINAL_CACHE_RUN, "uniform_D4"),
}
ALLOWED_SEEDS = {
    "kd_full": SEEDS,
    "e2_teacher": (PEER_TEACHER_SEED,),
    "kd_band_peer": SEEDS,
    "decoder_aug": E3_SEEDS,
    "kd_band_aug": E3_SEEDS,
}


def run_name(arm: str, seed: int) -> str:
    return f"acdc_followup_{arm}_seed{seed}"


def make_config(arm: str, seed: int, registration_sha: str) -> dict:
    """Stage 3 config of the parent arm; only seed, revision.*, kd.mode and kd.cache_run differ."""
    parent, region, cache, augmentation = ARMS[arm]
    config = copy.deepcopy(stage3_config(parent, seed, registration_sha))
    config["kd"]["mode"] = region
    config["kd"]["cache_run"] = cache
    revision = config["revision"]
    revision["purpose"] = f"Follow-up control {arm} (exploratory); see {REGISTRATION_REL}"
    revision["protocol"] = REGISTRATION_REL
    revision["protocol_sha256"] = registration_sha
    revision["arm"] = f"acdc_followup_{arm}"
    revision["train_augmentation"] = augmentation
    return config


def arm_of(config: dict) -> str:
    arm = config.get("revision", {}).get("arm", "")
    if not arm.startswith("acdc_followup_") or arm.removeprefix("acdc_followup_") not in ARMS:
        raise ValueError(f"not a follow-up config: {arm!r}")
    return arm.removeprefix("acdc_followup_")


def validate(config: dict, check_registration_file: bool = True) -> str:
    """Config must equal make_config for its arm and seed, exactly, after YAML parsing."""
    arm = arm_of(config)
    seed = config["project"]["seed"]
    if seed not in ALLOWED_SEEDS[arm]:
        raise ValueError(f"seed {seed} is not admitted for {arm}")
    if config != make_config(arm, seed, config["revision"]["protocol_sha256"]):
        raise ValueError("config differs from the registered generator output")
    if check_registration_file and sha(REGISTRATION) != config["revision"]["protocol_sha256"]:
        raise ValueError("registration changed after the config was generated")
    return arm


def queue_runs(queue: str) -> list[tuple[str, int, str]]:
    """(arm, seed, kind) in launch order. E1 first, then the E2 teacher, its cache, E2 students."""
    if queue == "e1e2":
        runs = [("kd_full", s, "train") for s in SEEDS]
        runs += [
            ("e2_teacher", PEER_TEACHER_SEED, "train"),
            ("e2_teacher", PEER_TEACHER_SEED, "cache"),
        ]
        return runs + [("kd_band_peer", s, "train") for s in SEEDS]
    if queue == "e3":
        return [(arm, s, "train") for s in E3_SEEDS for arm in ("decoder_aug", "kd_band_aug")]
    raise ValueError(queue)


def failed_runs() -> dict[str, dict]:
    """Runs declared failed after one retry (registration section 6), by run name."""
    if not FAILED.exists():
        return {}
    import json

    return {r["run"]: r for r in json.loads(FAILED.read_text())["runs"]}


def log_event(kind: str, text: str) -> str:
    """Append a timestamped entry to revision/followup/PROTOCOL_LOG.md; returns the stamp."""
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if not LOG.exists():
        LOG.parent.mkdir(parents=True, exist_ok=True)
        LOG.write_text(
            "# Follow-up controls: protocol log\n\n"
            "Registration record, operational events and every deviation from "
            "revision/followup_registration.md, with UTC timestamps. Append-only.\n\n"
            "| UTC | Kind | Entry |\n|---|---|---|\n"
        )
    with LOG.open("a") as stream:
        stream.write(f"| {stamp} | {kind} | {text.replace('|', '/')} |\n")
    return stamp
