"""Constants, config generation and the strict config check for the ACDC replication.

Pre-registration: preregistration/acdc_confirmation.md. Every frozen file of the parent KD study
(scripts/kd_mc/, scripts/kd/, scripts/revision/) is imported unchanged; nothing here edits them.
"""

from __future__ import annotations

import copy
from pathlib import Path

from scripts.kd.common import sha
from scripts.kd_mc.register_mc import student_config

ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "preregistration/acdc_confirmation.md"
PREREG_REL = "preregistration/acdc_confirmation.md"
LOG = ROOT / "revision/acdc_confirm/PROTOCOL_LOG.md"
FAILED = ROOT / "revision/acdc_confirm/failed_runs.json"
WORKSPACE = ROOT / "revision/acdc_confirm"
RUNS = ROOT / "results/revision_runs"
EVAL = ROOT / "results/acdc_confirm_eval"
CONFIGS = ROOT / "configs/revision/acdc_confirm"
QUEUE = ROOT / "revision/queue_acdc_confirm.yaml"
DATASET = "acdc"

SEEDS = tuple(range(201, 218))  # fixed in advance; N = 17 per arm
LAMBDA = {"kd_band": 1.0, "decoder": 0.0}  # the only difference between the two arms

# Held-out official ACDC test patients (101-150), preprocessed by official_test.py.
OFFICIAL_RAW = Path("/home/jc/bds-followup/data/raw/acdc_official_test")
OFFICIAL_PROCESSED = ROOT / "data/processed/acdc_official_test"
OFFICIAL_MANIFEST = WORKSPACE / "official_test_manifest.json"
OFFICIAL_PREPROCESSING = WORKSPACE / "official_test_preprocessing.json"

# Set once, before the freeze commit (pre-registration section 2):
# "official_test" if patients 101-150 passed integrity and preprocessing, else "internal_test".
PRIMARY_SET = "official_test"

# Exploratory runs of the parent study (reference only; never retrained). best.pt sha256.
EXPLORATORY = {
    "kd_band": {
        "acdc_kd_band_seed1": "d3d704e439a188bf46440dc700a95f9ccb403848c6166b87dfcae962e14ab038",
        "acdc_kd_band_seed2": "4b2fe9a1f82a2861bf79e68e826908ceec7e028c65b12e1e7f74f1398359364a",
        "acdc_kd_band_seed3": "2c376074a64d2797b4f956d6f5db913b23c4d55b6588fd76112be8ca4b7c3920",
    },
    "decoder": {
        "acdc_m2_aux_boundary_only_unweighted_seed1": (
            "96d21b73e5ab22c6fbb9ab7f4141c478f19024d4b924ca5a407fd300dd36e56b"
        ),
        "acdc_m2_aux_boundary_only_unweighted_seed2": (
            "5d66a6faa5e154f6fd22853c6ea1991a9fe6cd42b9ce3e6df87bf9503510caa2"
        ),
        "acdc_m2_aux_boundary_only_unweighted_seed3": (
            "ba888f9fab0c78e8ccdca955c234e8ff47d2f051f95e4c859c6aa9219217faf2"
        ),
    },
}
TEACHER_BEST_SHA256 = "735747303c444e13dd57457e3540ee410b5de43421b68fd92e2eda69c88dafa0"
TEACHER_CACHE_SHA256 = "42266ea089e495d651cf654ae7c5c287a2b40d77cc1131ea78f3a0d059a6077e"


def registered_runs() -> list[tuple[str, int]]:
    """Run order: seed by seed, decoder-only then D+KD-band."""
    return [(arm, seed) for seed in SEEDS for arm in ("decoder", "kd_band")]


def run_name(arm: str, seed: int) -> str:
    return f"acdc_confirm_{arm}_seed{seed}"


def make_config(arm: str, seed: int, protocol_sha: str) -> dict:
    """The parent study's frozen student config.

    Only project.seed, revision.* and (decoder-only arm) kd.lambda_kd differ.
    """
    config = copy.deepcopy(student_config(DATASET, seed, protocol_sha))
    config["kd"]["lambda_kd"] = LAMBDA[arm]
    config["revision"]["purpose"] = f"ACDC replication {arm}; see {PREREG_REL}"
    config["revision"]["protocol"] = PREREG_REL
    config["revision"]["protocol_sha256"] = protocol_sha
    config["revision"]["arm"] = f"acdc_confirm_{arm}"
    return config


def arm_of(config: dict) -> str:
    arm = config.get("revision", {}).get("arm", "")
    if not arm.startswith("acdc_confirm_") or arm.removeprefix("acdc_confirm_") not in LAMBDA:
        raise ValueError(f"not an ACDC replication config: {arm!r}")
    return arm.removeprefix("acdc_confirm_")


def validate(config: dict, allowed_seeds=SEEDS, check_protocol_file: bool = True) -> str:
    """Config must equal make_config for its arm and seed, byte for byte after YAML parsing."""
    arm = arm_of(config)
    seed = config["project"]["seed"]
    if seed not in allowed_seeds:
        raise ValueError(f"seed {seed} is not admitted")
    expected = make_config(arm, seed, config["revision"]["protocol_sha256"])
    if config != expected:
        raise ValueError("config differs from the pre-registered generator output")
    if check_protocol_file and sha(PREREG) != config["revision"]["protocol_sha256"]:
        raise ValueError("pre-registration changed after the config was generated")
    return arm


def failed_runs() -> dict[str, dict]:
    """Runs declared failed under the pre-registered policy (section 5), by run name."""
    if not FAILED.exists():
        return {}
    import json

    return {r["run"]: r for r in json.loads(FAILED.read_text())["runs"]}


def log_event(kind: str, text: str) -> str:
    """Append a timestamped entry to revision/acdc_confirm/PROTOCOL_LOG.md; returns the stamp."""
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if not LOG.exists():
        LOG.parent.mkdir(parents=True, exist_ok=True)
        LOG.write_text(
            "# ACDC replication: protocol log\n\n"
            "Freeze record, operational events and every deviation from "
            "preregistration/acdc_confirmation.md, with UTC timestamps. Append-only.\n\n"
            "| UTC | Kind | Entry |\n|---|---|---|\n"
        )
    with LOG.open("a") as stream:
        stream.write(f"| {stamp} | {kind} | {text.replace('|', '/')} |\n")
    return stamp
