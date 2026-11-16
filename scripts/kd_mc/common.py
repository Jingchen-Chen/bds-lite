"""Constants and locked checks for the ACDC/Synapse KD study.

Protocol: protocols/kd_acdc_synapse/KD_MULTICLASS_PROTOCOL.md.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from scripts.kd.common import (
    SMP_VERSION,
    WEIGHTS_REPO,
    WEIGHTS_REVISION,
    WEIGHTS_SHA256,
    sha,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "protocols/kd_acdc_synapse/KD_MULTICLASS_PROTOCOL.md"
WORKSPACE = ROOT / "results/kd_acdc_synapse"
RUNS = ROOT / "results/revision_runs"
STOP = ROOT / "revision/STOP_KD_TRAINING"  # shared KD stop marker; STOP_TRAINING is Experiment C's
GPU_LOCK = ROOT / "revision/.gpu.lock"
TEACHER_SEED = 100
STUDENT_SEEDS = (1, 2, 3)

# The registered decoder-only arm of each dataset (seeds 1-3; ACDC: the corrected unweighted runs).
DATASETS = {
    "acdc": dict(
        decoder_arm="m2_aux_boundary_only_unweighted",
        template="configs/revision/acdc_m2_aux_boundary_only_unweighted_seed1.yaml",
        input_norm="none",  # stored per-volume z-scored MRI, already on the normalised scale
    ),
    "synapse": dict(
        decoder_arm="m2_aux_boundary_only",
        template="configs/revision/synapse_m2_aux_boundary_only_seed1.yaml",
        input_norm="imagenet",  # stored windowed CT in [0, 1]
    ),
}

KD_SPEC = dict(
    lambda_kd=1.0,
    target="teacher_softmax",
    loss="soft_cross_entropy",
    band_labels="union_of_foreground_classes_of_teacher_argmax",
    band_kernel=3,
    structuring_element="square",
    border_value=0,
    normalization="batch",
    eps=1e-6,
    clamp=1e-6,
)


def teacher_run(dataset: str) -> str:
    return f"{dataset}_kd_teacher_seed{TEACHER_SEED}"


def cache_run(dataset: str) -> str:
    return f"{dataset}_kd_teacher_cache"


def student_run(dataset: str, seed: int) -> str:
    return f"{dataset}_kd_band_seed{seed}"


def decoder_runs(dataset: str) -> list[Path]:
    arm = DATASETS[dataset]["decoder_arm"]
    return [RUNS / f"{dataset}_{arm}_seed{s}" for s in STUDENT_SEEDS]


def template(dataset: str) -> dict:
    return yaml.safe_load((ROOT / DATASETS[dataset]["template"]).read_text())


def teacher_model(dataset: str) -> dict:
    t = template(dataset)
    return dict(
        name="smp_unet_resnet34",
        in_channels=t["dataset"]["in_channels"],
        num_classes=t["dataset"]["num_classes"],
        input_adapter="replicate_to_3_channels",
        input_norm=DATASETS[dataset]["input_norm"],
    )


def teacher_kd_block() -> dict:
    return dict(
        role="teacher",
        encoder="resnet34",
        encoder_weights="imagenet",
        smp_version=SMP_VERSION,
        weights_repo=WEIGHTS_REPO,
        weights_revision=WEIGHTS_REVISION,
        weights_sha256=WEIGHTS_SHA256,
    )


def kd_sources() -> dict[str, str]:
    """Source hashes recorded by every run: existing training code, both KD packages, evaluator."""
    folders = ["src/bds_lite", "scripts/revision", "scripts/kd", "scripts/kd_mc"]
    hashes = {
        str(p.relative_to(ROOT)): sha(p)
        for folder in folders
        for p in (ROOT / folder).rglob("*.py")
    }
    for extra in [
        "analysis/revision/recover_predictions.py",
        "analysis/revision/expansion_statistics.py",
        "analysis/revision/family_statistics.py",
    ]:
        hashes[extra] = sha(ROOT / extra)
    return dict(sorted(hashes.items()))


def require_protocol(config: dict) -> None:
    """Students equal the dataset's decoder-only recipe except the kd block; teacher is fixed."""
    dataset = config["dataset"]["name"]
    if dataset not in DATASETS:
        raise ValueError("multiclass KD study is ACDC/Synapse only")
    kd = config.get("kd") or {}
    base = template(dataset)
    for block in ["data", "dataset"]:
        if config[block] != base[block]:
            raise ValueError(f"{block} block differs from the decoder-only recipe")
    if config["revision"]["manifests"] != base["revision"]["manifests"]:
        raise ValueError("split manifests differ from the decoder-only recipe")
    if config.get("augmentation") or config["data"].get("augmentation"):
        raise ValueError("online augmentation is not permitted")
    if kd.get("role") == "student":
        if config["project"]["seed"] not in STUDENT_SEEDS:
            raise ValueError("student seed must be 1-3")
        for block in ["model", "loss", "trainer"]:
            if config[block] != base[block]:
                raise ValueError(f"student {block} block differs from the decoder-only recipe")
        if {k: kd.get(k) for k in KD_SPEC} != KD_SPEC or kd.get("mode") != "band":
            raise ValueError("student kd parameters differ from the frozen specification")
        if kd.get("cache_run") != cache_run(dataset):
            raise ValueError("student must read its dataset's registered teacher cache")
    elif kd.get("role") == "teacher":
        if config["project"]["seed"] != TEACHER_SEED:
            raise ValueError(f"teacher seed must be {TEACHER_SEED}")
        expected_trainer = copy.deepcopy(base["trainer"])  # batch 16, lr 1e-4, wd 1e-5
        if config["trainer"] != expected_trainer:
            raise ValueError("teacher trainer block differs from the protocol")
        if config["loss"] != dict(base["loss"], boundary_weight=0.0):
            raise ValueError("teacher loss must be CE + Dice only")
        if kd != teacher_kd_block() or config["model"] != teacher_model(dataset):
            raise ValueError("teacher block differs from the frozen specification")
    else:
        raise ValueError("config lacks a valid kd block")
    trainer = config["trainer"]
    if trainer.get("init_from_checkpoint") or trainer.get("freeze_encoder_epochs", 0):
        raise ValueError("initialization/freezing not in locked protocol")
