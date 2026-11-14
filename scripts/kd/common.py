"""Shared constants and helpers for the KD comparator (revision/KD_COMPARATOR_PROTOCOL.md).

Nothing here changes the existing training sources. The locked student recipe is read from the
registered decoder-only configuration and compared field by field, so the new arms cannot drift
from it silently.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "revision/KD_COMPARATOR_PROTOCOL.md"
WORKSPACE = ROOT / "revision/kd_comparator"
RUNS = ROOT / "results/revision_runs"
# KD training has its own stop marker. revision/STOP_TRAINING is the author's stop for Experiment C
# and is left in place; it must not be removed to run this experiment.
STOP = ROOT / "revision/STOP_KD_TRAINING"
GPU_LOCK = ROOT / "revision/.gpu.lock"
DECODER_TEMPLATE = ROOT / "configs/revision/isic2018_m2_aux_boundary_only_seed4.yaml"

TEACHER_SEED = 100
TEACHER_RUN = "isic2018_kd_teacher_seed100"
CACHE_RUN = "isic2018_kd_teacher_cache"
STUDENT_SEEDS = tuple(range(1, 11))
TORCH_VERSION, CUDA_VERSION = "2.11.0+cu130", "13.0"

# Teacher encoder weights: smp 0.6.0 pin for resnet34/imagenet, verified 2026-11-07.
SMP_VERSION = "0.6.0"
WEIGHTS_REPO = "smp-hub/resnet34.imagenet"
WEIGHTS_REVISION = "7a57b34f723329ff020b3f8bc41771163c519d0c"
WEIGHTS_SHA256 = "6549ccd46a2d20a0a74fca866cfc5c6f9b3672d468b0a92aa94a58545a61103e"
TEACHER_MODEL = dict(name="smp_unet_resnet34", in_channels=3, num_classes=2)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Frozen loss specification (protocol 4.2 and OPEN-5).
KD_SPEC = dict(
    lambda_kd=1.0,
    threshold=0.5,
    band_kernel=3,
    structuring_element="square",
    border_value=0,
    normalization="batch",
    eps=1e-6,
    clamp=1e-6,
)

# Teacher quality gate (protocol OPEN-1): means of the ten existing decoder-only runs on the 519
# validation images, at their DSC-selected checkpoints (Appendix A of the protocol).
GATE = dict(
    boundary_f1_min=0.5376352265487206,
    dsc_min=0.8631866601530269,
    source="mean over decoder-only seeds 1-10 of final_metrics.json metrics.val.<metric>.mean",
)
DECODER_RUNS = [RUNS / f"isic2018_m2_aux_boundary_only_seed{s}" for s in range(1, 11)]


def sha(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def decoder_template() -> dict:
    return yaml.safe_load(DECODER_TEMPLATE.read_text())


def kd_sources() -> dict[str, str]:
    """Source hashes recorded by every KD run: the existing training code and the KD package."""
    folders = ["src/bds_lite", "scripts/revision", "scripts/kd"]
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
    """Locked checks for KD configs.

    Students must equal the decoder-only recipe in every block except kd.
    """
    kd = config.get("kd")
    if not kd or kd.get("role") not in {"teacher", "student"}:
        raise ValueError("config lacks a valid kd block")
    if config["dataset"]["name"] != "isic2018":
        raise ValueError("KD comparator is ISIC2018 only")
    if config["data"]["image_size"] != [224, 224]:
        raise ValueError("input must be 224x224")
    if config.get("augmentation") or config["data"].get("augmentation"):
        raise ValueError("online augmentation is not permitted")
    template = decoder_template()
    for block in ["data", "dataset"]:
        if config[block] != template[block]:
            raise ValueError(f"{block} block differs from the decoder-only recipe")
    if config["revision"]["manifests"] != template["revision"]["manifests"]:
        raise ValueError("split manifests differ from the decoder-only recipe")
    trainer = config["trainer"]
    if kd["role"] == "student":
        if config["project"]["seed"] not in STUDENT_SEEDS:
            raise ValueError("student seed must be 1-10")
        for block in ["model", "loss", "trainer"]:
            if config[block] != template[block]:
                raise ValueError(f"student {block} block differs from the decoder-only recipe")
        if kd.get("mode") not in {"band", "full"}:
            raise ValueError("student kd.mode must be band or full")
        if {k: kd.get(k) for k in KD_SPEC} != KD_SPEC:
            raise ValueError("student kd parameters differ from the frozen specification")
        if kd.get("cache_run") != CACHE_RUN:
            raise ValueError("student must read the registered teacher cache")
    else:
        if config["project"]["seed"] != TEACHER_SEED:
            raise ValueError(f"teacher seed must be {TEACHER_SEED}")
        expected_trainer = copy.deepcopy(template["trainer"])
        expected_trainer["lr"] = 1e-4  # protocol OPEN-6
        if trainer != expected_trainer:
            raise ValueError("teacher trainer block differs from protocol OPEN-6")
        expected_loss = dict(template["loss"], boundary_weight=0.0)
        if config["loss"] != expected_loss:
            raise ValueError("teacher loss must be CE + Dice only")
        expected_kd = dict(
            role="teacher",
            encoder="resnet34",
            encoder_weights="imagenet",
            smp_version=SMP_VERSION,
            weights_repo=WEIGHTS_REPO,
            weights_revision=WEIGHTS_REVISION,
            weights_sha256=WEIGHTS_SHA256,
        )
        if kd != expected_kd:
            raise ValueError("teacher kd block differs from the frozen specification")
        if config["model"] != TEACHER_MODEL:
            raise ValueError("teacher model block differs from the frozen specification")
    if trainer.get("init_from_checkpoint") or trainer.get("freeze_encoder_epochs", 0):
        raise ValueError("initialization/freezing not in locked protocol")
