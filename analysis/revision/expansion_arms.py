"""Configuration builders for the expansion runs; shared by the gate and the registration."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
ZERO = dict(boundary_weight=0.0, distill_weight=0.0, surface_weight=0.0, distance_weight=0.0)


def _template(name: str) -> dict:
    return yaml.safe_load((ROOT / f"configs/revision/{name}.yaml").read_text())


def isic_unet(seed: int) -> dict:
    """U-Net: the decoder-only configuration with the auxiliary path removed."""
    config = _template("isic2018_m2_aux_boundary_only_seed1")
    config["model"] = dict(name="unet", base_channels=32)
    config["loss"].update(ZERO)
    return _finish(config, seed, "m0_unet")


def isic_decoder(seed: int) -> dict:
    return _finish(_template("isic2018_m2_aux_boundary_only_seed1"), seed, "m2_aux_boundary_only")


def isic_full(seed: int) -> dict:
    """BDS-Lite: the decoder-only configuration with surface, distillation, and gate enabled."""
    config = _template("isic2018_m2_aux_boundary_only_seed1")
    config["model"]["boundary_gate_scale"] = 0.25
    config["loss"].update(distill_weight=0.05, surface_weight=1.0)
    return _finish(config, seed, "m6_full")


def isic_gsl(seed: int) -> dict:
    return _finish(_template("isic2018_gsl_seed1"), seed, "gsl")


def decoder_surface(dataset: str, seed: int) -> dict:
    """Missing 2x2 cell: decoder and surface loss; no distillation, no gate; no class weights."""
    config = _template(f"{dataset}_m2_aux_boundary_only_seed1")
    config["loss"].update(surface_weight=1.0, foreground_class_weights=None)
    return _finish(config, seed, "m7_decoder_surface")


def acdc_decoder_unweighted(seed: int) -> dict:
    """Corrects the class-weighted ACDC decoder-only arm."""
    config = _template("acdc_m2_aux_boundary_only_seed1")
    config["loss"]["foreground_class_weights"] = None
    return _finish(config, seed, "m2_aux_boundary_only_unweighted")


def _finish(config: dict, seed: int, arm: str) -> dict:
    config = copy.deepcopy(config)
    config["project"]["seed"] = seed
    config["revision"]["arm"] = arm
    config["revision"]["gradient_probe"] = False
    for key in ("historical_config_source", "historical_checkpoint_sha256"):
        config["revision"].pop(key, None)
    return config
