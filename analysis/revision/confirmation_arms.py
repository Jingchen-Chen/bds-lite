"""Configuration builders for Experiment C (protocols/confirmation/CONFIRMATION_PROTOCOL.md)."""

from __future__ import annotations

from analysis.revision import expansion_arms as base

ARM_NAMES = {
    "isic2018": {"unet": "m0_unet", "decoder": "m2_aux_boundary_only"},
    "acdc": {
        "unet": "m0_unet",
        "decoder": "m2_aux_boundary_only_unweighted",
        "full": "m6_full_unweighted",
        "gsl": "gsl",
    },
    "synapse": {
        "unet": "m0_unet",
        "decoder": "m2_aux_boundary_only",
        "full": "m6_full",
        "gsl": "gsl",
    },
}


def _decoder_template(dataset: str) -> dict:
    if dataset == "acdc":
        return base._template("acdc_m2_aux_boundary_only_unweighted_seed1")
    return base._template(f"{dataset}_m2_aux_boundary_only_seed1")


def build(dataset: str, arm: str, seed: int) -> dict:
    """Same recipe as the existing arm; only the seed differs."""
    name = ARM_NAMES[dataset][arm]
    if arm == "gsl":
        return base._finish(base._template(f"{dataset}_gsl_seed1"), seed, name)
    config = _decoder_template(dataset)
    if arm == "unet":
        config["model"] = dict(name="unet", base_channels=32)
        config["loss"].update(base.ZERO)
    elif arm == "full":
        config["model"]["boundary_gate_scale"] = 0.25
        config["loss"].update(distill_weight=0.05, surface_weight=1.0)
    return base._finish(config, seed, name)
