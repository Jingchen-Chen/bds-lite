"""Epoch-resume state and observational gradient probes for revision training."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from bds_lite.data import NpzSegmentationDataset


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def atomic_torch_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class ManifestDataset(NpzSegmentationDataset):
    """Use unchanged stored targets and exact frozen IDs; never scan into extras."""

    def __init__(self, root, split, manifest, classes, channels, boundary=False, sdf=False):
        super().__init__(root, split, require_boundary=boundary, require_sdf=sdf)
        ids = manifest["sample_ids"]
        if manifest["seed"] != 2026 or len(ids) != manifest["count"] or len(set(ids)) != len(ids):
            raise ValueError("invalid frozen split manifest")
        if ids != sorted(ids) or any(Path(i).name != i for i in ids):
            raise ValueError("split IDs must be sorted basenames")
        self.files = [Path(root) / split / f"{i}.npz" for i in ids]
        if not all(p.is_file() for p in self.files):
            raise FileNotFoundError("frozen sample missing")
        self.classes, self.channels = classes, channels

    def __getitem__(self, index):
        item = super().__getitem__(index)
        image, mask = item["image"], item["mask"]
        if image.shape != (self.channels, 224, 224) or mask.shape != (224, 224):
            raise ValueError("input shape differs from locked protocol")
        if not torch.isfinite(image).all() or mask.min() < 0 or mask.max() >= self.classes:
            raise ValueError("invalid image or class ID")
        for key in ["boundary", "sdf"]:
            if key in item and not torch.isfinite(item[key]).all():
                raise ValueError(f"nonfinite {key}")
        if self.require_boundary and item["boundary"].shape != (1, 224, 224):
            raise ValueError("diagnostic full models require union boundary targets")
        if self.require_sdf and item["sdf"].shape != (self.classes - 1, 224, 224):
            raise ValueError("SDF class shape mismatch")
        return item


def loss_components(outputs, batch, criterion):
    """The exact active CE/Dice + weighted SDF/boundary/distillation objective."""
    w = criterion.weights
    if w.use_gsl or w.distance or w.far_fp:
        raise ValueError("gradient probe only supports the registered BDS objective")
    logits = outputs["seg_logits"]
    zero = logits.sum() * 0
    components = {
        "seg": w.ce * F.cross_entropy(logits, batch["mask"])
        + w.dice * criterion.dice(logits, batch["mask"]),
        "sdf": w.surface * criterion.surface(logits, batch["sdf"]) if w.surface else zero,
        "boundary": zero,
        "distill": zero,
    }
    if w.boundary:
        components["boundary"] = w.boundary * (
            F.binary_cross_entropy_with_logits(outputs["boundary_logits"], batch["boundary"])
            + criterion.boundary_dice(outputs["boundary_logits"], batch["boundary"])
        )
    if w.distill:
        components["distill"] = w.distill * F.smooth_l1_loss(
            outputs["seg_boundary_features"], outputs["boundary_features"].detach()
        )
    total, _ = criterion(outputs, batch)
    if not torch.allclose(sum(components.values()), total, rtol=1e-6, atol=1e-7):
        raise AssertionError("probe objective differs from active training loss")
    components["aux"] = components["sdf"] + components["boundary"] + components["distill"]
    return components


def gradient_probe(model, batch, criterion):
    """FP32 train-mode probe; restore all buffers/RNG, leave .grad untouched."""
    if not model.training:
        raise ValueError("probe requires training mode")
    state = rng_state()
    buffers = {name: b.detach().clone() for name, b in model.named_buffers()}
    parameters = [p for p in model.encoder.parameters() if p.requires_grad]
    try:
        with torch.enable_grad(), torch.amp.autocast(batch["image"].device.type, enabled=False):
            components = loss_components(model(batch["image"].float()), batch, criterion)
            gradients = {}
            for name, loss in components.items():
                gs = (
                    torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
                    if loss.requires_grad
                    else [None] * len(parameters)
                )
                gradients[name] = torch.cat(
                    [
                        (g if g is not None else torch.zeros_like(p)).flatten()
                        for g, p in zip(gs, parameters, strict=True)
                    ]
                ).detach()
            seg = gradients["seg"].double()
            norm = torch.linalg.vector_norm(seg)
            result = {"seg_norm": float(norm), "precision": "fp32_pre_update"}
            for name in ["sdf", "boundary", "distill", "aux"]:
                other = gradients[name].double()
                other_norm = torch.linalg.vector_norm(other)
                result[name + "_norm"] = float(other_norm)
                result[name + "_cosine"] = (
                    float(torch.dot(seg, other) / (norm * other_norm))
                    if norm > 0 and other_norm > 0
                    else None
                )
                result[name + "_norm_ratio"] = float(other_norm / norm) if norm > 0 else None
            if any(
                v is not None and not np.isfinite(v) for k, v in result.items() if k != "precision"
            ):
                raise FloatingPointError("nonfinite gradient probe")
            return result
    finally:
        with torch.no_grad():
            for name, buffer in model.named_buffers():
                buffer.copy_(buffers[name])
        restore_rng(state)
