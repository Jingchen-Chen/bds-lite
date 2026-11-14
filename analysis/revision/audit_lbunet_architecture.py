"""Inspect LB-UNet parameter accounting and saved checkpoint shapes on CPU.

No forward passes, inference, training, configuration or checkpoint changes.
The unmodified upstream source is hash-checked by the existing loader.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from scripts.comparators.components import LBUNetK, _load_official_lbunet
from scripts.comparators.fetch_lbunet import COMMIT, PATH, SHA256

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    source_hash = hashlib.sha256(PATH.read_bytes()).hexdigest()
    if source_hash != SHA256:
        raise ValueError("upstream LB-UNet source hash mismatch")
    upstream = _load_official_lbunet()
    binary = upstream.LBUNet(num_classes=1, input_channels=3)
    binary_count = sum(p.numel() for p in binary.parameters())
    conv_bn = sum(
        p.numel()
        for module in binary.modules()
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d))
        for p in module.parameters(recurse=False)
    )
    omitted = {
        "learned_share_space_tensors": sum(
            p.numel() for name, p in binary.named_parameters() if "share_space" in name
        ),
        "custom_LayerNorm": sum(
            p.numel()
            for module in binary.modules()
            if type(module).__name__ == "LayerNorm"
            for p in module.parameters(recurse=False)
        ),
        "GroupNorm": sum(
            p.numel()
            for module in binary.modules()
            if isinstance(module, nn.GroupNorm)
            for p in module.parameters(recurse=False)
        ),
    }
    if conv_bn + sum(omitted.values()) != binary_count:
        raise ValueError("incomplete parameter accounting")
    rows = []
    for dataset, channels, classes in [("isic2018", 3, 2), ("acdc", 1, 4), ("synapse", 1, 9)]:
        model = LBUNetK(channels, classes)
        count = sum(p.numel() for p in model.parameters())
        official = upstream.LBUNet(num_classes=classes, input_channels=channels)
        if count != sum(p.numel() for p in official.parameters()):
            raise ValueError("wrapper added parameters")
        for seed in (1, 2, 3):
            folder = ROOT / f"results/revision_runs/{dataset}_c4_lbunet_seed{seed}"
            provenance = json.loads((folder / "provenance.json").read_text())
            if provenance["source_hashes"][str(PATH.relative_to(ROOT))] != source_hash:
                raise ValueError(f"recorded upstream source differs: {folder}")
            checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model"], strict=True)
            rows.append(
                dict(
                    dataset=dataset,
                    seed=seed,
                    in_channels=channels,
                    outputs=classes,
                    parameters=count,
                    checkpoint_keys=len(checkpoint["model"]),
                    shape_check="strict state_dict load passed",
                    wrapper_added_parameters=0,
                )
            )
    adapted_count = rows[0]["parameters"]
    report = dict(
        source_commit=COMMIT,
        source_sha256=source_hash,
        upstream_binary_parameters=binary_count,
        matched_rgb_two_output_parameters=adapted_count,
        output_head_increment=adapted_count - binary_count,
        wrapper_added_parameters=0,
        paper_reported_millions=0.038,
        conv_and_batchnorm_only=conv_bn,
        excluded_by_partial_count=omitted,
        checkpoints=rows,
        interpretation=(
            "The upstream binary graph already has 55,865 registered parameters; "
            "the two-output adaptation adds only nine. Counting only Conv2d/BatchNorm2d "
            "gives 38,489 (0.038M rounded), omitting 17,376 learned parameters. This "
            "demonstrates a possible counting-convention explanation, not proof of "
            "the paper's undocumented accounting. Do not claim exact reproduction."
        ),
        adaptations=(
            "Default upstream widths retained. K-class pre-sigmoid logits; CE+Dice; "
            "union morphological boundary targets instead of genetic-algorithm "
            "key-point maps; fixed 224px/150-epoch unaugmented schedule."
        ),
        scope="Static CPU parameter/checkpoint inspection only; no forward or training pass.",
    )
    target = ROOT / "revision/validation/lbunet_architecture_audit.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(binary_count, conv_bn, omitted, f"{len(rows)} strict checkpoint checks passed")


if __name__ == "__main__":
    main()
