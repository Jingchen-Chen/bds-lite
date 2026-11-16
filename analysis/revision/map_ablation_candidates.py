"""Map metadata signatures to roadmap arms without certifying historical objectives."""

import csv
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def main():
    signatures = {
        (0, 0, 0, False): "M0",
        (1, 0, 0, False): "M1",
        (0, 0.05, 0, False): "M2",
        (0, 0.05, 0.05, True): "M3",
        (1, 0.05, 0, True): "M4",
        (1, 0.05, 0.05, False): "M5",
        (1, 0.05, 0.05, True): "M6",
    }
    rows = []
    for record in csv.DictReader((ROOT / "results/audit/checkpoints.csv").open()):
        path = ROOT / record["path"]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        config = payload.get("config", {})
        model = config.get("model", {})
        loss = config.get("loss", {})
        trainer = config.get("trainer", {})
        state = payload.get("model", payload.get("state_dict", {}))
        surface = float(loss.get("surface_weight", 0))
        boundary = float(loss.get("boundary_weight", 0))
        distill = float(loss.get("distill_weight", 0))
        gate = bool(model.get("use_boundary_gate", model.get("boundary_gate", False)))
        scale = float(model.get("boundary_gate_scale", 0.5))
        active_gate = gate and scale != 0
        candidate = signatures.get((surface, boundary, distill, active_gate), "other")
        exclusions = []
        if loss.get("use_gsl") or loss.get("use_boundary_dou"):
            exclusions.append("different geometry-loss objective")
        if any(float(loss.get(key, 0)) for key in ["distance_weight", "far_fp_weight"]):
            exclusions.append("extra objective")
        if loss.get("foreground_class_weights") or loss.get("class_weights"):
            exclusions.append("class weighting")
        if active_gate and scale != 0.25:
            exclusions.append("nonlocked gate scale")
        dataset = config.get("dataset", {}).get("name")
        batch, lr = (8, 3e-4) if dataset == "isic2018" else (16, 1e-4)
        expected = dict(
            epochs=150,
            batch_size=batch,
            lr=lr,
            weight_decay=1e-5,
            amp=True,
            scheduler="cosine",
            grad_clip=1.0,
        )
        for key, value in expected.items():
            if trainer.get(key) != value:
                exclusions.append("trainer." + key)
        if config.get("project", {}).get("seed") not in [1, 2, 3]:
            exclusions.append("training seed")
        if any("gate" in key for key in state) and not gate:
            exclusions.append("gate keys despite disabled gate flag")
        rows.append(
            dict(
                path=record["path"],
                dataset=dataset,
                seed=record["seed"],
                signature_candidate=candidate,
                surface_weight=surface,
                boundary_weight=boundary,
                distill_weight=distill,
                enabled_gate=gate,
                effective_gate=active_gate,
                gate_scale=scale,
                gate_tensor_keys=sum("gate" in key for key in state),
                stored_model_name=model.get("name"),
                protocol_mismatches=";".join(exclusions),
                status="candidate only: source/objective/split verification required",
            )
        )
    with (ROOT / "results/validation/ablation_candidates.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print("Mapped", len(rows), "checkpoint paths; no arm marked complete by signature alone")


if __name__ == "__main__":
    main()
