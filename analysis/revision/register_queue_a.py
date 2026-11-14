"""Create queue-A configs (config-only ablations, M0w, Attention U-Net) and append
registered entries to revision/queue.yaml. Idempotent: existing run_ids are kept.

Arm definitions: revision/ABLATION_PROTOCOL.md (hashed into every config).
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "revision/ABLATION_PROTOCOL.md"
PY = "/home/jc/bds-followup/.venv/bin/python"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


ARMS = {
    # arm: (model overrides, loss overrides, datasets)
    "m1_sdf_only": (
        dict(name="unet"),
        dict(boundary_weight=0.0, distill_weight=0.0, surface_weight=1.0),
        ["acdc", "synapse"],
    ),
    "m3_no_sdf": (dict(), dict(surface_weight=0.0), ["acdc", "synapse"]),
    "m4_no_distill": (dict(), dict(distill_weight=0.0), ["isic2018", "acdc", "synapse"]),
    "m5_no_gate": (
        dict(boundary_gate_scale=0.0),
        dict(),
        ["isic2018", "acdc", "synapse"],
    ),
    "m0w_unet_classweights": (
        dict(name="unet"),
        dict(boundary_weight=0.0, distill_weight=0.0, surface_weight=0.0),
        ["acdc"],
    ),
    "c3_attention_unet": (
        dict(name="attention_unet"),
        dict(
            boundary_weight=0.0,
            distill_weight=0.0,
            surface_weight=0.0,
            foreground_class_weights=None,
        ),
        ["isic2018", "acdc", "synapse"],
    ),
    "m2_aux_boundary_only": (
        dict(boundary_gate_scale=0.0),
        dict(distill_weight=0.0, surface_weight=0.0),
        ["isic2018", "acdc", "synapse"],
    ),
}
ORDER = [
    "m1_sdf_only",
    "m3_no_sdf",
    "m4_no_distill",
    "m5_no_gate",
    "m0w_unet_classweights",
    "c3_attention_unet",
    "m2_aux_boundary_only",
]


def build_config(dataset: str, arm: str, seed: int) -> dict:
    template = yaml.safe_load(
        (ROOT / f"configs/revision/{dataset}_wp5_full_seed1.yaml").read_text()
    )
    config = copy.deepcopy(template)
    config["project"]["seed"] = seed
    model_over, loss_over, _ = ARMS[arm]
    if model_over.get("name") in {"unet", "attention_unet"}:
        config["model"] = dict(name=model_over["name"], base_channels=32)
    else:
        config["model"].update(model_over)
    config["loss"].update(loss_over)
    revision = config["revision"]
    revision["purpose"] = f"queue-A {arm}; see {PROTOCOL}"
    revision["protocol"] = PROTOCOL
    revision["protocol_sha256"] = sha(ROOT / PROTOCOL)
    revision["gradient_probe"] = False
    revision["arm"] = arm
    for key in ["historical_config_source", "historical_checkpoint_sha256"]:
        revision.pop(key, None)
    return config


def main() -> None:
    queue_path = ROOT / "revision/queue.yaml"
    queue = yaml.safe_load(queue_path.read_text())
    existing = {entry["run_id"] for entry in queue["runs"]}
    sources = [
        dict(path=s["path"], sha256=sha(ROOT / s["path"])) for s in queue["runs"][0]["sources"]
    ]
    for old, new in zip(queue["runs"][0]["sources"], sources, strict=True):
        if old["sha256"] != new["sha256"]:
            raise RuntimeError(f"registered source changed since WP5 registration: {old['path']}")
    added = []
    for arm in ORDER:
        for dataset in ARMS[arm][2]:
            for seed in (1, 2, 3):
                run_id = f"{dataset}_{arm}_seed{seed}"
                if run_id in existing:
                    continue
                config_path = ROOT / f"configs/revision/{run_id}.yaml"
                if config_path.exists():
                    raise FileExistsError(config_path)
                config = build_config(dataset, arm, seed)
                config_path.write_text(yaml.safe_dump(config, sort_keys=False))
                manifests = [
                    dict(path=m["path"], sha256=m["sha256"])
                    for m in config["revision"]["manifests"].values()
                ]
                output = f"results/revision_runs/{run_id}"
                argv = [
                    PY,
                    "scripts/revision/train_run.py",
                    "--config",
                    str(config_path.relative_to(ROOT)),
                    "--output",
                    output,
                ]
                queue["runs"].append(
                    dict(
                        run_id=run_id,
                        approved=True,
                        config=str(config_path.relative_to(ROOT)),
                        config_sha256=sha(config_path),
                        manifests=manifests,
                        sources=sources,
                        output=output,
                        argv=argv,
                        resume_argv=argv + ["--resume"],
                    )
                )
                added.append(run_id)
    temporary = queue_path.with_suffix(".yaml.tmp")
    header = (
        "# WP5 diagnostics (first three) + queue-A ablations/comparator; see ABLATION_PROTOCOL.md\n"
    )
    temporary.write_text(header + yaml.safe_dump(queue, sort_keys=False))
    temporary.replace(queue_path)
    print(f"added {len(added)} runs")
    for run_id in added:
        print(run_id)


if __name__ == "__main__":
    main()
