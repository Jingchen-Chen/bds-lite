"""Create queue-B comparator configs and revision/queue_b.yaml (idempotent append).

Definitions: revision/COMPARATOR_PROTOCOL.md (hashed into every config).
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "revision/COMPARATOR_PROTOCOL.md"
PY = "/home/jc/bds-followup/.venv/bin/python"
ADAPTER = "scripts/comparators/train_comparator.py"
DATASETS = ["isic2018", "acdc", "synapse"]
ZERO = dict(
    boundary_weight=0.0,
    distill_weight=0.0,
    surface_weight=0.0,
    distance_weight=0.0,
    foreground_class_weights=None,
)
ARMS = {
    "gsl": (
        dict(name="unet", base_channels=32),
        dict(use_gsl=True, lambda_gsl=1.0, gsl_alpha_schedule="step5"),
        dict(kind="gsl"),
    ),
    "c1_boundary_loss": (
        dict(name="unet", base_channels=32),
        dict(),
        dict(kind="c1_boundary_loss", kervadec_dist=True),
    ),
    "c4_lbunet": (
        dict(name="lbunet_official_913ee68"),
        dict(),
        dict(kind="c4_lbunet", boundary_target=True),
    ),
}


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sources() -> list[dict]:
    paths = sorted(
        p
        for folder in ["src/bds_lite", "scripts/revision", "scripts/comparators"]
        for p in (ROOT / folder).rglob("*.py")
    )
    paths.append(ROOT / "analysis/revision/recover_predictions.py")
    paths.append(ROOT / PROTOCOL)
    return [dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in paths]


def build_config(dataset: str, arm: str, seed: int) -> dict:
    template = yaml.safe_load(
        (ROOT / f"configs/revision/{dataset}_wp5_full_seed1.yaml").read_text()
    )
    config = copy.deepcopy(template)
    config["project"]["seed"] = seed
    model, loss, comparator = ARMS[arm]
    config["model"] = dict(model)
    config["loss"].update(ZERO)
    config["loss"].update(loss)
    config["comparator"] = dict(comparator)
    revision = config["revision"]
    revision["purpose"] = f"queue-B comparator {arm}; see {PROTOCOL}"
    revision["protocol"] = PROTOCOL
    revision["protocol_sha256"] = sha(ROOT / PROTOCOL)
    revision["gradient_probe"] = False
    revision["arm"] = arm
    for key in ["historical_config_source", "historical_checkpoint_sha256"]:
        revision.pop(key, None)
    return config


def main() -> None:
    queue_path = ROOT / "revision/queue_b.yaml"
    queue = yaml.safe_load(queue_path.read_text()) if queue_path.exists() else dict(runs=[])
    existing = {entry["run_id"] for entry in queue["runs"]}
    registered_sources = sources()
    added = []
    for arm in ARMS:
        for dataset in DATASETS:
            for seed in (1, 2, 3):
                run_id = f"{dataset}_{arm}_seed{seed}"
                if run_id in existing:
                    continue
                config_path = ROOT / f"configs/revision/{run_id}.yaml"
                if config_path.exists():
                    raise FileExistsError(config_path)
                config = build_config(dataset, arm, seed)
                config_path.write_text(yaml.safe_dump(config, sort_keys=False))
                output = f"results/revision_runs/{run_id}"
                argv = [
                    PY,
                    ADAPTER,
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
                        manifests=[
                            dict(path=m["path"], sha256=m["sha256"])
                            for m in config["revision"]["manifests"].values()
                        ],
                        sources=registered_sources,
                        output=output,
                        argv=argv,
                        resume_argv=argv + ["--resume"],
                    )
                )
                added.append(run_id)
    temporary = queue_path.with_suffix(".yaml.tmp")
    temporary.write_text(
        "# Queue B comparators; see revision/COMPARATOR_PROTOCOL.md\n"
        + yaml.safe_dump(queue, sort_keys=False)
    )
    temporary.replace(queue_path)
    print(f"added {len(added)} runs")


if __name__ == "__main__":
    main()
