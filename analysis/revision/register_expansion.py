"""Create the expansion configs and revision/queue_expansion.yaml. See EXPANSION_PROTOCOL.md."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from analysis.revision import expansion_arms as arms

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "revision/EXPANSION_PROTOCOL.md"
PY = "/home/jc/bds-followup/.venv/bin/python"
ENTRY = "scripts/expansion/train_expansion.py"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sources() -> list[dict]:
    paths = sorted(
        p
        for folder in [
            "src/bds_lite",
            "scripts/revision",
            "scripts/comparators",
            "scripts/expansion",
        ]
        for p in (ROOT / folder).rglob("*.py")
    )
    paths += [
        ROOT / "analysis/revision/recover_predictions.py",
        ROOT / "analysis/revision/expansion_arms.py",
        ROOT / "analysis/revision/expansion_statistics.py",
        ROOT / PROTOCOL,
    ]
    return [dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in paths]


def plan() -> list[tuple[str, dict]]:
    runs = [
        (f"acdc_m2_aux_boundary_only_unweighted_seed{s}", arms.acdc_decoder_unweighted(s))
        for s in (1, 2, 3)
    ]
    for dataset in ("isic2018", "acdc", "synapse"):
        for seed in (1, 2, 3):
            config = arms.decoder_surface(dataset, seed)
            runs.append((f"{dataset}_m7_decoder_surface_seed{seed}", config))
    for seed in range(4, 11):
        runs += [
            (f"isic2018_m0_unet_seed{seed}", arms.isic_unet(seed)),
            (f"isic2018_m2_aux_boundary_only_seed{seed}", arms.isic_decoder(seed)),
            (f"isic2018_m6_full_seed{seed}", arms.isic_full(seed)),
            (f"isic2018_gsl_seed{seed}", arms.isic_gsl(seed)),
        ]
    return runs


def main() -> None:
    registered = sources()
    entries = []
    for run_id, config in plan():
        config["revision"]["purpose"] = f"expansion {run_id}; see {PROTOCOL}"
        config["revision"]["protocol"] = PROTOCOL
        config["revision"]["protocol_sha256"] = sha(ROOT / PROTOCOL)
        path = ROOT / f"configs/revision/{run_id}.yaml"
        if path.exists():
            raise FileExistsError(path)
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        output = f"results/revision_runs/{run_id}"
        argv = [PY, ENTRY, "--config", str(path.relative_to(ROOT)), "--output", output]
        entries.append(
            dict(
                run_id=run_id,
                approved=True,
                config=str(path.relative_to(ROOT)),
                config_sha256=sha(path),
                manifests=[
                    dict(path=m["path"], sha256=m["sha256"])
                    for m in config["revision"]["manifests"].values()
                ],
                sources=registered,
                output=output,
                argv=argv,
                resume_argv=[*argv, "--resume"],
            )
        )
    queue = ROOT / "revision/queue_expansion.yaml"
    queue.write_text(
        f"# Expansion runs; see {PROTOCOL}\n" + yaml.safe_dump(dict(runs=entries), sort_keys=False)
    )
    print(f"registered {len(entries)} runs")


if __name__ == "__main__":
    main()
