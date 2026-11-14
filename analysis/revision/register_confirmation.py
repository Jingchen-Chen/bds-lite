"""Create the Experiment C configs and queue. See revision/CONFIRMATION_PROTOCOL.md."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from analysis.revision.confirmation_arms import build

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "revision/CONFIRMATION_PROTOCOL.md"
PY = "/home/jc/bds-followup/.venv/bin/python"
ENTRY = "scripts/confirmation/train_confirmation.py"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sources() -> list[dict]:
    folders = ["src/bds_lite", "scripts/revision", "scripts/comparators", "scripts/confirmation"]
    paths = sorted(p for folder in folders for p in (ROOT / folder).rglob("*.py"))
    paths += [
        ROOT / "analysis/revision/recover_predictions.py",
        ROOT / "analysis/revision/expansion_arms.py",
        ROOT / "analysis/revision/expansion_statistics.py",
        ROOT / "analysis/revision/confirmation_arms.py",
        ROOT / "analysis/revision/confirmation_statistics.py",
        ROOT / PROTOCOL,
        ROOT / "revision/NARRATIVE_CRITERIA_ADDENDUM.md",
    ]
    return [dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in paths]


def plan() -> list[tuple[str, str, str, int]]:
    from analysis.revision.confirmation_arms import ARM_NAMES

    runs = []
    for index in range(7):
        for dataset, seed in (
            ("acdc", 4 + index),
            ("synapse", 4 + index),
            ("isic2018", 11 + index),
        ):
            for arm, name in ARM_NAMES[dataset].items():
                runs.append((f"{dataset}_{name}_seed{seed}", dataset, arm, seed))
    return runs


def main() -> None:
    registered = sources()
    entries = []
    for run_id, dataset, arm, seed in plan():
        config = build(dataset, arm, seed)
        config["revision"]["purpose"] = f"confirmation {run_id}; see {PROTOCOL}"
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
    queue = ROOT / "revision/queue_confirmation.yaml"
    queue.write_text(
        f"# Experiment C; see {PROTOCOL}\n" + yaml.safe_dump(dict(runs=entries), sort_keys=False)
    )
    print(f"registered {len(entries)} runs")


if __name__ == "__main__":
    main()
