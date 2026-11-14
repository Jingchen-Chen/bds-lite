"""Write the ACDC/Synapse KD configs and queue with hashes (run once, at the freeze).

configs/revision/kd_mc/{acdc,synapse}_kd_teacher_seed100.yaml and
{dataset}_kd_band_seed{1,2,3}.yaml;
revision/queue_kd_mc.yaml in priority order: ACDC teacher, cache, students 1-3; then Synapse.
Student configs equal the dataset's registered decoder-only configs (seed 1 template; the seed 1-3
configs differ only in project.seed and revision.purpose) except project.seed,
revision.{purpose, protocol, protocol_sha256, arm} and the added kd block.
"""

from __future__ import annotations

import argparse
import copy

import yaml

from scripts.kd.common import sha
from scripts.kd_mc.common import (
    DATASETS,
    KD_SPEC,
    PROTOCOL,
    ROOT,
    STUDENT_SEEDS,
    TEACHER_SEED,
    cache_run,
    kd_sources,
    require_protocol,
    student_run,
    teacher_kd_block,
    teacher_model,
    teacher_run,
    template,
)

CONFIGS = ROOT / "configs/revision/kd_mc"
QUEUE = ROOT / "revision/queue_kd_mc.yaml"


def _revision(config: dict, arm: str, protocol_sha: str) -> None:
    config["revision"]["purpose"] = f"KD multiclass {arm}; see revision/KD_MULTICLASS_PROTOCOL.md"
    config["revision"]["protocol"] = "revision/KD_MULTICLASS_PROTOCOL.md"
    config["revision"]["protocol_sha256"] = protocol_sha
    config["revision"]["gradient_probe"] = False
    config["revision"]["arm"] = arm


def teacher_config(dataset: str, protocol_sha: str) -> dict:
    config = copy.deepcopy(template(dataset))
    config["project"]["seed"] = TEACHER_SEED
    config["model"] = teacher_model(dataset)
    config["loss"]["boundary_weight"] = 0.0
    config["kd"] = teacher_kd_block()
    _revision(config, "kd_teacher", protocol_sha)
    return config


def student_config(dataset: str, seed: int, protocol_sha: str) -> dict:
    config = copy.deepcopy(template(dataset))
    config["project"]["seed"] = seed
    config["kd"] = dict(role="student", mode="band", cache_run=cache_run(dataset), **KD_SPEC)
    _revision(config, "kd_band", protocol_sha)
    return config


def entry(run_id, kind, path, marker, sources) -> dict:
    config = yaml.safe_load(path.read_text())
    return dict(
        run_id=run_id,
        kind=kind,
        approved=True,
        config=str(path.relative_to(ROOT)),
        config_sha256=sha(path),
        output=f"results/revision_runs/{run_id}",
        marker=marker,
        manifests=[
            dict(path=m["path"], sha256=m["sha256"])
            for m in config["revision"]["manifests"].values()
        ],
        sources=sources,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    protocol_sha = sha(PROTOCOL)
    configs = {}
    for dataset in DATASETS:
        configs[teacher_run(dataset)] = teacher_config(dataset, protocol_sha)
        for seed in STUDENT_SEEDS:
            configs[student_run(dataset, seed)] = student_config(dataset, seed, protocol_sha)
    for config in configs.values():
        require_protocol(config)
    if not args.write:
        print(yaml.safe_dump(configs[teacher_run("acdc")], sort_keys=False))
        return
    CONFIGS.mkdir(parents=True, exist_ok=True)
    paths = {}
    for run_id, config in configs.items():
        paths[run_id] = CONFIGS / f"{run_id}.yaml"
        if paths[run_id].exists():
            raise FileExistsError(paths[run_id])
        paths[run_id].write_text(yaml.safe_dump(config, sort_keys=False))
    sources = [dict(path=k, sha256=v) for k, v in kd_sources().items()]
    runs = []
    for dataset in DATASETS:
        teacher = paths[teacher_run(dataset)]
        runs.append(entry(teacher_run(dataset), "train", teacher, "final_metrics.json", sources))
        runs.append(entry(cache_run(dataset), "cache", teacher, "cache_complete.json", sources))
        for seed in STUDENT_SEEDS:
            name = student_run(dataset, seed)
            runs.append(entry(name, "train", paths[name], "final_metrics.json", sources))
    if QUEUE.exists():
        raise FileExistsError(QUEUE)
    QUEUE.write_text(
        yaml.safe_dump(
            dict(
                protocol="revision/KD_MULTICLASS_PROTOCOL.md",
                protocol_sha256=protocol_sha,
                runs=runs,
            ),
            sort_keys=False,
        )
    )
    print(QUEUE.relative_to(ROOT), len(runs))


if __name__ == "__main__":
    main()
