"""Write the KD configs and queue files with their hashes (run once, at the freeze).

Configs (new files only, AGENTS.md rule 7) in configs/revision/kd/:
  isic2018_kd_teacher_seed100.yaml      teacher (protocol 5, OPEN-6)
  isic2018_kd_band_seed{1..10}.yaml     D+KD-band students
  isic2018_kd_full_seed{1..10}.yaml     D+KD-full students (Phase 2b only)
Queues:
  revision/queue_kd_phase1.yaml   teacher, cache step, band seeds 1-3 (approved)
  revision/queue_kd_phase2a.yaml  band seeds 4-10 (not approved until the Phase 2 decision)
  revision/queue_kd_phase2b.yaml  full seeds 1-10 (not approved; only if C1 is confirmed)
Student configs equal configs/revision/isic2018_m2_aux_boundary_only_seed4.yaml except
project.seed, revision.{purpose, protocol, protocol_sha256, arm}, and the added kd block.
"""

from __future__ import annotations

import argparse
import copy

import yaml

from scripts.kd.common import (
    CACHE_RUN,
    KD_SPEC,
    PROTOCOL,
    ROOT,
    SMP_VERSION,
    TEACHER_MODEL,
    TEACHER_RUN,
    TEACHER_SEED,
    WEIGHTS_REPO,
    WEIGHTS_REVISION,
    WEIGHTS_SHA256,
    decoder_template,
    kd_sources,
    require_protocol,
    sha,
)

CONFIGS = ROOT / "configs/revision/kd"


def _revision(config: dict, arm: str, protocol_sha: str) -> None:
    config["revision"]["purpose"] = f"KD comparator {arm}; see revision/KD_COMPARATOR_PROTOCOL.md"
    config["revision"]["protocol"] = "revision/KD_COMPARATOR_PROTOCOL.md"
    config["revision"]["protocol_sha256"] = protocol_sha
    config["revision"]["gradient_probe"] = False
    config["revision"]["arm"] = arm


def teacher_config(protocol_sha: str) -> dict:
    config = copy.deepcopy(decoder_template())
    config["project"]["seed"] = TEACHER_SEED
    config["model"] = dict(TEACHER_MODEL)
    config["loss"]["boundary_weight"] = 0.0
    config["trainer"]["lr"] = 1e-4
    config["kd"] = dict(
        role="teacher",
        encoder="resnet34",
        encoder_weights="imagenet",
        smp_version=SMP_VERSION,
        weights_repo=WEIGHTS_REPO,
        weights_revision=WEIGHTS_REVISION,
        weights_sha256=WEIGHTS_SHA256,
    )
    _revision(config, "kd_teacher", protocol_sha)
    return config


def student_config(seed: int, mode: str, protocol_sha: str) -> dict:
    config = copy.deepcopy(decoder_template())
    config["project"]["seed"] = seed
    config["kd"] = dict(role="student", mode=mode, cache_run=CACHE_RUN, **KD_SPEC)
    _revision(config, f"kd_{mode}", protocol_sha)
    return config


def entry(run_id: str, kind: str, config_path, marker: str, approved: bool, sources) -> dict:
    config = yaml.safe_load(config_path.read_text())
    return dict(
        run_id=run_id,
        kind=kind,
        approved=approved,
        config=str(config_path.relative_to(ROOT)),
        config_sha256=sha(config_path),
        output=f"results/revision_runs/{run_id}",
        marker=marker,
        manifests=[
            dict(path=m["path"], sha256=m["sha256"])
            for m in config["revision"]["manifests"].values()
        ],
        sources=sources,
    )


def write(path, value) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    protocol_sha = sha(PROTOCOL)
    configs = {TEACHER_RUN: teacher_config(protocol_sha)}
    for mode in ["band", "full"]:
        for seed in range(1, 11):
            configs[f"isic2018_kd_{mode}_seed{seed}"] = student_config(seed, mode, protocol_sha)
    for config in configs.values():
        require_protocol(config)
    if not args.write:
        print(yaml.safe_dump(configs[TEACHER_RUN], sort_keys=False))
        return
    CONFIGS.mkdir(parents=True, exist_ok=True)
    paths = {}
    for run_id, config in configs.items():
        paths[run_id] = CONFIGS / f"{run_id}.yaml"
        if paths[run_id].exists():
            raise FileExistsError(f"registered config exists: {paths[run_id]}")
        write(paths[run_id], config)
    sources = [dict(path=k, sha256=v) for k, v in kd_sources().items()]
    teacher = paths[TEACHER_RUN]
    queues = {
        "phase1": [
            entry(TEACHER_RUN, "train", teacher, "final_metrics.json", True, sources),
            entry(CACHE_RUN, "cache", teacher, "cache_complete.json", True, sources),
        ]
        + [
            entry(
                f"isic2018_kd_band_seed{s}",
                "train",
                paths[f"isic2018_kd_band_seed{s}"],
                "final_metrics.json",
                True,
                sources,
            )
            for s in (1, 2, 3)
        ],
        "phase2a": [
            entry(
                f"isic2018_kd_band_seed{s}",
                "train",
                paths[f"isic2018_kd_band_seed{s}"],
                "final_metrics.json",
                False,
                sources,
            )
            for s in range(4, 11)
        ],
        "phase2b": [
            entry(
                f"isic2018_kd_full_seed{s}",
                "train",
                paths[f"isic2018_kd_full_seed{s}"],
                "final_metrics.json",
                False,
                sources,
            )
            for s in range(1, 11)
        ],
    }
    for name, runs in queues.items():
        path = ROOT / f"revision/queue_kd_{name}.yaml"
        if path.exists():
            raise FileExistsError(f"queue exists: {path}")
        write(
            path,
            dict(
                protocol="revision/KD_COMPARATOR_PROTOCOL.md",
                protocol_sha256=protocol_sha,
                runs=runs,
            ),
        )
        print(path.relative_to(ROOT), len(runs))


if __name__ == "__main__":
    main()
