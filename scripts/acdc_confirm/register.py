"""Write the 34 replication configs and the queue (run only after the pre-registration is final).

    .venv/bin/python scripts/acdc_confirm/register.py            # dry run
    .venv/bin/python scripts/acdc_confirm/register.py --write

Each config is make_config(arm, seed, sha256 of the pre-registration); the queue registers the
sha256 of the pre-registration, every config, the manifests and every source file (the parent
study's kd_sources() plus scripts/acdc_confirm/*.py), and refuses to launch or resume on any change.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.acdc_confirm.common import (  # noqa: E402
    CONFIGS,
    PREREG,
    PREREG_REL,
    QUEUE,
    TEACHER_BEST_SHA256,
    TEACHER_CACHE_SHA256,
    make_config,
    registered_runs,
    run_name,
    validate,
)
from scripts.kd.common import sha  # noqa: E402
from scripts.kd_mc.common import RUNS, cache_run, kd_sources  # noqa: E402
from scripts.kd_mc.register_mc import entry  # noqa: E402


def sources() -> list[dict]:
    hashes = kd_sources()
    for path in sorted((ROOT / "scripts/acdc_confirm").glob("*.py")):
        hashes[str(path.relative_to(ROOT))] = sha(path)
    return [dict(path=k, sha256=v) for k, v in sorted(hashes.items())]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if sha(RUNS / cache_run("acdc") / "train_teacher_prob_fp16.npy") != TEACHER_CACHE_SHA256:
        raise ValueError("teacher cache differs from the pre-registration")
    if sha(RUNS / "acdc_kd_teacher_seed100/best.pt") != TEACHER_BEST_SHA256:
        raise ValueError("teacher checkpoint differs from the pre-registration")
    protocol_sha = sha(PREREG)
    configs = {run_name(a, s): make_config(a, s, protocol_sha) for a, s in registered_runs()}
    for config in configs.values():
        validate(config)
    if not args.write:
        print(len(configs), "configs; protocol sha256", protocol_sha)
        return
    CONFIGS.mkdir(parents=True, exist_ok=True)
    registered = sources()
    runs = []
    for name, config in configs.items():
        path = CONFIGS / f"{name}.yaml"
        if path.exists():
            raise FileExistsError(path)
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        runs.append(entry(name, "train", path, "final_metrics.json", registered))
    if QUEUE.exists():
        raise FileExistsError(QUEUE)
    QUEUE.write_text(
        yaml.safe_dump(
            dict(protocol=PREREG_REL, protocol_sha256=protocol_sha, runs=runs), sort_keys=False
        )
    )
    print(QUEUE.relative_to(ROOT), len(runs))


if __name__ == "__main__":
    main()
