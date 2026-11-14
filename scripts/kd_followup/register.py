"""Write the follow-up configs and both queues (run only after the registration text is final).

    .venv/bin/python scripts/kd_followup/register.py            # dry run
    .venv/bin/python scripts/kd_followup/register.py --write

Each config is make_config(arm, seed, sha256 of the registration). Each queue registers the sha256
of the registration, every config, the manifests and every source file (the parent study's
kd_sources() plus scripts/acdc_confirm/*.py and scripts/kd_followup/*.py), and refuses to launch
or resume on any change. The E3 queue is written now so that it is fixed in advance; it is
launched only under the conditions of registration section 3 (E3).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.acdc_confirm.common import TEACHER_BEST_SHA256, TEACHER_CACHE_SHA256  # noqa: E402
from scripts.kd.common import sha  # noqa: E402
from scripts.kd_followup.common import (  # noqa: E402
    CONFIGS,
    ORIGINAL_CACHE_RUN,
    PEER_CACHE_RUN,
    QUEUES,
    REGISTRATION,
    REGISTRATION_REL,
    RUNS,
    make_config,
    queue_runs,
    run_name,
    validate,
)
from scripts.kd_mc.common import kd_sources  # noqa: E402
from scripts.kd_mc.register_mc import entry  # noqa: E402


def sources() -> list[dict]:
    hashes = kd_sources()
    for folder in ["scripts/acdc_confirm", "scripts/kd_followup"]:
        for path in sorted((ROOT / folder).glob("*.py")):
            hashes[str(path.relative_to(ROOT))] = sha(path)
    return [dict(path=k, sha256=v) for k, v in sorted(hashes.items())]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if sha(RUNS / ORIGINAL_CACHE_RUN / "train_teacher_prob_fp16.npy") != TEACHER_CACHE_SHA256:
        raise ValueError("external teacher cache differs from Stage 3")
    if sha(RUNS / "acdc_kd_teacher_seed100/best.pt") != TEACHER_BEST_SHA256:
        raise ValueError("external teacher checkpoint differs from Stage 3")
    registration_sha = sha(REGISTRATION)
    plan = {q: queue_runs(q) for q in QUEUES}
    configs = {}
    for runs in plan.values():
        for arm, seed, _ in runs:
            configs[run_name(arm, seed)] = make_config(arm, seed, registration_sha)
    for config in configs.values():
        validate(config)
    if not args.write:
        print(len(configs), "configs; registration sha256", registration_sha)
        return
    CONFIGS.mkdir(parents=True, exist_ok=True)
    for name, config in configs.items():
        path = CONFIGS / f"{name}.yaml"
        if path.exists():
            raise FileExistsError(path)
        path.write_text(yaml.safe_dump(config, sort_keys=False))
    registered = sources()
    for queue, runs in plan.items():
        entries = []
        for arm, seed, kind in runs:
            path = CONFIGS / f"{run_name(arm, seed)}.yaml"
            if kind == "cache":
                entries.append(
                    entry(PEER_CACHE_RUN, "cache", path, "cache_complete.json", registered)
                )
            else:
                entries.append(
                    entry(run_name(arm, seed), "train", path, "final_metrics.json", registered)
                )
        if QUEUES[queue].exists():
            raise FileExistsError(QUEUES[queue])
        QUEUES[queue].write_text(
            yaml.safe_dump(
                dict(protocol=REGISTRATION_REL, protocol_sha256=registration_sha, runs=entries),
                sort_keys=False,
            )
        )
        print(QUEUES[queue].relative_to(ROOT), len(entries))


if __name__ == "__main__":
    main()
