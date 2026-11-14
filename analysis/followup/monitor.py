"""Watch a running follow-up queue; exit (so the operator is notified) when something needs a look.

Exits with a one-line JSON reason when: the supervisor process ends; a run completes (when
--on-complete is given); any run's history shows a non-finite training or validation loss; or
--max-hours elapse (heartbeat). Read-only: it never touches the queue, runs or GPU.

    PYTHONPATH=src:. .venv/bin/python analysis/followup/monitor.py --queue e1e2 --max-hours 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.kd_followup.common import QUEUES, WORKSPACE  # noqa: E402


def supervisor_alive() -> bool:
    pid_file = WORKSPACE / "supervision/supervisor.pid"
    if not pid_file.exists():
        return False
    try:
        os.kill(int(pid_file.read_text().strip()), 0)
        return True
    except (ProcessLookupError, ValueError):
        return False


def state(queue: dict) -> dict:
    done, bad, current = [], [], None
    for entry in queue["runs"]:
        run = ROOT / entry["output"]
        if (run / entry["marker"]).exists():
            done.append(entry["run_id"])
            continue
        progress = run / "progress.json"
        if progress.exists():
            p = json.loads(progress.read_text())
            current = dict(run=entry["run_id"], epoch=p["epoch"], best_dsc=p["best_dsc"])
            last = p["last"]
            numbers = [v for v in list(last["train"].values()) + list(last["val"].values())]
            if any(isinstance(v, float) and not math.isfinite(v) for v in numbers):
                bad.append(entry["run_id"])
    return dict(done=done, nonfinite=bad, current=current)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", required=True, choices=sorted(QUEUES))
    parser.add_argument("--max-hours", type=float, default=3.0)
    parser.add_argument("--on-complete", action="store_true")
    args = parser.parse_args()
    queue = yaml.safe_load(QUEUES[args.queue].read_text())
    start, first = time.monotonic(), state(queue)
    seen = set(first["done"])
    while True:
        s = state(queue)
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,memory.used,utilization.gpu",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
        ).stdout.strip()
        reason = None
        if s["nonfinite"]:
            reason = "nonfinite"
        elif not supervisor_alive():
            reason = "supervisor_ended"
        elif args.on_complete and set(s["done"]) - seen:
            reason = "run_completed"
        elif time.monotonic() - start > args.max_hours * 3600:
            reason = "heartbeat"
        if reason:
            print(
                json.dumps(
                    dict(
                        reason=reason,
                        completed=len(s["done"]),
                        total=len(queue["runs"]),
                        new=sorted(set(s["done"]) - seen),
                        current=s["current"],
                        nonfinite=s["nonfinite"],
                        gpu=gpu,
                    )
                )
            )
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
