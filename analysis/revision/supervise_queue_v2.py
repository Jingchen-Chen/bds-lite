"""Supervisor v2: sequential multi-queue controller with bounded thermal resume.

Differences from supervise_gradient_queue.py (kept unchanged for provenance of the
WP5 attempts):
- Waits (blocking flock) for any previous controller to exit, so it can be started
  while v1 still runs; never two controllers at once.
- Reads revision/queue_plan.yaml at every attempt: an ordered list of
  {runner, queue}. Default: the validated run_queue.py on revision/queue.yaml.
- A queue is skipped when every registered output has final_metrics.json; the
  runner itself re-verifies hashes/completeness.
- When everything is complete it idles (polling every 5 min, up to 72 h) so that
  newly registered approved runs start without manual relaunch.
- No fixed attempt cap; halts after five consecutive no-progress thermal restarts.
- Same thermal classification as v1 (fixed near-threshold rule) and the same
  STOP_TRAINING / unrecognized-failure halts. It never kills or signals training.
"""

import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.revision.supervise_gradient_queue import (  # noqa: E402
    DIRECTORY,
    ROOT,
    cool,
    log,
    thermal_evidence,
)

PLAN = ROOT / "revision/queue_plan.yaml"
DEFAULT = [dict(runner="scripts/revision/run_queue.py", queue="revision/queue.yaml")]


def plan():
    if PLAN.exists():
        return yaml.safe_load(PLAN.read_text())["queues"]
    return DEFAULT


def pending(item):
    queue = yaml.safe_load((ROOT / item["queue"]).read_text())
    return [
        r["run_id"]
        for r in queue["runs"]
        if r.get("approved") and not (ROOT / r["output"] / "final_metrics.json").exists()
    ]


def main():
    DIRECTORY.mkdir(exist_ok=True)
    with (DIRECTORY / "controller.lock").open("a+") as lock:
        log("v2_waiting_for_controller_lock", pid=os.getpid())
        fcntl.flock(lock, fcntl.LOCK_EX)  # blocks until v1 exits
        (DIRECTORY / "supervisor.pid").write_text(str(os.getpid()) + "\n")
        log("v2_acquired_controller_lock", pid=os.getpid())
        idle_since = None
        attempt = 0
        stalled, last_progress = 0, None
        while True:
            if (ROOT / "revision/STOP_TRAINING").exists():
                raise RuntimeError("explicit STOP_TRAINING request; no resume")
            items = [(item, pending(item)) for item in plan()]
            work = [(item, ids) for item, ids in items if ids]
            if not work:
                if idle_since is None:
                    idle_since = time.monotonic()
                    log("v2_all_registered_runs_finished_idling")
                if time.monotonic() - idle_since > 72 * 3600:
                    log("v2_idle_timeout_exit")
                    return
                time.sleep(300)
                continue
            idle_since = None
            item, ids = work[0]
            attempt += 1
            cool()
            destination = DIRECTORY / f"queue_attempt_{time.time_ns()}.log"
            log(
                "queue_start",
                attempt=attempt,
                controller="v2",
                runner=item["runner"],
                queue=item["queue"],
                next_run=ids[0],
                stdout=str(destination.relative_to(ROOT)),
            )
            env = os.environ.copy()
            env.update(PYTHONPATH=str(ROOT / "src") + ":" + str(ROOT), PYTHONDONTWRITEBYTECODE="1")
            with destination.open("w") as stream:
                child = subprocess.Popen(
                    [sys.executable, item["runner"], "--queue", item["queue"], "--execute"],
                    cwd=ROOT,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
                while child.poll() is None:
                    time.sleep(10)
            text = destination.read_text()
            if child.returncode == 0:
                log("queue_finished", queue=item["queue"], attempt=attempt)
                continue
            if (ROOT / "revision/STOP_TRAINING").exists():
                raise RuntimeError("explicit STOP request; automatic resume forbidden")
            evidence = thermal_evidence(text)
            if evidence is None:
                log("halt_unrecognized_failure", returncode=child.returncode, queue=item["queue"])
                raise RuntimeError("preserved failed queue log; no automatic retry")
            log("guarded_resume_pending_cooling", evidence=evidence)
            # No fixed attempt cap (laptop stops every ~6-15 epochs); instead halt when
            # five consecutive thermal restarts make no saved-epoch progress.
            progress = (evidence.get("run"), evidence.get("saved_progress_epoch"))
            stalled = stalled + 1 if progress == last_progress else 0
            last_progress = progress
            if stalled >= 5:
                log("halt_no_progress", evidence=evidence)
                raise RuntimeError("thermal restarts make no progress; inspect cooling")


if __name__ == "__main__":
    print(json.dumps(dict(start=time.time())), flush=True)
    main()
