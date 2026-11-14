"""ACDC/Synapse copy of scripts/kd/supervise_kd.py: drives scripts/kd_mc/run_queue_mc.py and logs to
revision/kd_multiclass/supervision/. Rules unchanged. Original description follows.

Thermal supervisor for one KD queue (copy of analysis/revision/supervise_queue_v2.py rules).

Same rules as the existing supervisor: before every attempt, wait until no GPU process runs and the
GPU is <= 70 C with < 1 GiB used (at most 20 min); relaunch the queue only after a verified thermal
epoch stop of the current run or the queue's cool-launch guard; halt on any other failure, on the
stop marker, or after five consecutive thermal restarts without saved-epoch progress. It never kills
or signals training. Differences: it drives a single KD queue file, uses the KD stop marker
revision/STOP_KD_TRAINING (revision/STOP_TRAINING is Experiment C's and blocks the existing
supervisor), and logs to revision/kd_comparator/supervision/. It shares the controller lock
revision/training_supervision/controller.lock, so it never runs beside the existing supervisor.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

from scripts.kd_mc.common import ROOT, RUNS, STOP, WORKSPACE

DIRECTORY = WORKSPACE / "supervision"
CONTROLLER_LOCK = ROOT / "revision/training_supervision/controller.lock"


def log(event, **details):
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    record = dict(utc=datetime.now(timezone.utc).isoformat(), event=event, **details)
    with (DIRECTORY / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


def gpu_state():
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    line = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    temperature, memory = map(int, line.split(","))
    return processes, temperature, memory


def cool():
    deadline = time.monotonic() + 1200
    while True:
        if STOP.exists():
            raise RuntimeError("explicit STOP_KD_TRAINING request; no resume")
        processes, temperature, memory = gpu_state()
        if processes:
            raise RuntimeError("GPU occupied; supervisor never interrupts or retries occupied work")
        if temperature <= 70 and memory < 1024:
            log("cool_ready", temperature_c=temperature, memory_mib=memory)
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("GPU failed to cool within 20 min; manual investigation required")
        time.sleep(10)


def thermal_evidence(text: str):
    """The existing classification, with the KD stop marker as the excluded explicit-stop cause."""
    if "GPU not idle/cool:" in text:
        return "cool_launch_guard"
    match = re.search(r"run failed: ([a-z0-9_]+): 130", text)
    if match is None:
        return None
    folder = RUNS / match[1]
    telemetry = folder / "gpu_telemetry.csv"
    if not telemetry.exists() or time.time() - telemetry.stat().st_mtime > 60:
        return None
    rows = telemetry.read_text().splitlines()[-3:]
    other_cause = STOP.exists() or shutil.disk_usage(ROOT).free < 20 * 1024**3
    if other_cause or not any(int(row.split(",")[1].strip()) >= 80 for row in rows):
        return None
    checkpoint = folder / "last.pt"
    if checkpoint.exists() and time.time() - checkpoint.stat().st_mtime < 120:
        progress = json.loads((folder / "progress.json").read_text())
        return dict(
            kind="thermal_epoch_stop", run=folder.name, saved_progress_epoch=progress["epoch"]
        )
    marker = folder / "interrupted.json"
    if marker.exists() and time.time() - marker.stat().st_mtime < 120:
        return dict(kind="thermal_cache_stop", run=folder.name, saved_progress_epoch=None)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", required=True)
    args = parser.parse_args()
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    with CONTROLLER_LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (DIRECTORY / "supervisor.pid").write_text(str(os.getpid()) + "\n")
        log("kd_mc_supervisor_started", pid=os.getpid(), queue=args.queue)
        attempt, stalled, last_progress = 0, 0, None
        while True:
            if STOP.exists():
                log("halt_stop_marker")
                return 2
            cool()
            attempt += 1
            destination = DIRECTORY / f"queue_attempt_{time.time_ns()}.log"
            log("queue_start", attempt=attempt, queue=args.queue, stdout=str(destination))
            env = os.environ.copy()
            env.update(PYTHONPATH=str(ROOT / "src") + ":" + str(ROOT), PYTHONDONTWRITEBYTECODE="1")
            with destination.open("w") as stream:
                child = subprocess.Popen(
                    [
                        sys.executable,
                        "scripts/kd_mc/run_queue_mc.py",
                        "--queue",
                        args.queue,
                        "--execute",
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
                while child.poll() is None:
                    time.sleep(10)
            text = destination.read_text()
            if child.returncode == 0:
                log("queue_finished", queue=args.queue, attempt=attempt)
                return 0
            if STOP.exists():
                log("halt_stop_marker_after_attempt", returncode=child.returncode)
                return 2
            evidence = thermal_evidence(text)
            if evidence is None:
                log("halt_unrecognized_failure", returncode=child.returncode, log=str(destination))
                return 1
            log("guarded_resume_pending_cooling", evidence=evidence)
            progress = (
                (evidence.get("run"), evidence.get("saved_progress_epoch"))
                if isinstance(evidence, dict)
                else None
            )
            stalled = stalled + 1 if progress is not None and progress == last_progress else 0
            last_progress = progress
            if stalled >= 5:
                log("halt_no_progress", evidence=evidence)
                return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # logged, then re-raised: the supervisor halts
        log("halt_exception", error=repr(error))
        raise
