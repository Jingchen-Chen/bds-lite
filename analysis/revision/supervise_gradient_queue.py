"""Resume registered queue after verified thermal/cool-launch guards only."""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "revision/training_supervision"


def log(event, **details):
    record = dict(utc=datetime.now(timezone.utc).isoformat(), event=event, **details)
    with (DIRECTORY / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


def cool():
    deadline = time.monotonic() + 1200
    while True:
        if (ROOT / "revision/STOP_TRAINING").exists():
            raise RuntimeError("explicit STOP_TRAINING request; no resume")
        processes = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
        ).strip()
        temperature, memory = map(
            int,
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            .strip()
            .split(","),
        )
        if processes:
            raise RuntimeError("GPU occupied; supervisor never interrupts or retries occupied work")
        if temperature <= 70 and memory < 1024:
            log("cool_ready", temperature_c=temperature, memory_mib=memory)
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("GPU failed to cool within20min; manual investigation required")
        time.sleep(10)


def thermal_evidence(text):
    # The queue may also stop before the next run because it is still too warm.
    if "GPU not idle/cool:" in text:
        return "cool_launch_guard"
    if ": 130" not in text or "run failed:" not in text:
        return None
    match = re.search(r"run failed: ([a-z0-9_]+): 130", text)
    if match is None:
        return None
    folder = ROOT / "results/revision_runs" / match[1]
    telemetry = folder / "gpu_telemetry.csv"
    if not telemetry.exists() or time.time() - telemetry.stat().st_mtime > 60:
        return None
    rows = telemetry.read_text().splitlines()[-3:]
    # run_queue samples temperature twice per 10 s tick: once into the CSV and once
    # for the >=85 C stop decision. The triggering reading can therefore be absent
    # from the CSV (observed 2026-11-02: rows at 84 C, stop at epoch 92). Accept a
    # near-threshold CSV row only when the other two SIGTERM causes are excluded.
    other_cause = (ROOT / "revision/STOP_TRAINING").exists() or (
        __import__("shutil").disk_usage(ROOT).free < 20 * 1024**3
    )
    # 2026-11-04: a stop at epoch 64 of synapse_c1_boundary_loss_seed1 left CSV rows at 77/82/82 C
    # (spike between samples); with STOP_TRAINING and low disk excluded, SIGTERM can only be
    # thermal, so the near-threshold bound is lowered to the launch guard's 80 C.
    if not other_cause and any(int(row.split(",")[1].strip()) >= 80 for row in rows):
        checkpoint = folder / "last.pt"
        progress = json.loads((folder / "progress.json").read_text())
        if checkpoint.exists() and time.time() - checkpoint.stat().st_mtime < 120:
            return dict(
                kind="thermal_epoch_stop", run=folder.name, saved_progress_epoch=progress["epoch"]
            )
    return None


def main():
    DIRECTORY.mkdir(exist_ok=True)
    pidfile = DIRECTORY / "supervisor.pid"
    # Exclusive controller prevents two automatic supervisors.
    import fcntl

    with (DIRECTORY / "controller.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pidfile.write_text(str(os.getpid()) + "\n")
        for attempt in range(1, 401):
            cool()
            destination = DIRECTORY / f"queue_attempt_{time.time_ns()}.log"
            log("queue_start", attempt=attempt, stdout=str(destination.relative_to(ROOT)))
            env = os.environ.copy()
            env.update(PYTHONPATH=str(ROOT / "src") + ":" + str(ROOT), PYTHONDONTWRITEBYTECODE="1")
            with destination.open("w") as stream:
                child = subprocess.Popen(
                    [sys.executable, "scripts/revision/run_queue.py", "--execute"],
                    cwd=ROOT,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
                while child.poll() is None:
                    time.sleep(10)
            text = destination.read_text()
            if child.returncode == 0:
                log("all_registered_runs_finished", attempt=attempt)
                return
            if (ROOT / "revision/STOP_TRAINING").exists():
                raise RuntimeError("explicit STOP request; automatic resume forbidden")
            evidence = thermal_evidence(text)
            if evidence is None:
                log("halt_unrecognized_failure", returncode=child.returncode)
                raise RuntimeError("preserved failed queue log; no automatic retry")
            log("guarded_resume_pending_cooling", evidence=evidence)
        raise RuntimeError("400 controller attempts exhausted; inspect cooling before continuing")


if __name__ == "__main__":
    main()
