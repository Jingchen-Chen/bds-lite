"""ACDC/Synapse copy of scripts/kd/run_queue_kd.py (revision/KD_MULTICLASS_PROTOCOL.md 6).

Identical guards, telemetry and thermal stop. Differences: launches scripts/kd_mc/train_mc.py and
scripts/kd_mc/cache_mc.py; checks configs with scripts/kd_mc/common.require_protocol; if a
dataset's teacher fails its quality gate, that dataset's students are skipped (logged) and the
queue continues with the next dataset, instead of halting.

Original description of scripts/kd/run_queue_kd.py follows.

Same launch guards, telemetry, and thermal stop as the validated dispatcher: refuse an occupied
GPU; launch only below 80 C with < 1 GiB used and >= 25 GiB free disk; log nvidia-smi every 10 s to
<run>/gpu_telemetry.csv; SIGTERM at >= 85 C, < 20 GiB free, or the stop marker, after which the
trainer saves at the end of the epoch and exits 130. Shares revision/.gpu.lock, so it never runs
beside another queue. Differences: only scripts/kd/train_kd.py and scripts/kd/cache_teacher.py are
launched; the stop marker is revision/STOP_KD_TRAINING; completion requires validation outputs
and the absence of any test output; students require a teacher cache that passed the gate.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

from scripts.kd.common import CUDA_VERSION, TORCH_VERSION, sha
from scripts.kd_mc.common import GPU_LOCK, PROTOCOL, ROOT, STOP, cache_run, require_protocol

ENTRY = {"train": "scripts/kd_mc/train_mc.py", "cache": "scripts/kd_mc/cache_mc.py"}
TEST_OUTPUTS = ["preds_test.npz", "per_case_test.csv"]


def complete(run: Path, entry: dict, config_hash: str) -> bool:
    marker = run / entry["marker"]
    for name in TEST_OUTPUTS:
        if (run / name).exists():
            raise ValueError(f"test output present before protocol step 8: {run / name}")
    if not marker.exists():
        return False
    record = json.loads(marker.read_text())
    if record.get("config_sha256") != config_hash:
        raise ValueError(f"completion marker config mismatch: {run}")
    if entry["kind"] == "train":
        if record.get("epochs") != 150 or record.get("test_evaluated") is not False:
            raise ValueError("completion lacks full training evidence")
        for name, checksum in record["artifacts"].items():
            if Path(name).name != name or sha(run / name) != checksum:
                raise ValueError(f"completed artifact hash mismatch: {name}")
    else:
        if sha(run / "train_teacher_prob_fp16.npy") != record["cache_sha256"]:
            raise ValueError("teacher cache hash mismatch")
    return True


def gate_status(dataset: str) -> str:
    marker = ROOT / "results/revision_runs" / cache_run(dataset) / "cache_complete.json"
    if not marker.exists():
        raise RuntimeError("teacher cache not complete; students cannot start")
    return json.loads(marker.read_text())["gate"]["status"]


def gpu_reading():
    line = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    temperature, memory = map(int, line.split(","))
    return temperature, memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    queue = yaml.safe_load(args.queue.read_text())
    ids = [r["run_id"] for r in queue["runs"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate run_id")
    if not args.execute:
        print(json.dumps(queue, indent=2))
        return
    gate = json.loads((ROOT / "revision/reproduction_gate.json").read_text())
    if gate.get("status") != "COMPLETE — VERIFIED" or not gate.get("evidence"):
        raise ValueError("reproduction has not passed; GPU execution forbidden")
    import torch

    if torch.__version__ != TORCH_VERSION or torch.version.cuda != CUDA_VERSION:
        raise ValueError("locked environment mismatch")
    if sha(PROTOCOL) != queue["protocol_sha256"]:
        raise ValueError("KD multiclass protocol changed after registration")
    with GPU_LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for entry in queue["runs"]:
            if not entry.get("approved", False):
                raise ValueError(f"run not approved: {entry['run_id']}")
            if STOP.exists():
                raise RuntimeError("STOP_KD_TRAINING exists; queue paused")
            for source in entry["sources"]:
                if sha(ROOT / source["path"]) != source["sha256"]:
                    raise ValueError(f"registered source changed: {source['path']}")
            config = (ROOT / entry["config"]).resolve()
            if sha(config) != entry["config_sha256"]:
                raise ValueError("registered config changed")
            resolved = yaml.safe_load(config.read_text())
            require_protocol(resolved)
            if (
                sha(ROOT / resolved["revision"]["protocol"])
                != resolved["revision"]["protocol_sha256"]
            ):
                raise ValueError("training protocol changed")
            for manifest in entry["manifests"]:
                if sha(ROOT / manifest["path"]) != manifest["sha256"]:
                    raise ValueError("split manifest hash mismatch")
            run = (ROOT / entry["output"]).resolve()
            if not run.is_relative_to(ROOT / "results/revision_runs"):
                raise ValueError("output must be under results/revision_runs")
            chash = sha(config)
            if complete(run, entry, chash):
                continue
            if entry["kind"] == "train" and resolved["kd"]["role"] == "student":
                status = gate_status(resolved["dataset"]["name"])
                if status != "PASSED":
                    print(f"skip {entry['run_id']}: teacher quality gate {status}", flush=True)
                    continue
            processes = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
            ).strip()
            if processes:
                raise RuntimeError(f"GPU occupied; no process interrupted: {processes}")
            if shutil.disk_usage(ROOT).free < 25 * 1024**3:
                raise OSError("less than 25 GiB free")
            temperature, memory = gpu_reading()
            if temperature >= 80 or memory >= 1024:
                raise RuntimeError(f"GPU not idle/cool: {temperature} C, {memory} MiB")
            argv = [sys.executable, ENTRY[entry["kind"]], "--config", entry["config"]]
            argv += ["--output", entry["output"]]
            if run.exists() and entry["kind"] == "train":
                if not (run / "last.pt").is_file():
                    raise FileExistsError(f"unknown/incomplete run; refusing overwrite: {run}")
                argv.append("--resume")
            if not run.exists():
                run.mkdir(parents=True)
            with (
                (run / "queue_stdout.log").open("a") as stdout,
                (run / "gpu_telemetry.csv").open("a") as telemetry,
            ):
                env = os.environ.copy()
                env.update(
                    BDS_REVISION_GPU_LOCK_FD=str(lock.fileno()),
                    PYTHONPATH=str(ROOT / "src") + ":" + str(ROOT),
                    PYTHONDONTWRITEBYTECODE="1",
                    HF_HUB_OFFLINE="1",
                )
                child = subprocess.Popen(
                    argv,
                    cwd=ROOT,
                    stdout=stdout,
                    stderr=subprocess.STDOUT,
                    env=env,
                    pass_fds=(lock.fileno(),),
                )
                stopping = False
                while child.poll() is None:
                    subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=timestamp,temperature.gpu,clocks.sm,clocks.mem,memory.used,power.draw",
                            "--format=csv,noheader",
                        ],
                        stdout=telemetry,
                        check=True,
                    )
                    telemetry.flush()
                    temperature, _ = gpu_reading()
                    if (
                        temperature >= 85
                        or shutil.disk_usage(ROOT).free < 20 * 1024**3
                        or STOP.exists()
                    ):
                        if not stopping:
                            child.send_signal(signal.SIGTERM)
                            stopping = True
                    time.sleep(10)
                if child.returncode:
                    raise RuntimeError(f"run failed: {entry['run_id']}: {child.returncode}")
            if not complete(run, entry, chash):
                raise RuntimeError("step exited without a verified complete artifact set")


if __name__ == "__main__":
    main()
