"""Confirmation-queue copy of scripts/expansion/run_queue_expansion.py.

See revision/CONFIRMATION_PROTOCOL.md. Differences: seeds 1-17 are accepted through
scripts/confirmation/train_confirmation.py, and only that entry point may be launched.
Shares revision/.gpu.lock, so it never runs alongside another queue.

WP0 sequential dispatcher. Execution gated on recorded reproduction evidence.

Training adapters must implement resume and emit final_metrics.json only after
saving best.pt, config.json and both prediction archives. No historical result
is relabeled complete by this dispatcher.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_protocol(config):
    t = config["trainer"]
    dataset = config["dataset"]["name"]
    batch, lr = (8, 3e-4) if dataset == "isic2018" else (16, 1e-4)
    expected = dict(
        epochs=150,
        batch_size=batch,
        lr=lr,
        weight_decay=1e-5,
        scheduler="cosine",
        grad_clip=1.0,
        amp=True,
    )
    for key, value in expected.items():
        if t.get(key) != value:
            raise ValueError(f"locked protocol differs: {key}: {t.get(key)} != {value}")
    if config["project"]["seed"] not in range(1, 18):
        raise ValueError("training seed must be 1-17")
    if config["data"]["image_size"] != [224, 224]:
        raise ValueError("input must be 224x224")
    if config.get("augmentation") or config["data"].get("augmentation"):
        raise ValueError("online augmentation is not permitted")
    if dataset not in ["isic2018", "acdc", "synapse"]:
        raise ValueError("unknown dataset")
    if t.get("init_from_checkpoint") or t.get("freeze_encoder_epochs", 0):
        raise ValueError("initialization/freezing not in locked protocol")
    # Split files are required by each queue entry and checked below.


def complete(run, config_hash):
    marker = run / "final_metrics.json"
    if not marker.exists():
        return False
    metadata = json.loads(marker.read_text())
    if metadata.get("config_sha256") != config_hash:
        raise ValueError(f"completion marker config mismatch: {run}")
    for name in ["best.pt", "config.json", "preds_val.npz", "preds_test.npz"]:
        if not (run / name).is_file():
            raise ValueError(f"completion marker lacks {name}: {run}")
    if metadata.get("epochs") != 150 or not metadata.get("artifacts"):
        raise ValueError("completion lacks full training/hash evidence")
    for name, checksum in metadata["artifacts"].items():
        if Path(name).name != name or digest(run / name) != checksum:
            raise ValueError(f"completed artifact hash mismatch: {name}")
    return True


def gpu_reading():
    line = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    temperature, memory = map(int, line.split(","))
    return temperature, memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, default=ROOT / "revision/queue_confirmation.yaml")
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
    if not all((ROOT / p).is_file() for p in gate["evidence"]):
        raise ValueError("reproduction evidence missing")
    import torch

    if torch.__version__ != "2.11.0+cu130" or torch.version.cuda != "13.0":
        raise ValueError("locked environment mismatch")
    with (ROOT / "revision/.gpu.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for entry in queue["runs"]:
            if not entry.get("approved", False):
                raise ValueError(f"run not ready: {entry['run_id']}")
            if (ROOT / "revision/STOP_TRAINING").exists():
                raise RuntimeError("STOP_TRAINING exists; queue paused")
            for source in entry.get("sources", []):
                if digest(ROOT / source["path"]) != source["sha256"]:
                    raise ValueError("registered training source changed")
            config = (ROOT / entry["config"]).resolve()
            if digest(config) != entry["config_sha256"]:
                raise ValueError("registered config changed")
            resolved = yaml.safe_load(config.read_text())
            require_protocol(resolved)
            if (
                digest(ROOT / resolved["revision"]["protocol"])
                != resolved["revision"]["protocol_sha256"]
            ):
                raise ValueError("training protocol changed")
            for manifest in entry["manifests"]:
                if digest(ROOT / manifest["path"]) != manifest["sha256"]:
                    raise ValueError("split manifest hash mismatch")
            run = (ROOT / entry["output"]).resolve()
            if not run.is_relative_to(ROOT):
                raise ValueError("output must remain in primary repository")
            chash = digest(config)
            if complete(run, chash):
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
            if run.exists():
                if not (run / "last.pt").is_file() or "resume_argv" not in entry:
                    raise FileExistsError(f"unknown/incomplete run; refusing overwrite: {run}")
                argv = entry["resume_argv"]
            else:
                run.mkdir(parents=True)
                argv = entry["argv"]
            with (
                (run / "queue_stdout.log").open("a") as stdout,
                (run / "gpu_telemetry.csv").open("a") as telemetry,
            ):
                if argv[:2] != [sys.executable, "scripts/confirmation/train_confirmation.py"]:
                    raise ValueError("confirmation queue only runs the confirmation entry point")
                env = os.environ.copy()
                env.update(
                    BDS_REVISION_GPU_LOCK_FD=str(lock.fileno()),
                    PYTHONPATH=str(ROOT / "src") + ":" + str(ROOT),
                    PYTHONDONTWRITEBYTECODE="1",
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
                        or (ROOT / "revision/STOP_TRAINING").exists()
                    ):
                        if not stopping:
                            child.send_signal(signal.SIGTERM)
                            stopping = True
                    time.sleep(10)
                if child.returncode:
                    raise RuntimeError(f"run failed: {entry['run_id']}: {child.returncode}")
            if not complete(run, chash):
                raise RuntimeError("trainer exited without verified complete artifact set")


if __name__ == "__main__":
    main()
