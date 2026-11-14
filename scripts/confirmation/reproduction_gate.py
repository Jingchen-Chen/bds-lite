"""Gate before Experiment C: seed 1 of each rebuilt arm must replay the archived trajectory.

Trains seed 1 for a few epochs and compares validation DSC per epoch with the archived seed-1 run.
Epochs only; training stops before any test evaluation. Writes
revision/confirmation_gate/gate.json. Exit code 1 on any mismatch.
"""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

from analysis.revision.confirmation_arms import build

ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT / "revision/confirmation_gate"
EPOCHS = 4
ARMS = {
    "acdc_unet": ("acdc", "unet", "p1b", "acdc_p1b_unet_seed1"),
    "acdc_decoder": ("acdc", "decoder", "run", "acdc_m2_aux_boundary_only_unweighted_seed1"),
    "acdc_full": ("acdc", "full", "run", "acdc_m6_full_unweighted_seed1"),
    "acdc_gsl": ("acdc", "gsl", "run", "acdc_gsl_seed1"),
    "synapse_unet": ("synapse", "unet", "p1b", "synapse_p1b_unet_seed1"),
    "synapse_decoder": ("synapse", "decoder", "run", "synapse_m2_aux_boundary_only_seed1"),
    "synapse_full": ("synapse", "full", "p1b", "synapse_p1b_bdslite_full_current_seed1"),
    "synapse_gsl": ("synapse", "gsl", "run", "synapse_gsl_seed1"),
}


def history(kind: str, name: str) -> list[float]:
    if kind == "p1b":
        path = ROOT / f"results/runs/{name}/metrics.jsonl"
        return [json.loads(line)["val"]["val_dsc"] for line in path.open()]
    records = json.loads((ROOT / f"results/revision_runs/{name}/history.json").read_text())
    return [r["val"]["val_dsc"] for r in records]


def main() -> int:
    results = []
    with (ROOT / "revision/.gpu.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for arm, (dataset, role, kind, name) in ARMS.items():
            GATE.mkdir(exist_ok=True)
            config_path = GATE / f"{arm}_seed1.yaml"
            config_path.write_text(yaml.safe_dump(build(dataset, role, 1), sort_keys=False))
            output = ROOT / f"results/revision_runs/gate_confirmation_{arm}_seed1"
            if output.exists():
                raise FileExistsError(output)
            output.mkdir(parents=True)
            env = os.environ | dict(
                BDS_REVISION_GPU_LOCK_FD=str(lock.fileno()),
                PYTHONPATH=f"{ROOT / 'src'}:{ROOT}",
                PYTHONDONTWRITEBYTECODE="1",
            )
            argv = [
                sys.executable,
                "scripts/confirmation/train_confirmation.py",
                "--config",
                str(config_path.relative_to(ROOT)),
                "--output",
                str(output.relative_to(ROOT)),
            ]
            with (output / "stdout.log").open("w") as log:
                child = subprocess.Popen(
                    argv,
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    pass_fds=(lock.fileno(),),
                )
                sent = False
                while child.poll() is None:
                    progress = output / "progress.json"
                    if not sent and progress.exists():
                        if json.loads(progress.read_text())["epoch"] >= EPOCHS - 1:
                            child.send_signal(signal.SIGTERM)
                            sent = True
                    time.sleep(1)
            lines = [
                json.loads(line)
                for line in (output / "stdout.log").read_text().splitlines()
                if line.startswith("{")
            ]
            new = [record["val"]["val_dsc"] for record in lines]
            old = history(kind, name)[: len(new)]
            difference = max(abs(a - b) for a, b in zip(new, old, strict=True))
            results.append(
                dict(
                    arm=arm,
                    epochs_compared=len(new),
                    new=new,
                    archived=old,
                    max_abs_difference=difference,
                    exact=difference == 0.0,
                    exit_code=child.returncode,
                )
            )
            print(results[-1], flush=True)
    passed = all(r["exact"] and r["epochs_compared"] >= 3 for r in results)
    (GATE / "gate.json").write_text(
        json.dumps(dict(passed=passed, results=results), indent=1) + "\n"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
