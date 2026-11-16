"""Gate before any expansion run: seed 1 through the new entry point must replay history.

For each ISIC2018 arm, trains seed 1 for a few epochs and compares validation DSC per epoch with the
archived seed-1 run. Epochs only; training stops before any test evaluation. Writes
results/gates/expansion/gate.json. Exit code 1 on any mismatch.
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

from analysis.revision import expansion_arms as arms

ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT / "results/gates/expansion"
EPOCHS = 4
ARMS = {
    "m0_unet": (arms.isic_unet, "p1b", "unet"),
    "m6_full": (arms.isic_full, "p1b", "bdslite_full_current"),
    "m2_aux_boundary_only": (arms.isic_decoder, "run", "isic2018_m2_aux_boundary_only_seed1"),
    "gsl": (arms.isic_gsl, "run", "isic2018_gsl_seed1"),
}


def history(kind: str, name: str) -> list[float]:
    if kind == "p1b":
        path = ROOT / f"results/runs/isic2018_p1b_{name}_seed1/metrics.jsonl"
        return [json.loads(line)["val"]["val_dsc"] for line in path.open()]
    records = json.loads((ROOT / f"results/revision_runs/{name}/history.json").read_text())
    return [r["val"]["val_dsc"] for r in records]


def main() -> int:
    results = []
    with (ROOT / "revision/.gpu.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for arm, (builder, kind, name) in ARMS.items():
            config_path = GATE / f"{arm}_seed1.yaml"
            config_path.write_text(yaml.safe_dump(builder(1), sort_keys=False))
            output = ROOT / f"results/revision_runs/gate_expansion_{arm}_seed1"
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
                "scripts/expansion/train_expansion.py",
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
