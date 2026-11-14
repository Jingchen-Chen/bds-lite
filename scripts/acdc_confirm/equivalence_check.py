"""Pre-freeze equivalence check: decoder-only via train_run.py vs the launcher at lambda_kd = 0.

Seed 1, a fixed number of optimizer steps (50), no evaluation of any kind. Each path runs its
unmodified main() in a fresh process; the only intervention is that the module's `epoch_train`
name is wrapped so that the first epoch's loader yields STEPS batches, after which the model,
optimizer and batch-order evidence are dumped and the process exits (code 0). Validation,
checkpoint selection, the STOP_TRAINING check and train_run.py's automatic test scoring all come
after epoch_train returns and are therefore never reached. (train_run.py does build its test
dataset object and hashes the test files at start-up, as it always does; no model sees them.)

A third process repeats the train_run.py path to measure run-to-run determinism.

    .venv/bin/python scripts/acdc_confirm/equivalence_check.py
"""

from __future__ import annotations

import argparse
import fcntl
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from scripts.acdc_confirm.common import CONFIGS, RUNS, WORKSPACE, make_config  # noqa: E402
from scripts.kd.common import sha  # noqa: E402

STEPS = 50
SEED = 1
TRAIN_RUN_CONFIG = ROOT / "configs/revision/acdc_m2_aux_boundary_only_unweighted_seed1.yaml"
LAUNCHER_CONFIG = CONFIGS / "equivalence/acdc_equivcheck_decoder_lambda0_seed1.yaml"
OUTPUTS = {
    "train_run": RUNS / "acdc_equivcheck_trainrun_seed1",
    "launcher": RUNS / "acdc_equivcheck_launcher_seed1",
    "train_run_repeat": RUNS / "acdc_equivcheck_trainrun_repeat_seed1",
}
REPORT = WORKSPACE / "equivalence_check.json"


class Truncated:
    def __init__(self, loader, steps, record):
        self.loader, self.steps, self.record = loader, steps, record

    def __iter__(self):
        for batch in itertools.islice(iter(self.loader), self.steps):
            self.record.append(list(batch["id"]))
            yield batch

    def __len__(self):
        return self.steps


def child(path: str, output: Path) -> None:
    import torch

    if path == "launcher":
        from scripts.acdc_confirm.launch import admit
        from scripts.kd_mc import train_mc as module

        admit(yaml.safe_load(LAUNCHER_CONFIG.read_text()), (SEED,), check_protocol_file=False)
        config = LAUNCHER_CONFIG
    else:
        from scripts.revision import train_run as module

        config = TRAIN_RUN_CONFIG
    original = module.epoch_train

    def wrapped(model, loader, criterion, optimizer, scheduler, epoch, step, probes, enabled):
        initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        ids: list[list[str]] = []
        scalars, step = original(
            model, Truncated(loader, STEPS, ids), criterion, optimizer, scheduler, epoch, step,
            probes, enabled,
        )  # fmt: skip
        torch.save(
            dict(
                initial=initial,
                final={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                ids=ids,
                scalars={k: v for k, v in scalars.items() if k != "amp_overflows"},
                overflows=scalars["amp_overflows"],
                step=step,
                criterion=type(criterion).__name__,
            ),
            output / "equivalence_dump.pt",
        )
        raise SystemExit(0)

    module.epoch_train = wrapped
    sys.argv = [module.__file__, "--config", str(config), "--output", str(output)]
    module.main()
    raise RuntimeError("epoch_train was never reached")


def compare(a: dict, b: dict) -> dict:
    import torch

    def tensors(x: dict, y: dict) -> dict:
        diffs = {}
        for key in x:
            u, v = x[key], y[key]
            if u.dtype.is_floating_point:
                diffs[key] = float((u.double() - v.double()).abs().max())
            else:
                diffs[key] = float((u != v).sum())
        return diffs

    out = {}
    for part in ["initial", "final"]:
        if set(a[part]) != set(b[part]):
            raise ValueError(f"state keys differ in {part}")
        diffs = tensors(a[part], b[part])
        equal = [k for k in a[part] if torch.equal(a[part][k], b[part][k])]
        out[part] = dict(
            tensors=len(diffs),
            bitwise_equal=len(equal),
            max_abs_diff=max(diffs.values()),
            worst=sorted(diffs.items(), key=lambda t: -t[1])[:5],
        )
    sa, sb = a["optimizer"]["state"], b["optimizer"]["state"]
    opt = []
    for index in sa:
        for name in ["exp_avg", "exp_avg_sq", "step"]:
            opt.append(float((sa[index][name].double() - sb[index][name].double()).abs().max()))
    out["optimizer_state_max_abs_diff"] = max(opt)
    out["optimizer_param_groups_equal"] = a["optimizer"]["param_groups"] == b["optimizer"][
        "param_groups"
    ]  # fmt: skip
    out["scheduler_equal"] = a["scheduler"] == b["scheduler"]
    out["batch_order_identical"] = a["ids"] == b["ids"]
    out["steps"] = (a["step"], b["step"])
    out["amp_overflows"] = (len(a["overflows"]), len(b["overflows"]))
    common = sorted(set(a["scalars"]) & set(b["scalars"]))
    out["train_scalars"] = {k: (a["scalars"][k], b["scalars"][k]) for k in common}
    out["train_scalar_keys_only_in"] = (
        sorted(set(a["scalars"]) - set(b["scalars"])),
        sorted(set(b["scalars"]) - set(a["scalars"])),
    )
    out["criterion"] = (a["criterion"], b["criterion"])
    return out


def controller() -> None:
    import torch

    GPU_LOCK = ROOT / "revision/.gpu.lock"
    for output in OUTPUTS.values():
        if output.exists():
            raise FileExistsError(f"refusing to reuse {output}")
    LAUNCHER_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    config = make_config("decoder", SEED, "equivalence-check-not-a-registered-run")
    text = yaml.safe_dump(config, sort_keys=False)
    if LAUNCHER_CONFIG.exists() and LAUNCHER_CONFIG.read_text() != text:
        raise ValueError("existing equivalence config differs")
    LAUNCHER_CONFIG.write_text(text)
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if processes:
        raise RuntimeError(f"GPU occupied: {processes}")
    with GPU_LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for path, output in OUTPUTS.items():
            output.mkdir(parents=True)
            env = os.environ.copy()
            env.update(
                BDS_REVISION_GPU_LOCK_FD=str(lock.fileno()),
                PYTHONPATH=str(ROOT / "src") + ":" + str(ROOT),
                PYTHONDONTWRITEBYTECODE="1",
                HF_HUB_OFFLINE="1",
            )
            name = "launcher" if path == "launcher" else "train_run"
            with (output / "equivalence_stdout.log").open("w") as stdout:
                code = subprocess.call(
                    [sys.executable, __file__, "--child", name, "--output", str(output)],
                    cwd=ROOT,
                    env=env,
                    stdout=stdout,
                    stderr=subprocess.STDOUT,
                    pass_fds=(lock.fileno(),),
                )
            if code != 0:
                raise RuntimeError(f"{path} exited {code}; see {output}/equivalence_stdout.log")
    dumps = {
        p: torch.load(o / "equivalence_dump.pt", map_location="cpu", weights_only=False)
        for p, o in OUTPUTS.items()
    }
    for o in OUTPUTS.values():
        for name in ["preds_val.npz", "preds_test.npz", "per_case_test.csv", "final_metrics.json"]:
            if (o / name).exists():
                raise RuntimeError(f"evaluation output produced: {o / name}")
    report = dict(
        steps=STEPS,
        seed=SEED,
        train_run_config=str(TRAIN_RUN_CONFIG.relative_to(ROOT)),
        train_run_config_sha256=sha(TRAIN_RUN_CONFIG),
        launcher_config=str(LAUNCHER_CONFIG.relative_to(ROOT)),
        launcher_config_sha256=sha(LAUNCHER_CONFIG),
        outputs={p: str(o.relative_to(ROOT)) for p, o in OUTPUTS.items()},
        dump_sha256={p: sha(o / "equivalence_dump.pt") for p, o in OUTPUTS.items()},
        evaluation_outputs_produced=False,
        train_run_vs_launcher=compare(dumps["train_run"], dumps["launcher"]),
        train_run_vs_repeat=compare(dumps["train_run"], dumps["train_run_repeat"]),
        environment=dict(torch=torch.__version__, cuda=torch.version.cuda),
    )
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", choices=["train_run", "launcher"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.child:
        child(args.child, args.output)
    else:
        controller()
