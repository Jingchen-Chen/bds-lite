"""One-time test evaluation of a finished follow-up queue (registration section 5).

Requires integrity.py to pass for the queue. Uses the unchanged scripts/revision/train_run.py::
evaluate and the frozen official-test loader of scripts/acdc_confirm/evaluate_test.py (NPZ hashes
checked against the preprocessing record), on patients 101-150 only, writing only under
results/followup_eval/<run>/. Refuses any run that already has an evaluation folder. For the e1e2
queue the E2 compact teacher is evaluated in the same job (descriptive context only).

    PYTHONPATH=src:. .venv/bin/python analysis/followup/evaluate_test.py --queue e1e2
"""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis/followup"))
sys.path.insert(0, str(ROOT / "src"))
from integrity import check  # noqa: E402

from scripts.acdc_confirm.evaluate_test import load, official_loader  # noqa: E402
from scripts.kd.common import GPU_LOCK, sha, write_json  # noqa: E402
from scripts.kd_followup.common import EVAL, QUEUES, RUNS, failed_runs  # noqa: E402
from scripts.revision.train_run import evaluate  # noqa: E402

SPLIT = "official_test"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", required=True, choices=sorted(QUEUES))
    args = parser.parse_args()
    result = check(args.queue)
    if not result["passed"]:
        raise RuntimeError(f"integrity checks failed: {result['problems']}")
    queue = yaml.safe_load(QUEUES[args.queue].read_text())
    failed = failed_runs()
    jobs = [
        e["run_id"] for e in queue["runs"] if e["kind"] == "train" and e["run_id"] not in failed
    ]
    for name in jobs:
        if (EVAL / name).exists():
            raise FileExistsError(f"{EVAL / name} exists: each run is evaluated exactly once")
    record_path = EVAL / f"evaluation_record_{args.queue}.json"
    if record_path.exists():
        raise FileExistsError(record_path)
    EVAL.mkdir(parents=True, exist_ok=True)
    log = dict(
        started_utc=datetime.now(timezone.utc).isoformat(),
        queue=args.queue,
        split=SPLIT,
        failed_runs_not_evaluated=sorted(failed),
        runs=[],
    )
    with GPU_LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data = official_loader()
        for name in jobs:
            folder = RUNS / name
            config = json.loads((folder / "config.json").read_text())
            out = EVAL / name
            out.mkdir()
            metrics = evaluate(load(folder, config), data, 4, out, SPLIT)
            log["runs"].append(
                dict(
                    run=name,
                    split=SPLIT,
                    best_sha256=sha(folder / "best.pt"),
                    metrics=metrics,
                    artifacts={
                        f: sha(out / f) for f in [f"preds_{SPLIT}.npz", f"per_case_{SPLIT}.csv"]
                    },
                )
            )
            print(name, SPLIT, "evaluated once", flush=True)
    log["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(record_path, log)


if __name__ == "__main__":
    main()
