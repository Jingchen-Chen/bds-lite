"""Declare a replication run failed under the pre-registered policy (section 5). Append-only.

    .venv/bin/python scripts/acdc_confirm/declare_failed.py --run RUN --kind diverged --reason "..."
    .venv/bin/python scripts/acdc_confirm/declare_failed.py --run RUN --kind cannot_complete \\
        --reason "..."

Evidence is checked in the run's queue_stdout.log: `diverged` requires the frozen trainer's
FloatingPointError; `cannot_complete` requires at least three failure tracebacks (three resumes
from the last checkpoint that ended in a non-thermal failure). The run folder is left unchanged;
the queue skips the run; the entry goes to failed_runs.json and PROTOCOL_LOG.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.acdc_confirm.common import (  # noqa: E402
    FAILED,
    RUNS,
    failed_runs,
    log_event,
    registered_runs,
    run_name,
)

KINDS = ("diverged", "cannot_complete")


def evidence(text: str, kind: str) -> bool:
    if kind == "diverged":
        return "FloatingPointError" in text
    return text.count("Traceback (most recent call last)") >= 3


def declare(run: str, kind: str, reason: str) -> dict:
    if run not in {run_name(a, s) for a, s in registered_runs()}:
        raise ValueError(f"not a registered replication run: {run}")
    if kind not in KINDS:
        raise ValueError(kind)
    if not reason.strip():
        raise ValueError("a reason is required")
    if run in failed_runs():
        raise ValueError(f"{run} is already declared failed")
    folder = RUNS / run
    if (folder / "final_metrics.json").exists():
        raise ValueError(f"{run} completed; a completed run cannot be declared failed")
    log = folder / "queue_stdout.log"
    if not log.exists() or not evidence(log.read_text(), kind):
        raise ValueError(f"no {kind} evidence in {log}")
    progress = folder / "progress.json"
    last_epoch = json.loads(progress.read_text())["epoch"] if progress.exists() else 0
    stamp = log_event("failed run", f"{run}: {kind} at last saved epoch {last_epoch}. {reason}")
    record = dict(run=run, kind=kind, reason=reason, last_epoch=last_epoch, utc=stamp)
    data = json.loads(FAILED.read_text()) if FAILED.exists() else dict(runs=[])
    data["runs"].append(record)
    FAILED.write_text(json.dumps(data, indent=2) + "\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--kind", required=True, choices=KINDS)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    print(json.dumps(declare(args.run, args.kind, args.reason), indent=2))
