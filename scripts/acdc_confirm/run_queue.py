"""Queue runner: the frozen scripts/kd_mc/run_queue_mc.py, unchanged, with three module bindings.

All guards, telemetry and the thermal stop are those of run_queue_mc.main(). Rebound for this
process only:
- ENTRY["train"] -> scripts/acdc_confirm/launch.py (which runs the frozen train_mc.py);
- PROTOCOL -> preregistration/acdc_confirmation.md (the queue's registered protocol hash);
- require_protocol -> scripts/acdc_confirm/common.validate (config equals the generator output);
- complete -> also True for a run declared failed under the pre-registered policy (section 5;
  scripts/acdc_confirm/declare_failed.py), so the queue skips it and continues. Its folder is not
  touched.

    .venv/bin/python scripts/acdc_confirm/run_queue.py \
        --queue revision/queue_acdc_confirm.yaml --execute
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.acdc_confirm.common import PREREG, failed_runs, validate  # noqa: E402
from scripts.kd_mc import run_queue_mc  # noqa: E402

_complete = run_queue_mc.complete


def complete(run, entry, config_hash):
    if entry["run_id"] in failed_runs():
        print(f"skip {entry['run_id']}: declared failed (pre-registration section 5)", flush=True)
        return True
    return _complete(run, entry, config_hash)


run_queue_mc.ENTRY = {"train": "scripts/acdc_confirm/launch.py"}
run_queue_mc.PROTOCOL = PREREG
run_queue_mc.require_protocol = validate
run_queue_mc.complete = complete

if __name__ == "__main__":
    run_queue_mc.main()
