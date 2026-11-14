"""Queue runner: the frozen scripts/kd_mc/run_queue_mc.py, unchanged, with module bindings.

All guards, telemetry and the thermal stop are those of run_queue_mc.main(). Rebound for this
process only (as in scripts/acdc_confirm/run_queue.py):
- ENTRY -> scripts/kd_followup/launch.py (train) and scripts/kd_followup/cache_peer.py (cache);
- PROTOCOL -> revision/followup_registration.md (the queue's registered hash);
- require_protocol -> scripts/kd_followup/common.validate;
- complete -> also True for a run declared failed after one retry (registration section 6).

    .venv/bin/python scripts/kd_followup/run_queue.py --queue revision/queue_followup_e1e2.yaml \
        --execute
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.kd_followup.common import REGISTRATION, failed_runs, validate  # noqa: E402
from scripts.kd_mc import run_queue_mc  # noqa: E402

_complete = run_queue_mc.complete


def complete(run, entry, config_hash):
    if entry["run_id"] in failed_runs():
        print(f"skip {entry['run_id']}: declared failed (registration section 6)", flush=True)
        return True
    return _complete(run, entry, config_hash)


run_queue_mc.ENTRY = {
    "train": "scripts/kd_followup/launch.py",
    "cache": "scripts/kd_followup/cache_peer.py",
}
run_queue_mc.PROTOCOL = REGISTRATION
run_queue_mc.require_protocol = validate
run_queue_mc.complete = complete

if __name__ == "__main__":
    run_queue_mc.main()
