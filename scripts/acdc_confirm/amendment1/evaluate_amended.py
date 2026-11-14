"""Amendment 1: run the unchanged one-time evaluation with the amended integrity gate.

scripts/acdc_confirm/evaluate_test.py is not modified. Its module-level name `check` (imported from
integrity.py) is rebound, for this process only, to integrity_amended.check, which runs every
original check and tolerates only the documented source-manifest defect. Everything else in
evaluate_test.main is unchanged: the refusal if results/acdc_confirm_eval exists, the hash checks
on the official-test NPZ files and the exploratory checkpoints, the jobs, the frozen evaluate().
Before evaluating, the pre-amendment snapshot is re-verified (no preserved artifact changed).

    .venv/bin/python scripts/acdc_confirm/amendment1/evaluate_amended.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.acdc_confirm import evaluate_test  # noqa: E402
from scripts.acdc_confirm import integrity as original_integrity  # noqa: E402
from scripts.acdc_confirm.amendment1 import integrity_amended, snapshot  # noqa: E402


def gate() -> dict:
    return integrity_amended.check(allow_test=False)


def main() -> None:
    if evaluate_test.check is not original_integrity.check:
        raise RuntimeError("evaluate_test.check is not the original integrity check")
    changed = snapshot.verify()
    if changed:
        raise RuntimeError(f"preserved artifacts changed: {changed}")
    evaluate_test.check = gate
    evaluate_test.main()
    print(json.dumps(dict(evaluated_once=True, record=str(evaluate_test.RECORD)), indent=2))


if __name__ == "__main__":
    main()
