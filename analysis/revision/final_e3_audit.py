"""Read-only verification of completed E3 training; write a separate audit receipt.

Does not execute a model or read Experiment C outcomes. Does not write training,
validation or test outputs. The E3 test-evaluation status is a dated observation.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def finite_numbers(value) -> bool:
    if isinstance(value, dict):
        return all(finite_numbers(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_numbers(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def main() -> None:
    path = ROOT / "analysis/followup/integrity.py"
    spec = importlib.util.spec_from_file_location("e3_frozen_integrity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    integrity = module.check("e3")
    queue_path = ROOT / "protocols/followup/queue_followup_e3.yaml"
    queue = yaml.safe_load(queue_path.read_text())
    runs = []
    problems = list(integrity["problems"])
    for entry in queue["runs"]:
        folder = ROOT / entry["output"]
        final = json.loads((folder / entry["marker"]).read_text())
        history = json.loads((folder / "history.json").read_text())
        checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
        config = json.loads((folder / "config.json").read_text())
        values = [row["val"]["val_dsc"] for row in history]
        first_max = values.index(max(values)) + 1
        bad_test = [
            name
            for name in (
                "per_case_test.csv",
                "preds_test.npz",
                "per_case_official_test.csv",
                "preds_official_test.npz",
            )
            if (folder / name).exists()
        ]
        checks = {
            "150_sequential_epochs": [row["epoch"] for row in history] == list(range(1, 151)),
            "finite_training_history": finite_numbers(history),
            "first_max_checkpoint": final["selected_epoch"] == first_max,
            "checkpoint_epoch_matches": checkpoint["epoch"] == final["selected_epoch"],
            "checkpoint_config_matches": checkpoint["config"] == config,
            "best_validation_score": abs(final["best_val_dsc"] - max(values)) < 1e-12,
            "training_record_test_not_evaluated": final["test_evaluated"] is False,
            "no_test_output_in_training_folder": not bad_test,
        }
        for key, passed in checks.items():
            if not passed:
                problems.append(f"{entry['run_id']}: {key}")
        runs.append(
            {
                "run": entry["run_id"],
                "epochs": len(history),
                "selected_epoch": final["selected_epoch"],
                "checks": checks,
                "best_checkpoint_sha256": sha(folder / "best.pt"),
                "history_sha256": sha(folder / "history.json"),
                "final_record_sha256": sha(folder / entry["marker"]),
                "provenance_sha256": sha(folder / "provenance.json"),
            }
        )
    evaluation = ROOT / "results/followup_eval/evaluation_record_e3.json"
    result = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "training_integrity_passed": not problems,
        "registered_run_count": len(queue["runs"]),
        "completed_run_count": len(runs),
        "problems": problems,
        "queue_sha256": sha(queue_path),
        "registration_sha256": sha(ROOT / "protocols/followup/followup_registration.md"),
        "frozen_integrity_source_sha256": sha(path),
        "evaluation_source_sha256": sha(ROOT / "analysis/followup/evaluate_test.py"),
        "registered_e3_evaluation_record_exists_at_check": evaluation.exists(),
        "scientific_status": (
            "training complete; existing evaluation record requires separate verification"
            if evaluation.exists()
            else "training complete; registered test evaluation missing at this observation"
        ),
        "analysis_contract": (
            "E3 primary band-aug minus decoder-aug, 10 runs per arm, Boundary F1, "
            "registered estimator, no multiplicity adjustment; secondary/descriptive contrasts "
            "remain exploratory on already examined official patients 101-150."
        ),
        "safe_evaluation_contract": (
            "Frozen analysis/followup/evaluate_test.py --queue e3 checks integrity, refuses "
            "existing run output folders or an existing e3 record, takes the GPU flock, "
            "uses the frozen official loader and evaluate function, and adds E3 outputs only. "
            "It does not retrain or overwrite E1/E2 outputs. analyze.py main would rewrite "
            "existing aggregate E1/E2 outputs; use isolated new derived audit outputs instead."
        ),
        "runs": runs,
    }
    target = ROOT / "evidence/final_audit_e3.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
