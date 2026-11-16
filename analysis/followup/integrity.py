"""Integrity check of a finished follow-up queue.

See protocols/followup/followup_registration.md, section 5.

For every registered entry that is not declared failed:
- train: final_metrics.json with 150 epochs and test_evaluated False; config sha256 equal to the
  queue's; every recorded artifact hash; provenance.json config hash, and source hashes equal to
  the registered hashes of the files the frozen trainer records (kd_sources());
- cache: cache_complete.json, cache bytes, the registered config hash;
- no test output (preds_test.npz, per_case_test.csv) in any run folder;
- every registered source file still has its registered hash.

    PYTHONPATH=src:. .venv/bin/python analysis/followup/integrity.py --queue e1e2
Writes results/followup/integrity_<queue>.json; exits 1 on any problem.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.kd.common import sha, write_json  # noqa: E402
from scripts.kd_followup.common import QUEUES, REGISTRATION, WORKSPACE, failed_runs  # noqa: E402
from scripts.kd_mc.common import kd_sources  # noqa: E402

TEST_OUTPUTS = ["preds_test.npz", "per_case_test.csv"]


def check(queue: str) -> dict:
    record = yaml.safe_load(QUEUES[queue].read_text())
    problems, checked = [], []
    if sha(REGISTRATION) != record["protocol_sha256"]:
        problems.append("registration changed after the queue was written")
    failed = failed_runs()
    recorded_keys = set(kd_sources())
    for entry in record["runs"]:
        name, run = entry["run_id"], ROOT / entry["output"]
        if name in failed:
            checked.append(dict(run=name, status="declared failed"))
            continue
        for output in TEST_OUTPUTS:
            if (run / output).exists():
                problems.append(f"{name}: test output {output} in the run folder")
        for source in entry["sources"]:
            if sha(ROOT / source["path"]) != source["sha256"]:
                problems.append(f"{name}: source changed: {source['path']}")
        if sha(ROOT / entry["config"]) != entry["config_sha256"]:
            problems.append(f"{name}: config changed")
        marker = run / entry["marker"]
        if not marker.exists():
            problems.append(f"{name}: not complete")
            continue
        final = json.loads(marker.read_text())
        if final.get("config_sha256") != entry["config_sha256"]:
            problems.append(f"{name}: completion record config hash differs")
        if entry["kind"] == "train":
            if final.get("epochs") != 150 or final.get("test_evaluated") is not False:
                problems.append(f"{name}: incomplete training record")
            for artifact, digest in final["artifacts"].items():
                if sha(run / artifact) != digest:
                    problems.append(f"{name}: artifact changed: {artifact}")
            provenance = json.loads((run / "provenance.json").read_text())
            registered = {
                s["path"]: s["sha256"] for s in entry["sources"] if s["path"] in recorded_keys
            }
            if provenance["config_sha256"] != entry["config_sha256"]:
                problems.append(f"{name}: provenance config hash differs")
            if provenance["source_hashes"] != registered:
                problems.append(f"{name}: provenance source hashes differ from registration")
        elif sha(run / "train_teacher_prob_fp16.npy") != final["cache_sha256"]:
            problems.append(f"{name}: cache bytes differ")
        checked.append(dict(run=name, kind=entry["kind"], status="checked"))
    return dict(queue=queue, passed=not problems, problems=problems, runs=checked)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", required=True, choices=sorted(QUEUES))
    args = parser.parse_args()
    result = check(args.queue)
    write_json(WORKSPACE / f"integrity_{args.queue}.json", result)
    print(json.dumps(dict(passed=result["passed"], problems=result["problems"][:20]), indent=2))
    if not result["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
