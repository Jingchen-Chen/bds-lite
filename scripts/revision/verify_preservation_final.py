"""Final preservation check with exact report-backup and live-state exceptions."""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = {
    f"results/final_manuscript_audit_v2/{name}"
    for name in [
        "final_completion_report.md",
        "final_audit_report.md",
        "final_artifact_manifest.csv",
    ]
}
LIVE = {
    "research/boundary_attribution_20261017/monitoring/events/" + name
    for name in ["LATEST.json", "last_delivery_failure.json"]
}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    failures = []
    updates = []
    live = []
    count = 0
    for row in csv.DictReader((ROOT / "revision/audit/inventory.csv").open()):
        if row["kind"] != "file":
            continue
        path = Path(row["root"]) / row["relative_path"]
        count += 1
        if str(ROOT) == row["root"] and row["relative_path"] in REPORTS:
            backup = ROOT / "revision/audit/prior_final_reports" / path.name
            if sha(backup) != row["sha256"]:
                raise ValueError("original report backup differs")
            updates.append(str(path.relative_to(ROOT)))
        elif row["root"] == "/home/jc/bds-followup" and row["relative_path"] in LIVE:
            data = path.read_bytes()
            current = hashlib.sha256(data).hexdigest()
            if current != row["sha256"]:
                snapshot = (
                    ROOT / "revision/audit/live_monitor_drift" / f"{path.stem}_{current[:12]}.json"
                )
                snapshot.write_bytes(data)
                live.append(
                    dict(path=str(path), sha256=current, snapshot=str(snapshot.relative_to(ROOT)))
                )
        elif sha(path) != row["sha256"]:
            failures.append(dict(path=str(path), reason="unexpected original change"))
    migrated = 0
    for row in csv.DictReader((ROOT / "revision/audit/migration.csv").open()):
        path = ROOT / row["destination"]
        migrated += 1
        if not path.resolve().is_relative_to(ROOT) or sha(path) != row["sha256"]:
            failures.append(dict(path=str(path), reason="destination mismatch or external link"))
    result = dict(
        original_files_checked=count,
        authorized_report_updates_with_verified_original_backups=updates,
        observed_live_operational_changes=live,
        migrated_files_checked=migrated,
        unexpected_failures=failures,
        scope="Scientific content preserved; live-state drift separated",
    )
    (ROOT / "revision/audit/migration_final_verification.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        count, "original paths;", migrated, "migrated paths;", len(failures), "unexpected failures"
    )
    if failures:
        raise ValueError("preservation failed")


if __name__ == "__main__":
    main()
