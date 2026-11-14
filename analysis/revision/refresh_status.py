"""Refresh task registry and correctly aggregate letter-suffixed WP subtasks.

The frozen training-source snapshot includes build_status.py. This companion
corrects its reporting-only WP grouping without altering a running code snapshot.
"""

import csv
import re
from pathlib import Path

from scripts.revision.build_status import main as build_tasks

ROOT = Path(__file__).resolve().parents[2]


def main():
    build_tasks()
    path = ROOT / "revision/tasks.csv"
    rows = list(csv.DictReader(path.open()))
    requirements = list(csv.DictReader((ROOT / "revision/reviewer_requirements.csv").open()))
    overview = ["## Work-package checklist", ""]
    for number in range(12):
        group = [r for r in rows if int(re.match(r"WP(\d+)", r["task"])[1]) == number]
        values = [r["status"] for r in group]
        done = bool(values) and all(v == "COMPLETE — VERIFIED" for v in values)
        state = (
            "COMPLETE — VERIFIED"
            if done
            else "BLOCKED"
            if values and all(v == "BLOCKED" for v in values)
            else "NOT STARTED"
            if not values or all(v == "NOT STARTED" for v in values)
            else "PARTIAL"
        )
        overview.append(f"- [{'x' if done else ' '}] WP{number}: {state}")
    overview += ["", "Detailed task evidence and reviewer IDs are also in tasks.csv.", ""]
    status = ROOT / "revision/STATUS.md"
    text = status.read_text()
    start, end = text.index("## Work-package checklist"), text.index("| Task |")
    status.write_text(text[:start] + "\n".join(overview) + "\n" + text[end:])
    for row in rows:
        number = re.match(r"WP(\d+)", row["task"])[1]
        row["reviewer_ids"] = ";".join(
            r["reviewer_id"] for r in requirements if number in re.findall(r"\d+", r["wp"])
        )
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print("Corrected WP letter-suffix grouping; empty groups cannot certify completion")


if __name__ == "__main__":
    main()
