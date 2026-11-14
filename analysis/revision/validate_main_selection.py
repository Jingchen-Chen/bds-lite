"""Compare regenerated validation overlap to actual selected-epoch training logs."""

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    entries = json.loads((ROOT / "revision/main_prediction_queue.json").read_text())["runs"]
    catalog = {
        row["path"]: row for row in csv.DictReader((ROOT / "revision/audit/checkpoints.csv").open())
    }
    checks = []
    for entry in entries:
        if entry["split"] != "val":
            continue
        name = Path(entry["checkpoint"]).parent.name
        path = ROOT / f"results/runs/{name}/metrics.jsonl"
        logs = [json.loads(line) for line in path.read_text().splitlines()]
        if [row["epoch"] for row in logs] != list(range(1, 151)):
            raise ValueError("incomplete 150-epoch training history")
        epoch = int(catalog[entry["checkpoint"]]["epoch"])
        selected = next(row for row in logs if row["epoch"] == epoch)
        best = max(logs, key=lambda row: row["val"]["val_dsc"])
        rows = list(csv.DictReader((ROOT / entry["output"] / "per_case.csv").open()))
        observed = float(np.mean([float(row["dsc"]) for row in rows]))
        old = float(selected["val"]["val_dsc"])
        checks.append(
            dict(
                run_id=entry["run_id"],
                selected_epoch=epoch,
                logged_best_epoch=best["epoch"],
                logged_dsc=old,
                replayed_dsc=observed,
                absolute_delta=abs(old - observed),
                within_reproduction_dsc_tolerance=abs(old - observed) <= 0.005,
            )
        )
    with (ROOT / "revision/validation/main_selection_replay.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(checks[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(checks)
    print(
        "Checked",
        len(checks),
        "selected checkpoints; maximum DSC delta",
        max(r["absolute_delta"] for r in checks),
    )
    if any(
        not row["within_reproduction_dsc_tolerance"]
        or row["selected_epoch"] != row["logged_best_epoch"]
        for row in checks
    ):
        raise ValueError("main-family replay/selection discrepancy")


if __name__ == "__main__":
    main()
