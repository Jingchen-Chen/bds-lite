#!/usr/bin/env python
"""Aggregate Phase 16 matched GSL evaluation CSVs across all seeds."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/gsl"
METRICS = ("dsc", "iou", "hd95", "assd", "boundary_f1")


def read_mean(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mean_rows = [row for row in rows if row["label"] == "mean"]
    if len(mean_rows) != 1:
        raise ValueError(f"{path}: expected one mean row")
    return {metric: float(mean_rows[0][metric]) for metric in METRICS}


def main() -> None:
    rows = []
    seed_rows = []
    for dataset in ("isic2018", "acdc", "synapse"):
        splits = ("val",) if dataset == "isic2018" else ("val", "test")
        for split in splits:
            values = {metric: [] for metric in METRICS}
            for seed in (1, 2, 3):
                experiment = f"phase16_matched_gsl_{dataset}_seed{seed}"
                path = OUT / "evaluations" / f"{experiment}_{split}.csv"
                mean = read_mean(path)
                seed_rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "method": "unet_gsl",
                        "seed": seed,
                        **mean,
                        "source_artifact": str(path.relative_to(ROOT)),
                    }
                )
                for metric in METRICS:
                    values[metric].append(mean[metric])
            for metric in METRICS:
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "method": "unet_gsl",
                        "metric": metric,
                        "n_seeds": 3,
                        "mean": statistics.mean(values[metric]),
                        "std": statistics.stdev(values[metric]),
                        "seed_values": "|".join(f"{value:.17g}" for value in values[metric]),
                    }
                )
    for path, data in (
        (OUT / "matched_gsl_seed_results.csv", seed_rows),
        (OUT / "matched_gsl_summary.csv", rows),
    ):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(data[0]),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(data)
    (OUT / "aggregation_summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "seed_rows": len(seed_rows),
                "summary_rows": len(rows),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Aggregated {len(seed_rows)} seed/split rows into {len(rows)} metric rows")


if __name__ == "__main__":
    main()
