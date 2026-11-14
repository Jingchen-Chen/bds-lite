#!/usr/bin/env python
"""Aggregate multi-seed validation results and report mean ± std.

Reads eval_val.csv files from a list of checkpoint directories that correspond
to repeated runs of the same experiment with different random seeds, computes
per-metric mean and std across seeds, and writes a summary CSV.

Usage:
    python scripts/prepare_seed_averaged_results.py
"""

from __future__ import annotations

import csv
from pathlib import Path

CHECKPOINTS_DIR = Path("results/checkpoints")
OUTPUT_DIR = Path("results/tables")

SEED_GROUPS: dict[str, list[str]] = {
    "isic2018_bdslite_boundary_gate_damped_surface": [
        "isic2018_bdslite_boundary_gate_damped_surface",
        "isic2018_bdslite_boundary_gate_damped_surface_seed2027",
        "isic2018_bdslite_boundary_gate_damped_surface_seed2028",
    ],
    "acdc_bdslite_boundary_gate_damped_surface_class13_mild": [
        "acdc_bdslite_boundary_gate_damped_surface_class13_mild",
        "acdc_bdslite_boundary_gate_damped_surface_class13_mild_seed2027",
        "acdc_bdslite_boundary_gate_damped_surface_class13_mild_seed2028",
    ],
}

METRICS = ["dsc", "iou", "hd95", "assd", "boundary_f1"]


def _read_mean_row(csv_path: Path) -> dict[str, float] | None:
    if not csv_path.exists():
        return None
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    mean_rows = [r for r in rows if r.get("label", "") == "mean"]
    if mean_rows:
        row = mean_rows[0]
    elif rows:
        row = rows[-1]
    else:
        return None
    return {m: float(row[m]) for m in METRICS if m in row}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_rows: list[dict[str, str]] = []

    for group_name, experiments in SEED_GROUPS.items():
        seed_results: list[dict[str, float]] = []
        missing: list[str] = []
        for exp in experiments:
            csv_path = CHECKPOINTS_DIR / exp / "eval_val.csv"
            data = _read_mean_row(csv_path)
            if data is None:
                missing.append(exp)
            else:
                seed_results.append(data)

        if not seed_results:
            print(f"[SKIP] {group_name}: no eval CSVs found yet")
            continue

        n = len(seed_results)
        for metric in METRICS:
            vals = [r[metric] for r in seed_results]
            mean = sum(vals) / n
            variance = sum((v - mean) ** 2 for v in vals) / n if n > 1 else 0.0
            std = variance**0.5
            out_rows.append(
                {
                    "experiment": group_name,
                    "metric": metric,
                    "n_seeds": str(n),
                    "mean": f"{mean:.6f}",
                    "std": f"{std:.6f}",
                    "seeds": ";".join(experiments[:n]),
                    "missing": ";".join(missing),
                }
            )
        suffix = f"; missing: {missing}" if missing else ""
        print(f"[OK] {group_name}: aggregated {n} seed(s){suffix}")

    out_path = OUTPUT_DIR / "seed_averaged_results.csv"
    if out_rows:
        fields = ["experiment", "metric", "n_seeds", "mean", "std", "seeds", "missing"]
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"\nWrote {out_path}")
    else:
        print("\nNo results to write; run training first.")


if __name__ == "__main__":
    main()
