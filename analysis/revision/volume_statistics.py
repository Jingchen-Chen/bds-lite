"""Registered patient/case-level main-family volume uncertainty and family-V tests."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import norm, wilcoxon

ROOT = Path(__file__).resolve().parents[2]


def main():
    data = {}
    unit_rows = []
    metrics = ["dsc", "hd95", "assd_pooled", "hd95_penalized", "assd_penalized"]
    for family in ["main_p1b_unet", "main_p1b_full"]:
        for dataset in ["acdc", "synapse"]:
            for seed in [1, 2, 3]:
                folder = ROOT / f"revision/volume_results/{family}/{dataset}_seed{seed}"
                if not (folder / "volume_complete.json").exists():
                    raise ValueError("main volume cohorts incomplete")
                groups = defaultdict(list)
                for row in csv.DictReader((folder / "per_volume_class.csv").open()):
                    groups[row["unit"]].append(row)
                if len(groups) != (20 if dataset == "acdc" else 12):
                    raise ValueError("wrong independent unit count")
                for unit, rows in groups.items():
                    expected = 6 if dataset == "acdc" else 8
                    if len(rows) != expected:
                        raise ValueError("incomplete patient/case class/frame rows")
                    for metric in metrics:
                        values = [row[metric] for row in rows]
                        value = (
                            float(np.mean(np.array(values, float))) if "" not in values else None
                        )
                        data[family, dataset, seed, unit, metric] = value
                        unit_rows.append(
                            dict(
                                family=family,
                                dataset=dataset,
                                seed=seed,
                                unit=unit,
                                metric=metric,
                                value=value,
                                complete_class_frames=value is not None,
                            )
                        )
    output = ROOT / "revision/tables"
    with (output / "volume_per_unit_seed.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(unit_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(unit_rows)
    results = []
    for dataset in ["acdc", "synapse"]:
        units = sorted({key[3] for key in data if key[1] == dataset})
        dataset_rows = []
        for metric in metrics:
            delta = []
            for unit in units:
                values = {
                    family: [data[family, dataset, seed, unit, metric] for seed in [1, 2, 3]]
                    for family in ["main_p1b_unet", "main_p1b_full"]
                }
                if any(None in row for row in values.values()):
                    continue
                delta.append(np.mean(values["main_p1b_full"]) - np.mean(values["main_p1b_unet"]))
            delta = np.array(delta)
            rng = np.random.default_rng(20261102)
            draws = (
                delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(1)
                if len(delta)
                else None
            )
            low, high = np.quantile(draws, [0.025, 0.975]) if draws is not None else (None, None)
            raw_p = (
                float(
                    wilcoxon(
                        delta, zero_method="wilcox", alternative="two-sided", method="auto"
                    ).pvalue
                )
                if len(delta) and np.any(delta)
                else 1.0
                if len(delta)
                else None
            )
            judgement = "not assigned: uncalibrated 3D index distances"
            if metric == "dsc" and low is not None:
                judgement = (
                    "superior"
                    if low > 0.01
                    else "inferior"
                    if high < -0.01
                    else "practically equivalent"
                    if low > -0.01 and high < 0.01
                    else "inconclusive"
                )
            mde = (
                (norm.ppf(1 - 0.05 / (2 * 3)) + norm.ppf(0.8))
                * np.std(delta, ddof=1)
                / np.sqrt(len(delta))
                if len(delta) > 1
                else None
            )
            dataset_rows.append(
                dict(
                    dataset=dataset,
                    family="V" if metric in metrics[:3] else "V-sensitivity",
                    comparison="full-U",
                    metric=metric,
                    n=len(delta),
                    excluded_units=len(units) - len(delta),
                    mean_difference=float(delta.mean()) if len(delta) else None,
                    ci95_low=float(low) if low is not None else None,
                    ci95_high=float(high) if high is not None else None,
                    p_raw=raw_p if metric in metrics[:3] else None,
                    p_holm=None,
                    practical_judgement=judgement,
                    approx_normal_mde80_alpha05_over3=float(mde) if mde is not None else None,
                    unit="patient" if dataset == "acdc" else "case",
                    geometry="processed224 index grid"
                    if dataset == "acdc"
                    else "native H5 index grid",
                    bootstrap_draws=10000,
                    rng_seed=20261102,
                )
            )
        ordered = sorted(
            [
                i
                for i, row in enumerate(dataset_rows)
                if row["family"] == "V" and row["p_raw"] is not None
            ],
            key=lambda i: dataset_rows[i]["p_raw"],
        )
        previous = 0.0
        for rank, index in enumerate(ordered):
            previous = max(previous, min(1.0, (3 - rank) * dataset_rows[index]["p_raw"]))
            dataset_rows[index]["p_holm"] = previous
        results.extend(dataset_rows)
    with (output / "volume_primary_statistics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)
    (output / "volume_statistics_scope.json").write_text(
        json.dumps(
            dict(
                protocol="revision/volume_statistics_protocol.md",
                family="V: three volume endpoints within dataset",
                surface_eligibility="Complete units across methods/seeds; penalties separate",
                mde="Normal paired MDE80, alpha .05/3; not exact test power",
                training_uncertainty="Conditional on trained seeds; no population inference",
                physical_units=False,
            ),
            indent=2,
        )
        + "\n"
    )
    print("Generated", len(results), "volume statistical rows with independent patient/case units")


if __name__ == "__main__":
    main()
