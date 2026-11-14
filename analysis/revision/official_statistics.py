"""WP6 registered image-unit primary comparison, fixed three trained seeds."""

import csv
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS = {"dsc": 0.01, "iou": 0.01, "boundary_f1": 0.01, "hd95": 2.0, "assd": 0.5}


def main():
    base = ROOT / "revision/official_isic/test"
    paired = {}
    for arm in "UF":
        paired[arm] = []
        for seed in [1, 2, 3]:
            rows = list(csv.DictReader((base / f"{arm}_seed{seed}/per_case.csv").open()))
            if len(rows) != 1000:
                raise ValueError("official primary family incomplete")
            paired[arm].append({r["id"]: r for r in rows})
    ids = sorted(paired["U"][0])
    if any(set(x) != set(ids) for tables in paired.values() for x in tables):
        raise ValueError("paired IDs differ")
    results = []
    sensitivity = {"hd95_penalized": 2.0, "assd_penalized": 0.5}
    for metric, sesoi in (THRESHOLDS | sensitivity).items():
        differences = []
        for sid in ids:
            values = {arm: [table[sid][metric] for table in paired[arm]] for arm in "UF"}
            if any("" in values[arm] for arm in "UF"):
                continue
            differences.append(
                np.mean(np.array(values["F"], float)) - np.mean(np.array(values["U"], float))
            )
        delta = np.asarray(differences)
        rng = np.random.default_rng(20261102)
        means = []
        for _ in range(100):
            means.extend(delta[rng.integers(0, len(delta), size=(100, len(delta)))].mean(1))
        low, high = np.quantile(means, [0.025, 0.975])
        oriented_low, oriented_high = (
            (-high, -low) if metric.startswith(("hd95", "assd")) else (low, high)
        )
        judgement = (
            "superior"
            if oriented_low > sesoi
            else "inferior"
            if oriented_high < -sesoi
            else "practically equivalent"
            if oriented_low > -sesoi and oriented_high < sesoi
            else "inconclusive"
        )
        raw_p = (
            float(
                wilcoxon(delta, zero_method="wilcox", alternative="two-sided", method="auto").pvalue
            )
            if np.any(delta)
            else 1.0
        )
        results.append(
            dict(
                family="P" if metric in THRESHOLDS else "P-sensitivity",
                split="official_test",
                comparison="F-U",
                metric=metric,
                n=len(delta),
                missing_paired_units=1000 - len(delta),
                mean_difference=float(delta.mean()),
                ci95_low=float(low),
                ci95_high=float(high),
                p_raw=raw_p if metric in THRESHOLDS else None,
                p_holm=None,
                sesoi=sesoi,
                practical_judgement=judgement,
                bootstrap_draws=10000,
                rng_seed=20261102,
                unit="image averaged across seeds1/2/3",
                limitation="conditional on trained seeds; patient independence unestablished",
            )
        )
    ordered = sorted(
        [i for i, r in enumerate(results) if r["family"] == "P"],
        key=lambda i: results[i]["p_raw"],
    )
    previous = 0.0
    for rank, index in enumerate(ordered):
        previous = max(previous, min(1.0, (len(ordered) - rank) * results[index]["p_raw"]))
        results[index]["p_holm"] = previous
    path = ROOT / "revision/tables/official_isic_primary_statistics.csv"
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)
    print("Primary family complete:", path)


if __name__ == "__main__":
    main()
