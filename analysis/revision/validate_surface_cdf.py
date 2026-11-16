"""Independent saved-HD95 joins and KDTree distance checks for CDF outputs."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from analysis.revision.error_decomposition import load_targets

ROOT = Path(__file__).resolve().parents[2]


def main():
    directory = ROOT / "results/surface_cdf"
    scope = json.loads((directory / "scope.json").read_text())
    entries = {(e["dataset"], e["family"], str(e["seed"])): e for e in scope["entries"]}
    original = {}
    for key, entry in entries.items():
        path = (ROOT / entry["path"]).parent / "per_case.csv"
        original[key] = {
            (r["id"], int(r.get("label", "1"))): r for r in csv.DictReader(path.open())
        }
    maximum, rows, valid_count, checks = 0.0, 0, 0, []
    selected = set()
    for row in csv.DictReader((directory / "per_image_class.csv").open()):
        rows += 1
        key = row["dataset"], row["family"], row["seed"]
        old = original[key][row["id"], int(row["label"])]
        valid = row["valid"] == "True"
        if valid != (old["hd95"] not in ["", "nan"]):
            raise AssertionError("surface missingness differs from saved evaluation")
        if not valid:
            continue
        valid_count += 1
        delta = abs(float(row["q95"]) - float(old["hd95"]))
        maximum = max(maximum, delta)
        if delta > 2e-5:
            raise AssertionError("CDF95 differs from saved metric")
        # Deterministic first valid unit for each dataset/family at seed1: 10 real checks.
        if row["seed"] != "1" or key in selected:
            continue
        selected.add(key)
        entry = entries[key]
        with np.load(ROOT / entry["path"]) as z:
            index = list(z["ids"].astype(str)).index(row["id"])
            pred = z["predictions"][index] == int(row["label"])
        target = load_targets(entry["dataset"], entry["split"], [row["id"]])[row["id"]]
        truth = target == int(row["label"])
        a = np.argwhere(pred ^ ndi.binary_erosion(pred))
        b = np.argwhere(truth ^ ndi.binary_erosion(truth))
        distances = np.concatenate([cKDTree(b).query(a)[0], cKDTree(a).query(b)[0]])
        independent = np.quantile(distances, [0.5, 0.95, 0.99])
        recorded = np.array([float(row[m]) for m in ["q50", "q95", "q99"]])
        error = float(np.max(np.abs(independent - recorded)))
        if error > 2e-5:
            raise AssertionError("independent KDTree mismatch")
        checks.append(
            dict(
                dataset=key[0],
                family=key[1],
                id=row["id"],
                label=row["label"],
                max_quantile_delta=error,
            )
        )
    groups = defaultdict(list)
    for row in csv.DictReader((directory / "full_cdf.csv").open()):
        groups[row["dataset"], row["family"], row["seed"]].append(row)
    for values in groups.values():
        x = [int(r["distance_squared"]) for r in values]
        assert x == sorted(set(x))
        for column in ["pooled_cdf", "equal_image_class_cdf"]:
            y = np.array([float(r[column]) for r in values])
            assert np.all(np.diff(y) >= -1e-14) and abs(y[-1] - 1) < 1e-12
        pooled = np.cumsum([int(r["points"]) for r in values], dtype=np.float64)
        pooled /= pooled[-1]
        assert np.allclose(pooled, [float(r["pooled_cdf"]) for r in values], atol=1e-14)
    for row in csv.DictReader((directory / "threshold_cdf.csv").open()):
        values = groups[row["dataset"], row["family"], row["seed"]]
        before = [
            r for r in values if int(r["distance_squared"]) <= float(row["threshold_px"]) ** 2
        ]
        for metric in ["pooled_cdf", "equal_image_class_cdf"]:
            expected = float(before[-1][metric]) if before else 0
            assert abs(expected - float(row[metric])) < 1e-12
    report = dict(
        image_class_rows=rows,
        valid_surfaces=valid_count,
        max_saved_hd95_delta=maximum,
        cdf_cohorts=len(groups),
        independent_kdtree_checks=checks,
        checks="all savedHD95/missingness, count-based pooledCDF, monotonic/endpoints/thresholds",
    )
    (ROOT / "results/validation/surface_cdf.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
