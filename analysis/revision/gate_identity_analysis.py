"""Matched inference-gate effects with shared cell eligibility and unit-level uncertainty."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
METRICS = {
    "dsc": 0.01,
    "iou": 0.01,
    "boundary_f1": 0.01,
    "hd95": 2.0,
    "assd": 0.5,
    "hd95_penalized": 2.0,
    "assd_penalized": 0.5,
}


def read_cases(path):
    return {(row["id"], row.get("label", "1")): row for row in csv.DictReader(path.open())}


def baseline(entry):
    ds, split, seed = entry["dataset"], entry["split"], entry["seed"]
    if ds != "isic2018":
        folder = ROOT / f"results/prediction_recovery/main_p1b_full/{ds}_seed{seed}/{split}"
        return folder / "per_case.csv", folder / "predictions.npz"
    if split == "val":
        folder = ROOT / "recovered/followup/results/followup_20261018_v2" / f"isic2018_F_seed{seed}"
        return folder / "per_case.csv", folder / "predictions"
    split = split.removeprefix("official_")
    folder = ROOT / f"results/official_isic/{split}/F_seed{seed}"
    return folder / "per_case.csv", folder / f"preds_{split}.npz"


def read_predictions(path):
    if path.is_dir():
        return {p.stem: np.load(p) for p in path.glob("*.npy")}
    with np.load(path) as z:
        return dict(zip(z["ids"], z["predictions"], strict=True))


def main():
    entries = json.loads((ROOT / "protocols/gate_identity/gate_identity_queue.json").read_text())[
        "runs"
    ]
    cohorts = defaultdict(list)
    pixels = []
    for entry in entries:
        folder = ROOT / entry["output"]
        if not (folder / "evaluation_complete.json").exists():
            raise ValueError("gate recovery incomplete")
        provenance = json.loads((folder / "provenance.json").read_text())
        if not provenance["gate_identity_operator_parity"]["exact_logits"]:
            raise ValueError("counterfactual operator parity not verified")
        base_cases, base_predictions = baseline(entry)
        before = read_cases(base_cases)
        after = read_cases(folder / "per_case.csv")
        if set(before) != set(after):
            raise ValueError("matched case/class rows differ")
        a = read_predictions(base_predictions)
        b = read_predictions(folder / "predictions.npz")
        if set(a) != set(b):
            raise ValueError("paired prediction IDs differ")
        counts = [int(np.count_nonzero(a[sid] != b[sid])) for sid in a]
        pixels.append(
            dict(
                dataset=entry["dataset"],
                split=entry["split"],
                seed=entry["seed"],
                images=len(a),
                changed_images=sum(value > 0 for value in counts),
                changed_pixels=sum(counts),
                max_changed_pixels_per_image=max(counts),
            )
        )
        cohorts[entry["dataset"], entry["split"]].append((entry["seed"], before, after))
    output = ROOT / "results/tables"
    unit_rows = []
    seed_rows = []
    results = []
    for (dataset, split), runs in cohorts.items():
        runs.sort()
        if [r[0] for r in runs] != [1, 2, 3]:
            raise ValueError("three trained seeds required")
        keys = set(runs[0][1])
        if any(set(before) != keys for _, before, _ in runs):
            raise ValueError("seed cohort differs")
        units = defaultdict(list)
        for key in sorted(keys):
            units[key[0] if dataset == "isic2018" else key[0].split("_")[0]].append(key)
        primary = split == "official_test" if dataset == "isic2018" else split == "test"
        cohort_rows = []
        for metric, sesoi in METRICS.items():
            common = {
                key
                for key in keys
                if all(
                    before[key][metric] not in ["", "nan"] and after[key][metric] not in ["", "nan"]
                    for _, before, after in runs
                )
            }
            differences = []
            seed_delta = defaultdict(list)
            for unit, cells in units.items():
                valid = [key for key in cells if key in common]
                if not valid:
                    continue
                per_seed = []
                for seed, before, after in runs:
                    delta = float(
                        np.mean(
                            [
                                float(after[key][metric]) - float(before[key][metric])
                                for key in valid
                            ]
                        )
                    )
                    per_seed.append(delta)
                    seed_delta[seed].append(delta)
                    unit_rows.append(
                        dict(
                            dataset=dataset,
                            split=split,
                            seed=seed,
                            unit=unit,
                            metric=metric,
                            delta_identity_minus_full=delta,
                            eligible_cells=len(valid),
                            missing_cells=len(cells) - len(valid),
                        )
                    )
                differences.append(float(np.mean(per_seed)))
            delta = np.asarray(differences)
            for seed, values in seed_delta.items():
                seed_rows.append(
                    dict(
                        dataset=dataset,
                        split=split,
                        seed=seed,
                        metric=metric,
                        n=len(values),
                        mean_delta=float(np.mean(values)),
                        max_abs_unit_delta=float(np.max(np.abs(values))),
                    )
                )
            low = high = None
            pvalue = None
            judgement = "descriptive only"
            if primary and len(delta):
                rng = np.random.default_rng(20261102)
                draws = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(1)
                low, high = map(float, np.quantile(draws, [0.025, 0.975]))
                if metric not in ["hd95_penalized", "assd_penalized"]:
                    pvalue = (
                        float(
                            wilcoxon(
                                delta, zero_method="wilcox", alternative="two-sided", method="auto"
                            ).pvalue
                        )
                        if np.any(delta)
                        else 1.0
                    )
                lo, hi = (-high, -low) if metric.startswith(("hd95", "assd")) else (low, high)
                judgement = (
                    "superior"
                    if lo > sesoi
                    else "inferior"
                    if hi < -sesoi
                    else "practically equivalent"
                    if lo > -sesoi and hi < sesoi
                    else "inconclusive"
                )
            cohort_rows.append(
                dict(
                    dataset=dataset,
                    split=split,
                    metric=metric,
                    n=len(delta),
                    total_units=len(units),
                    common_cells=len(common),
                    dropped_cells=len(keys) - len(common),
                    mean_delta_identity_minus_full=float(delta.mean()) if len(delta) else None,
                    ci95_low=low,
                    ci95_high=high,
                    p_raw=pvalue,
                    p_holm=None,
                    sesoi=sesoi,
                    practical_judgement=judgement,
                    analysis_scope="exploratory amendment; conditional trained seeds",
                    intervention="inference only; not training M5",
                )
            )
        if primary:
            ordered = sorted(
                [i for i, row in enumerate(cohort_rows) if row["p_raw"] is not None],
                key=lambda i: cohort_rows[i]["p_raw"],
            )
            previous = 0.0
            for rank, index in enumerate(ordered):
                previous = max(previous, min(1.0, (5 - rank) * cohort_rows[index]["p_raw"]))
                cohort_rows[index]["p_holm"] = previous
        results.extend(cohort_rows)
    for name, rows in [
        ("gate_identity_effects.csv", results),
        ("gate_identity_per_unit_seed.csv", unit_rows),
        ("gate_identity_by_seed.csv", seed_rows),
        ("gate_identity_changed_pixels.csv", pixels),
    ]:
        with (output / name).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    print("Gate analysis complete:", len(entries), "cohorts,", len(results), "endpoint summaries")


if __name__ == "__main__":
    main()
