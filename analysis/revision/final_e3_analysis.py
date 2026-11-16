"""Registered E3 contrasts plus independent arithmetic, isolated from frozen aggregates.

Requires the completed one-time E3 evaluation record. Reads completed E3 and
stage-3 case files only. Writes final-audit derived files under evidence_freeze,
never the original results/followup/results outputs. Does not execute a model.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis/revision"))
import expansion_statistics as frozen  # noqa: E402

METRICS = ("dsc", "iou", "boundary_f1", "hd95", "assd")
PAIRS = (
    ("E3", "kd_band_aug", "decoder_aug"),
    ("E3a", "decoder_aug", "decoder"),
    ("E3b", "kd_band_aug", "kd_band"),
)
OUT = ROOT / "evidence"
TOLERANCE = 1e-10


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def independent_units(path: Path, label: int | None = None) -> dict:
    slices = defaultdict(lambda: defaultdict(list))
    with path.open() as stream:
        for row in csv.DictReader(stream):
            if label is not None and int(row["label"]) != label:
                continue
            if row["empty_prediction"] == "True" and row["empty_target"] == "True":
                continue
            for metric in METRICS:
                try:
                    value = float(row[metric])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    slices[row["id"]][metric].append(value)
    patients = defaultdict(lambda: defaultdict(list))
    for sample, values in slices.items():
        match = re.match(r"^(patient\d+)", sample)
        if match is None:
            raise ValueError(f"Unexpected sample identity: {sample}")
        for metric, vals in values.items():
            patients[metric][match.group(1)].append(float(np.mean(vals)))
    return {
        metric: {patient: float(np.mean(vals)) for patient, vals in patient_values.items()}
        for metric, patient_values in patients.items()
    }


def independent_contrast(a: list[dict], b: list[dict], metric: str) -> dict:
    common = sorted(set.intersection(*(set(row[metric]) for row in a + b)))
    if len(common) != 50:
        raise ValueError(f"Expected 50 test patients, got {len(common)}")
    sign = -1 if metric in ("hd95", "assd") else 1
    x = np.array([[sign * row[metric][patient] for patient in common] for row in a])
    y = np.array([[sign * row[metric][patient] for patient in common] for row in b])
    ra, rb = x.mean(axis=1), y.mean(axis=1)
    va, vb = ra.var(ddof=1) / len(ra), rb.var(ddof=1) / len(rb)
    vu = (x.mean(axis=0) - y.mean(axis=0)).var(ddof=1) / len(common)
    se = math.sqrt(va + vb + vu)
    df = (va + vb) ** 2 / (va**2 / (len(ra) - 1) + vb**2 / (len(rb) - 1))
    delta = float(ra.mean() - rb.mean())
    width = float(stats.t.ppf(0.975, df)) * se
    return {
        "delta": delta,
        "low": delta - width,
        "high": delta + width,
        "p": float(2 * stats.t.sf(abs(delta / se), df)),
        "se_run": math.sqrt(va + vb),
        "se_unit": math.sqrt(vu),
        "df": float(df),
        "n_a": len(a),
        "n_b": len(b),
        "units": len(common),
        "arm_a_mean": float(sign * ra.mean()),
        "arm_b_mean": float(sign * rb.mean()),
        "arm_a_sd": float(ra.std(ddof=1)),
        "arm_b_sd": float(rb.std(ddof=1)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    record_path = ROOT / "results/followup_eval/evaluation_record_e3.json"
    if not record_path.exists():
        raise FileNotFoundError("E3 evaluation is not complete; wait for its record")
    record = json.loads(record_path.read_text())
    expected = {
        f"acdc_followup_{arm}_seed{seed}"
        for arm in ("decoder_aug", "kd_band_aug")
        for seed in range(201, 211)
    }
    actual = {row["run"] for row in record["runs"]}
    if actual != expected or len(record["runs"]) != 20:
        raise ValueError("E3 evaluation has an unexpected run list")
    hashes = {str(record_path.relative_to(ROOT)): sha(record_path)}
    for source in (
        "analysis/revision/final_e3_analysis.py",
        "analysis/revision/expansion_statistics.py",
        "analysis/revision/family_statistics.py",
        "protocols/followup/followup_registration.md",
        "protocols/followup/queue_followup_e3.yaml",
        "results/acdc_confirm_eval/evaluation_record.json",
    ):
        hashes[source] = sha(ROOT / source)
    artifacts_checked = 0
    for row in record["runs"]:
        if row["split"] != "official_test":
            raise ValueError("Unexpected E3 test set")
        folder = ROOT / "results/followup_eval" / row["run"]
        for name, digest in row["artifacts"].items():
            path = folder / name
            if sha(path) != digest:
                raise ValueError(f"Evaluation artifact hash mismatch: {path}")
            hashes[str(path.relative_to(ROOT))] = digest
            artifacts_checked += 1
        checkpoint = ROOT / "results/revision_runs" / row["run"] / "best.pt"
        if sha(checkpoint) != row["best_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {checkpoint}")
        hashes[str(checkpoint.relative_to(ROOT))] = row["best_sha256"]
    paths = {}
    for arm in ("decoder_aug", "kd_band_aug", "decoder", "kd_band"):
        augmented = arm.endswith("_aug")
        base = ROOT / ("results/followup_eval" if augmented else "results/acdc_confirm_eval")
        prefix = "acdc_followup" if augmented else "acdc_confirm"
        seeds = range(201, 211) if augmented else range(201, 218)
        paths[arm] = {
            seed: base / f"{prefix}_{arm}_seed{seed}" / "per_case_official_test.csv"
            for seed in seeds
        }
    registered = {
        arm: [frozen.unit_values(path, "acdc") for path in arm_paths.values()]
        for arm, arm_paths in paths.items()
    }
    independent = {
        arm: [independent_units(path) for path in arm_paths.values()]
        for arm, arm_paths in paths.items()
    }
    for arm_paths in paths.values():
        for path in arm_paths.values():
            hashes[str(path.relative_to(ROOT))] = sha(path)
    contrasts, discrepancies = [], []
    for cid, first, second in PAIRS:
        for metric in METRICS:
            direct = independent_contrast(independent[first], independent[second], metric)
            original = frozen.contrast(registered[first], registered[second], metric)
            discrepancies.extend(abs(direct[k] - original[k]) for k in direct if k != "units")
            contrasts.append(
                {
                    "id": cid,
                    "first": first,
                    "second": second,
                    "metric": metric,
                    "status": "exploratory; previously examined official patients",
                    "registered_primary_endpoint": cid == "E3" and metric == "boundary_f1",
                    "multiplicity": "none; E3 primary is a single registered contrast",
                    **direct,
                    "reading": frozen.classify(direct["low"], direct["high"], frozen.SESOI[metric]),
                }
            )
    per_seed = []
    for arm, records in independent.items():
        for seed, values in zip(paths[arm], records, strict=True):
            for metric in METRICS:
                per_seed.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "metric": metric,
                        "value": float(np.mean(list(values[metric].values()))),
                    }
                )
    per_class = []
    for label, name in ((1, "RV"), (2, "myocardium"), (3, "LV")):
        for metric in ("dsc", "boundary_f1", "hd95"):
            a = [independent_units(p, label) for p in paths["kd_band_aug"].values()]
            b = [independent_units(p, label) for p in paths["decoder_aug"].values()]
            per_class.append(
                {"structure": name, "metric": metric, **independent_contrast(a, b, metric)}
            )
    maximum = max(discrepancies)
    if maximum > TOLERANCE:
        raise ValueError(f"Independent/registered calculation discrepancy {maximum}")
    primary = next(r for r in contrasts if r["registered_primary_endpoint"])
    training_costs = {}
    for arm in ("decoder_aug", "kd_band_aug"):
        seconds = []
        for seed in range(201, 211):
            path = ROOT / f"results/revision_runs/acdc_followup_{arm}_seed{seed}/history.json"
            hashes[str(path.relative_to(ROOT))] = sha(path)
            seconds.extend(row["seconds"] for row in json.loads(path.read_text()))
        training_costs[arm] = {
            "summed_epoch_hours": sum(seconds) / 3600,
            "median_epoch_seconds": float(np.median(seconds)),
            "epochs": len(seconds),
            "scope": "Recorded epoch time; excludes cooling and other wall-clock overhead.",
        }
    outcome = (
        "positive interval exclusion under augmentation (exploratory)"
        if primary["low"] > 0
        else (
            "negative interval exclusion under augmentation (exploratory)"
            if primary["high"] < 0
            else "effect under augmentation not established"
        )
    )
    result = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "status": "verified registered arithmetic; exploratory follow-up",
        "evaluation_started_utc": record["started_utc"],
        "evaluation_finished_utc": record["finished_utc"],
        "evaluation_artifacts_checked": artifacts_checked,
        "checkpoints_checked": 20,
        "contrast_count": len(contrasts),
        "independent_scalar_checks": len(discrepancies),
        "max_absolute_discrepancy": maximum,
        "tolerance": TOLERANCE,
        "primary_outcome": outcome,
        "primary": primary,
        "training_costs": training_costs,
        "contrasts": contrasts,
        "limitations": [
            "Official patients were already examined; E3 is exploratory, not a new confirmation.",
            "The registered Stage-3 primary estimator and its original reading remain unchanged.",
            "E3a/E3b compare 10 augmented runs with 17 existing unaugmented runs; descriptive.",
            "No direct comparison separates teacher, region, loss scale or generalization effects.",
            "Primary confidence intervals are nominal, not simultaneous or equivalence tests.",
            "Residual duplication and run-only degrees of freedom remain model-dependent limits.",
        ],
        "source_sha256": hashes,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "final_e3_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    write_csv(OUT / "final_e3_analysis.csv", contrasts)
    write_csv(OUT / "final_e3_per_seed.csv", per_seed)
    write_csv(OUT / "final_e3_per_class.csv", per_class)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("contrasts", "source_sha256")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
