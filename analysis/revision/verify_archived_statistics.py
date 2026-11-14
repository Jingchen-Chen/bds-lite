"""Recalculate final ACDC evidence from the two archives without large dependencies.

Requires NumPy/SciPy only; reads archives in memory, performs no extraction,
model execution, dataset access or experiment-result writes. Verifies 40 metric
contrasts (confirmation/E1/E2/E3 and descriptive E3 references) and 20 Holm values.
"""

from __future__ import annotations

import csv
import io
import math
import re
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
METRICS = ("dsc", "iou", "boundary_f1", "hd95", "assd")
PAIRS = {
    "S3": ("kd_band", "decoder"),
    "C1": ("kd_band", "kd_full"),
    "C2": ("kd_full", "decoder"),
    "C3": ("kd_band_peer", "decoder"),
    "C4": ("kd_band", "kd_band_peer"),
    "E3": ("kd_band_aug", "decoder_aug"),
    "E3a": ("decoder_aug", "decoder"),
    "E3b": ("kd_band_aug", "kd_band"),
}


def aggregate(data: bytes) -> dict:
    samples = defaultdict(lambda: defaultdict(list))
    for row in csv.DictReader(io.StringIO(data.decode())):
        if row["empty_target"] == row["empty_prediction"] == "True":
            continue
        for metric in METRICS:
            value = row[metric]
            if value and math.isfinite(float(value)):
                samples[row["id"]][metric].append(float(value))
    patients = defaultdict(lambda: defaultdict(list))
    for sid, values in samples.items():
        patient = re.match(r"^patient\d+", sid).group()
        for metric, vals in values.items():
            patients[metric][patient].append(float(np.mean(vals)))
    return {m: {p: float(np.mean(v)) for p, v in pp.items()} for m, pp in patients.items()}


def contrast(a: list, b: list, metric: str) -> dict:
    units = sorted(set.intersection(*(set(row[metric]) for row in a + b)))
    if len(units) != 50:
        raise ValueError("Expected 50 common patients")
    sign = -1 if metric in ("hd95", "assd") else 1
    x = sign * np.array([[row[metric][u] for u in units] for row in a])
    y = sign * np.array([[row[metric][u] for u in units] for row in b])
    va, vb = x.mean(1).var(ddof=1) / len(x), y.mean(1).var(ddof=1) / len(y)
    se = math.sqrt(va + vb + (x.mean(0) - y.mean(0)).var(ddof=1) / len(units))
    df = (va + vb) ** 2 / (va**2 / (len(x) - 1) + vb**2 / (len(y) - 1))
    delta = float(x.mean() - y.mean())
    margin = float(t.ppf(0.975, df)) * se
    return dict(
        delta=delta, low=delta - margin, high=delta + margin, p=float(2 * t.sf(abs(delta / se), df))
    )


def main() -> None:
    arms = defaultdict(dict)
    tables = {}
    pattern = re.compile(
        r"results/(?:acdc_confirm_eval/acdc_confirm|followup_eval/acdc_followup)_"
        r"(decoder|kd_band|kd_full|kd_band_peer|decoder_aug|kd_band_aug)_seed(2\d\d)/"
        r"per_case_official_test\.csv"
    )
    targets = {
        "revision/followup/results/contrasts.csv",
        "revision/evidence_freeze/final_e3_analysis.csv",
    }
    for name in ("primary-evidence-20261110.tar.gz", "final-supplement-20261110.tar.gz"):
        with tarfile.open(ROOT / "revision/evidence_freeze" / name, "r:gz") as archive:
            for member in archive:
                match = pattern.fullmatch(member.name)
                if match:
                    arm, seed = match.group(1), int(match.group(2))
                    arms[arm][seed] = aggregate(archive.extractfile(member).read())
                elif member.name in targets:
                    tables[member.name] = list(
                        csv.DictReader(io.StringIO(archive.extractfile(member).read().decode()))
                    )
    for arm, runs in arms.items():
        expected = set(range(201, 211 if arm.endswith("_aug") else 218))
        if set(runs) != expected:
            raise ValueError(f"Unexpected seeds: {arm}")
    if len(arms) != 6 or set(tables) != targets:
        raise ValueError("Incomplete archive-only statistical inputs")
    rows = [r for values in tables.values() for r in values]
    errors, recalculated = [], {}
    for row in rows:
        first, second = PAIRS[row["id"]]
        got = contrast(list(arms[first].values()), list(arms[second].values()), row["metric"])
        recalculated[row["id"], row["metric"]] = got
        errors.extend(abs(v - float(row[k])) for k, v in got.items())
    for metric in METRICS:
        group = sorted((recalculated[c, metric]["p"], c) for c in ("C1", "C2", "C3", "C4"))
        running = 0.0
        for rank, (p, cid) in enumerate(group):
            running = max(running, min(1.0, (4 - rank) * p))
            expected = next(r for r in rows if r["id"] == cid and r["metric"] == metric)
            errors.append(abs(running - float(expected["p_holm"])))
    if len(rows) != 40 or max(errors) > 1e-10:
        raise ValueError(f"Archive-only numeric verification failed: {max(errors)}")
    print(f"Archive-only: 40 contrasts and20 Holm values matched; max error {max(errors):.3g}")
    print("No datasets, checkpoints, prediction maps, teacher caches or torch required.")


if __name__ == "__main__":
    main()
