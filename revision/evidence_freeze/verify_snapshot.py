"""Independently verify the preserved archive and ACDC recipe-level estimates.

Read-only: consumes the frozen archive, checksum list, and committed contrast CSV.
No test set, training run, or original output is modified.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import re
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "revision/evidence_freeze/primary-evidence-20261110.tar.gz"
MANIFEST = ROOT / "revision/evidence_freeze/primary-evidence-20261110.sha256"
CONTRASTS = ROOT / "revision/followup/results/contrasts.csv"
METRICS = ("dsc", "iou", "boundary_f1", "hd95", "assd")
ARMS = ("decoder", "kd_band", "kd_full", "kd_band_peer")
COMPARISONS = (
    ("S3", "kd_band", "decoder"),
    ("C1", "kd_band", "kd_full"),
    ("C2", "kd_full", "decoder"),
    ("C3", "kd_band_peer", "decoder"),
    ("C4", "kd_band", "kd_band_peer"),
)


def sample_units(data: bytes) -> dict[str, dict[str, float]]:
    """Exclude both-empty labels, average labels per slice then slices per patient."""
    samples = defaultdict(lambda: defaultdict(list))
    for row in csv.DictReader(io.StringIO(data.decode())):
        if row["empty_prediction"] == "True" and row["empty_target"] == "True":
            continue
        for metric in METRICS:
            try:
                value = float(row[metric])
            except (ValueError, TypeError):
                continue
            if math.isfinite(value):
                samples[row["id"]][metric].append(value)
    patients = defaultdict(lambda: defaultdict(list))
    for sample, metric_values in samples.items():
        match = re.match(r"^(patient\d+)", sample)
        if match is None:
            raise ValueError(f"Unexpected sample ID: {sample}")
        for metric, values in metric_values.items():
            if values:
                patients[metric][match.group(1)].append(float(np.mean(values)))
    return {
        metric: {patient: float(np.mean(values)) for patient, values in units.items()}
        for metric, units in patients.items()
    }


def load_archive() -> dict[str, dict[int, dict]]:
    expected = {}
    for line in MANIFEST.read_text().splitlines():
        digest, name = line.split("  ", 1)
        if name in expected:
            raise ValueError(f"Duplicate manifest path: {name}")
        expected[name] = digest
    arms = {arm: {} for arm in ARMS}
    seen = set()
    pattern = re.compile(
        r"results/(?:acdc_confirm_eval/acdc_confirm|followup_eval/acdc_followup)_"
        r"(decoder|kd_band|kd_full|kd_band_peer)_seed(20[1-9]|21[0-7])/"
        r"per_case_official_test\.csv"
    )
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError(f"Unexpected archive member: {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Cannot read archive member: {member.name}")
            data = stream.read()
            if hashlib.sha256(data).hexdigest() != expected[member.name]:
                raise ValueError(f"Checksum mismatch: {member.name}")
            seen.add(member.name)
            match = pattern.fullmatch(member.name)
            if match:
                arms[match.group(1)][int(match.group(2))] = sample_units(data)
    if seen != set(expected):
        raise ValueError(f"Missing archive members: {sorted(set(expected) - seen)}")
    for arm, runs in arms.items():
        if set(runs) != set(range(201, 218)):
            raise ValueError(f"Unexpected seed set for {arm}: {sorted(runs)}")
    print(f"Archive: {len(seen)} members matched SHA-256; 68 student case files found.")
    return arms


def calculate(a: dict[int, dict], b: dict[int, dict], metric: str) -> dict[str, float]:
    """Reimplement the frozen two-component estimator without importing its code."""
    units = sorted(set.intersection(*(set(run[metric]) for run in [*a.values(), *b.values()])))
    if len(units) != 50:
        raise ValueError(f"Expected 50 patients, got {len(units)} for {metric}")
    sign = -1 if metric in ("hd95", "assd") else 1
    x = np.array([[sign * run[metric][unit] for unit in units] for run in a.values()])
    y = np.array([[sign * run[metric][unit] for unit in units] for run in b.values()])
    xbar, ybar = x.mean(axis=1), y.mean(axis=1)
    delta = float(xbar.mean() - ybar.mean())
    va, vb = xbar.var(ddof=1) / len(xbar), ybar.var(ddof=1) / len(ybar)
    vu = (x.mean(axis=0) - y.mean(axis=0)).var(ddof=1) / len(units)
    se = math.sqrt(va + vb + vu)
    df = (va + vb) ** 2 / (va**2 / (len(xbar) - 1) + vb**2 / (len(ybar) - 1))
    margin = float(t.ppf(0.975, df) * se)
    return {
        "delta": delta,
        "low": delta - margin,
        "high": delta + margin,
        "p": float(2 * t.sf(abs(delta / se), df)),
    }


def main() -> None:
    arms = load_archive()
    with CONTRASTS.open(newline="") as stream:
        expected = {(r["id"], r["metric"]): r for r in csv.DictReader(stream)}
    errors = []
    for metric in METRICS:
        family = []
        for cid, first, second in COMPARISONS:
            result = calculate(arms[first], arms[second], metric)
            row = expected[cid, metric]
            errors.extend(abs(value - float(row[key])) for key, value in result.items())
            if cid != "S3":
                family.append((result["p"], row))
            if metric == "boundary_f1":
                print(
                    f"{cid}: Δ={result['delta']:+.8f} "
                    f"95% CI [{result['low']:+.8f}, {result['high']:+.8f}]"
                )
        running = 0.0
        for rank, (p, row) in enumerate(sorted(family, key=lambda item: item[0])):
            running = max(running, min(1.0, (len(family) - rank) * p))
            errors.append(abs(running - float(row["p_holm"])))
    maximum = max(errors)
    print(f"25 contrasts and 20 Holm values: maximum absolute discrepancy {maximum:.3g}.")
    if maximum > 1e-10:
        raise ValueError("Frozen contrast check exceeded tolerance 1e-10")


if __name__ == "__main__":
    main()
