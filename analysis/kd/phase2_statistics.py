# ruff: noqa: E501
"""Frozen Phase 2 analysis (protocol 7). Run only after analysis/kd/evaluate_test.py (step 8).

C1 (primary): D+KD-band fresh seeds 4-10 minus decoder-only seeds 1-10, test Boundary F1, alpha 0.05.
C3 (secondary): D+KD-band seeds 4-10 minus D+KD-full seeds 1-10, tested only if C1 is a confirmed
gain (fixed sequence; Q2 is conditional on Q1 showing a gain). Confirmed: p < 0.05, interval excludes zero, all-seed estimate has the same sign.
Estimator: analysis/revision/expansion_statistics.py::contrast, unchanged. Descriptive: all-seed
versions, all five metrics, missed-structure and false-alarm rates, all-case (penalised) HD95/ASSD,
the new arms against BDS-Lite and U-Net, and the same contrasts on validation images.

--dry-run substitutes recorded arms (BDS-Lite for D+KD-band, U-Net+GSL for D+KD-full), writes only to
the given scratch path, and exists only to test this code on results already known.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis/revision"))
sys.path.insert(0, str(ROOT))
import expansion_statistics as ex  # noqa: E402

from analysis.revision import make_audit_assets as ma  # noqa: E402

RUNS = ROOT / "results/revision_runs"
OUT = ROOT / "results/kd_isic2018/phase2"
FRESH, ALL = list(range(4, 11)), list(range(1, 11))


def path(arm: str, seed: int, split: str, dry: bool) -> Path:
    if arm in ("kd_band", "kd_full"):
        if dry:
            stand_in = {"kd_band": "full", "kd_full": "gsl"}[arm]
            return (
                ex.isic_path(stand_in, seed)
                if split == "test"
                else ma.validation_path(stand_in, seed)
            )
        return RUNS / f"isic2018_{arm}_seed{seed}/per_case_{split}.csv"
    return ex.isic_path(arm, seed) if split == "test" else ma.validation_path(arm, seed)


def values(arm, seeds, split, dry):
    return [ex.unit_values(path(arm, s, split, dry), "isic2018") for s in seeds]


def confirmed(fresh: dict, overall: dict) -> bool:
    return (
        fresh["p"] < 0.05
        and (fresh["low"] > 0 or fresh["high"] < 0)
        and fresh["delta"] * overall["delta"] > 0
    )


def rates(arm, seeds, dry) -> dict:
    """Missed-structure and false-alarm rates and all-case (penalised) distances, per arm."""
    out = {"missed": [], "false_alarm": [], "hd95_penalized": [], "assd_penalized": []}
    for s in seeds:
        rows = list(csv.DictReader(path(arm, s, "test", dry).open()))
        target = [r["empty_target"] == "False" for r in rows]
        empty = [r["empty_prediction"] == "True" for r in rows]
        eligible_m = sum(target)
        eligible_f = len(rows) - eligible_m
        out["missed"].append(
            sum(t and e for t, e in zip(target, empty, strict=True)) / max(eligible_m, 1)
        )
        out["false_alarm"].append(
            sum((not t) and (not e) for t, e in zip(target, empty, strict=True)) / eligible_f
            if eligible_f
            else float("nan")
        )
        for key in ("hd95_penalized", "assd_penalized"):
            out[key].append(float(np.mean([float(r[key]) for r in rows])))
    return {
        k: float(np.nanmean(v)) if v and not np.isnan(v).all() else None  # None: no eligible image
        for k, v in out.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", type=Path, default=None)
    args = parser.parse_args()
    dry = args.dry_run is not None
    out = args.dry_run if dry else OUT
    rows = []
    c1 = {}
    for metric in ex.METRICS:
        fresh = ex.contrast(
            values("kd_band", FRESH, "test", dry), values("decoder", ALL, "test", dry), metric
        )
        overall = ex.contrast(
            values("kd_band", ALL, "test", dry), values("decoder", ALL, "test", dry), metric
        )
        c1[metric] = (fresh, overall)
        rows.append(
            dict(
                contrast="C1",
                metric=metric,
                fresh=fresh,
                all_seeds=overall,
                tested=metric == "boundary_f1",
                status=("confirmed" if confirmed(fresh, overall) else "not established")
                if metric == "boundary_f1"
                else "descriptive",
            )
        )
    c1_confirmed = confirmed(*c1["boundary_f1"])
    # Q2 is asked only if Q1 shows a gain: C3 (and Phase 2b) follow a confirmed positive C1 only.
    c1_gain = c1_confirmed and c1["boundary_f1"][0]["delta"] > 0
    full_present = all((RUNS / f"isic2018_kd_full_seed{s}/per_case_test.csv").exists() for s in ALL)
    if c1_gain and (full_present or dry):
        for metric in ex.METRICS:
            fresh = ex.contrast(
                values("kd_band", FRESH, "test", dry), values("kd_full", ALL, "test", dry), metric
            )
            overall = ex.contrast(
                values("kd_band", ALL, "test", dry), values("kd_full", ALL, "test", dry), metric
            )
            rows.append(
                dict(
                    contrast="C3",
                    metric=metric,
                    fresh=fresh,
                    all_seeds=overall,
                    tested=metric == "boundary_f1",
                    status=("confirmed" if confirmed(fresh, overall) else "not established")
                    if metric == "boundary_f1"
                    else "descriptive",
                )
            )
    for other in ("full", "unet"):
        for metric in ex.METRICS:
            overall = ex.contrast(
                values("kd_band", ALL, "test", dry), values(other, ALL, "test", dry), metric
            )
            rows.append(
                dict(
                    contrast=f"kd_band vs {other}",
                    metric=metric,
                    all_seeds=overall,
                    tested=False,
                    status="descriptive",
                )
            )
    for metric in ex.METRICS:
        overall = ex.contrast(
            values("kd_band", ALL, "val", dry), values("decoder", ALL, "val", dry), metric
        )
        rows.append(
            dict(
                contrast="C1 validation",
                metric=metric,
                all_seeds=overall,
                tested=False,
                status="descriptive",
            )
        )
    result = dict(
        dry_run=dry,
        c1_confirmed=c1_confirmed,
        c1_confirmed_gain=c1_gain,
        c3_run=c1_gain and (full_present or dry),
        contrasts=rows,
        rates={arm: rates(arm, ALL, dry) for arm in ("kd_band", "decoder")},
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase2_statistics.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    for row in rows:
        if row["metric"] == "boundary_f1":
            a = row["all_seeds"]
            fresh = row.get("fresh")
            text = f"fresh {fresh['delta']:+.4f} p={fresh['p']:.3f} | " if fresh else ""
            print(
                f"{row['contrast']}: {text}all {a['delta']:+.4f} [{a['low']:+.4f}, {a['high']:+.4f}] {row['status']}"
            )


if __name__ == "__main__":
    main()
