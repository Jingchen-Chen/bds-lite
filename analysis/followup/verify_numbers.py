"""Independent check of the follow-up numbers in the manuscript tables and keys.

Recomputes every follow-up contrast from the raw per-case CSVs with code written separately from
expansion_statistics.contrast (pandas-free, plain csv + numpy + scipy), then compares each value,
rounded as printed, with revision/manuscript/generated/{tab_followup_main.tex,
tab_s_followup_all.tex, keys_followup.json}. Exits 1 on any mismatch.

    PYTHONPATH=src:. .venv/bin/python analysis/followup/verify_numbers.py
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "revision/manuscript/generated"
SEEDS = range(201, 218)
S3_EVAL, FU_EVAL = ROOT / "results/acdc_confirm_eval", ROOT / "results/followup_eval"
CSV = "per_case_official_test.csv"
PATHS = {
    "decoder": [S3_EVAL / f"acdc_confirm_decoder_seed{s}" / CSV for s in SEEDS],
    "kd_band": [S3_EVAL / f"acdc_confirm_kd_band_seed{s}" / CSV for s in SEEDS],
    "kd_full": [FU_EVAL / f"acdc_followup_kd_full_seed{s}" / CSV for s in SEEDS],
    "kd_band_peer": [FU_EVAL / f"acdc_followup_kd_band_peer_seed{s}" / CSV for s in SEEDS],
}
PAIRS = {
    "S3": ("kd_band", "decoder"),
    "C1": ("kd_band", "kd_full"),
    "C2": ("kd_full", "decoder"),
    "C3": ("kd_band_peer", "decoder"),
    "C4": ("kd_band", "kd_band_peer"),
}
METRICS = ["dsc", "iou", "boundary_f1", "hd95", "assd"]


def patient_table(path: Path, metric: str) -> dict[str, float]:
    """patient -> mean over slices of the class mean over defined rows (absent classes skipped)."""
    by_slice = defaultdict(list)
    with path.open() as stream:
        for row in csv.DictReader(stream):
            value = row[metric]
            if value in ("", "nan"):
                continue
            if row["empty_target"] == "True" and row["empty_prediction"] == "True":
                continue
            v = float(value)
            if math.isfinite(v):
                by_slice[row["id"]].append(v)
    by_patient = defaultdict(list)
    for sid, values in by_slice.items():
        by_patient[sid.split("_")[0]].append(np.mean(values))
    return {p: float(np.mean(v)) for p, v in by_patient.items()}


def contrast(a: list[dict], b: list[dict], metric: str) -> dict:
    sign = -1.0 if metric in ("hd95", "assd") else 1.0
    units = sorted(set.intersection(*[set(t) for t in a + b]))
    ma = np.array([[t[u] for u in units] for t in a]) * sign
    mb = np.array([[t[u] for u in units] for t in b]) * sign
    ra, rb = ma.mean(1), mb.mean(1)
    va, vb = ra.var(ddof=1) / len(ra), rb.var(ddof=1) / len(rb)
    d = ma.mean(0) - mb.mean(0)
    se = math.sqrt(va + vb + d.var(ddof=1) / len(units))
    df = (va + vb) ** 2 / (va**2 / (len(ra) - 1) + vb**2 / (len(rb) - 1))
    delta = ra.mean() - rb.mean()
    t = stats.t.ppf(0.975, df)
    return dict(
        delta=delta, low=delta - t * se, high=delta + t * se, p=2 * stats.t.sf(abs(delta / se), df)
    )


def fmt(x: float, digits: int) -> str:
    return f"{x:+.{digits}f}".replace("-", "$-$")


def main() -> None:
    tables = {
        m: {arm: [patient_table(p, m) for p in paths] for arm, paths in PATHS.items()}
        for m in METRICS
    }
    keys = json.loads((GEN / "keys_followup.json").read_text())
    main_tab = (GEN / "tab_followup_main.tex").read_text()
    all_tab = (GEN / "tab_s_followup_all.tex").read_text()
    problems, checked = [], 0
    for metric in METRICS:
        digits = 2 if metric in ("hd95", "assd") else 4
        holm_input = {}
        for cid, (a, b) in PAIRS.items():
            c = contrast(tables[metric][a], tables[metric][b], metric)
            if cid != "S3":
                holm_input[cid] = c["p"]
            cell = f"{fmt(c['delta'], digits)} [{fmt(c['low'], digits)}, {fmt(c['high'], digits)}]"
            if cell not in all_tab:
                problems.append(f"{cid} {metric}: {cell} not in tab_s_followup_all.tex")
            if (
                metric in ("boundary_f1", "dsc")
                and cell.replace(f" [{fmt(c['low'], digits)}", f" [{fmt(c['low'], digits)}")
                not in main_tab
            ):
                problems.append(f"{cid} {metric}: {cell} not in tab_followup_main.tex")
            for suffix, value in (("delta", c["delta"]), ("low", c["low"]), ("high", c["high"])):
                key = f"FU.{cid}.{metric}.{suffix}"
                if keys.get(key) != fmt(value, digits):
                    problems.append(
                        f"{key}: key {keys.get(key)} != recomputed {fmt(value, digits)}"
                    )
            checked += 1
        order = sorted(holm_input, key=holm_input.get)
        running = 0.0
        for rank, cid in enumerate(order):
            running = max(running, min(1.0, (len(order) - rank) * holm_input[cid]))
            key = f"FU.{cid}.{metric}.holm"
            printed = f"{running:.3f}" if running >= 0.001 else None
            if printed and keys.get(key) != printed:
                problems.append(f"{key}: key {keys.get(key)} != recomputed {printed}")
    unresolved = re.findall(r"\[FU\.[^\]]*\]", main_tab + all_tab)
    problems += [f"unresolved {u}" for u in unresolved]
    print(json.dumps(dict(contrasts_checked=checked, problems=problems), indent=2))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
