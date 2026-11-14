"""Supplement to analyze.py (pre-registration section 4(c)): the cross-arm seed correlation.

analyze.py::paired computes `seed_correlation`, but analyze.py::row does not copy it into
contrasts.csv / results.json, so the value required by section 4(c) is missing from the outputs.
This script does not change analyze.py. It rebuilds the same inputs with analyze.py's own functions,
calls the unchanged analyze.paired, checks that delta/low/high/p equal the seed-paired rows already
written to revision/acdc_confirm/results/contrasts.csv, and writes the correlations to
revision/acdc_confirm/seed_correlation_supplement.json (descriptive only).

    .venv/bin/python scripts/acdc_confirm/post/seed_correlation.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analysis/revision"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import expansion_statistics as ex  # noqa: E402

from scripts.acdc_confirm import analyze  # noqa: E402
from scripts.acdc_confirm.common import SEEDS, WORKSPACE, failed_runs, run_name  # noqa: E402

LABELS = {
    "official_test": "replication seeds, primary set; seed-paired sensitivity",
    "internal_test": "replication seeds, internal 20-patient split (seen); seed-paired sensitivity",
    "val": "replication seeds, validation (selection set); seed-paired sensitivity",
}
OUT = WORKSPACE / "seed_correlation_supplement.json"


def main() -> None:
    failed = failed_runs()
    pairs = [
        s
        for s in SEEDS
        if run_name("kd_band", s) not in failed and run_name("decoder", s) not in failed
    ]
    with (WORKSPACE / "results/contrasts.csv").open() as stream:
        written = {(r["analysis"], r["metric"]): r for r in csv.DictReader(stream)}
    rows = []
    for split, label in LABELS.items():
        values = {
            arm: [ex.unit_values(analyze.new_path(arm, s, split), "acdc") for s in pairs]
            for arm in ("kd_band", "decoder")
        }
        for metric in analyze.METRICS:
            p = analyze.paired(values["kd_band"], values["decoder"], metric)
            ref = written[(label, metric)]
            for key in ("delta", "low", "high", "p"):
                if not math.isclose(p[key], float(ref[key]), rel_tol=0, abs_tol=1e-12):
                    raise ValueError(f"{split} {metric} {key}: {p[key]} != {ref[key]}")
            rows.append(
                dict(
                    split=split,
                    metric=metric,
                    n_pairs=p["n_pairs"],
                    seed_correlation=p["seed_correlation"],
                    delta=p["delta"],
                )
            )
    OUT.write_text(
        json.dumps(dict(source="analyze.paired (unchanged)", rows=rows), indent=2) + "\n"
    )
    for r in rows:
        print(f"{r['split']:13s} {r['metric']:11s} n={r['n_pairs']} r={r['seed_correlation']:+.3f}")


if __name__ == "__main__":
    main()
