"""Mechanism readout for D+KD-band (protocol 7, secondary, descriptive). Phase 2 only, after step 8.

Uses analysis/revision/boundary_transfer_probe.py unchanged (subset, collect, ridge_ap,
average_precision; same image subsets, pixel sampling seeds 1000+seed / 2000+seed, and ridge
settings). Seeds are aligned with the recorded decoder-only probe: D+KD-band seeds 1-3 against the
decoder-only rows (arm M2, seeds 1-3) of results/tables/families/boundary_transfer_probe.csv.
Refuses to run before the step-8 test evaluation of D+KD-band exists.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from analysis.revision import boundary_transfer_probe as btp  # noqa: E402
from bds_lite.models import BDSLiteUNet  # noqa: E402

RUNS = ROOT / "results/revision_runs"
OUT = ROOT / "results/kd_isic2018/phase2/mechanism.csv"
SEEDS = (1, 2, 3)


def main() -> None:
    for seed in range(1, 11):
        if not (RUNS / f"isic2018_kd_band_seed{seed}/test_metrics.json").exists():
            raise RuntimeError(
                "mechanism probe is Phase 2 only: run analysis/kd/evaluate_test.py first"
            )
    sets = {s: btp.subset("isic2018", s) for s in ("val", "test")}
    rows = []
    for seed in SEEDS:
        payload = torch.load(
            RUNS / f"isic2018_kd_band_seed{seed}/best.pt", map_location="cpu", weights_only=False
        )
        model = BDSLiteUNet(3, 2, 32, 1, True, 0.0)
        model.load_state_dict(payload["model"], strict=True)
        model = model.to(btp.DEVICE).eval()
        val = btp.collect(model, "M2", sets["val"], 1000 + seed)
        test = btp.collect(model, "M2", sets["test"], 2000 + seed)
        row = dict(dataset="isic2018", arm="KD_band", seed=seed, prevalence=float(test["y"].mean()))
        row["probe_F_s"] = btp.ridge_ap(val["F"], val["y"], test["F"], test["y"])
        row["probe_logit"] = btp.ridge_ap(val["logit"], val["y"], test["logit"], test["y"])
        rows.append(row)
        print(json.dumps(row), flush=True)
    with btp.OUT.open() as stream:
        reference = [
            r for r in csv.DictReader(stream) if r["dataset"] == "isic2018" and r["arm"] == "M2"
        ]
    for r in reference:
        rows.append(
            dict(
                dataset="isic2018",
                arm="decoder_recorded",
                seed=int(float(r["seed"])),
                prevalence=float(r["prevalence"]),
                probe_F_s=float(r["probe_F_s"]),
                probe_logit=float(r["probe_logit"]),
            )
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
