# ruff: noqa: E501, E741
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import yaml

pd.set_option(
    "display.width",
    230,
    "display.max_columns",
    40,
    "display.max_rows",
    300,
    "display.float_format",
    lambda x: f"{x:.4f}",
)
stage2 = set(r["run_id"] for r in yaml.safe_load(open("revision/queue_confirmation.yaml"))["runs"])
rows = []
# November revision runs, ISIC only, stage2 excluded
for p in sorted(glob.glob("results/revision_runs/isic2018_*_seed*/final_metrics.json")):
    rid = p.split("/")[2]
    if rid in stage2:
        continue
    m = re.match(r"isic2018_(.+)_seed(\d+)$", rid)
    arm, seed = m.group(1), int(m.group(2))
    if seed > 10:
        continue
    j = json.load(open(p))
    r = dict(
        src="nov",
        arm=arm,
        seed=seed,
        sel=j["selected_epoch"],
        valdsc=j["metrics"]["val"]["dsc"]["mean"],
        valbf1=j["metrics"]["val"]["boundary_f1"]["mean"],
        valhd=j["metrics"]["val"]["hd95"]["mean"],
    )
    if "test" in j["metrics"]:
        r.update(
            testdsc=j["metrics"]["test"]["dsc"]["mean"],
            testbf1=j["metrics"]["test"]["boundary_f1"]["mean"],
            testhd=j["metrics"]["test"]["hd95"]["mean"],
        )
    rows.append(r)
# October arms on the same validation images
base = "recovered/followup/results"
for root, arms in [
    ("followup_20261018_v2", "USBF"),
    ("pathway_controls_20261018", "GC"),
    ("boundary_specificity_20261018", "D"),
    ("region_baseline_20261018", ["UA"]),
]:
    for a in arms:
        for s in (1, 2, 3):
            d = f"{base}/{root}/isic2018_{a}_seed{s}"
            pc = pd.read_csv(d + "/per_case.csv")
            sm = json.load(open(d + "/summary.json")) if os.path.exists(d + "/summary.json") else {}
            rows.append(
                dict(
                    src="oct",
                    arm=a,
                    seed=s,
                    sel=sm.get("best_epoch", sm.get("selected_epoch")),
                    valdsc=pc["dsc"].mean() if "dsc" in pc else np.nan,
                    valbf1=pc["boundary_f1"].mean() if "boundary_f1" in pc else np.nan,
                    valhd=pc["hd95"].mean() if "hd95" in pc else np.nan,
                )
            )
R = pd.DataFrame(rows)
R.to_csv("revision/audit/project_audit_20261107/isic_val_test_runs.csv", index=False)
print(
    R.groupby(["src", "arm"])
    .agg(
        n=("seed", "count"),
        seeds=("seed", lambda x: f"{x.min()}-{x.max()}"),
        sel=("sel", "mean"),
        valdsc=("valdsc", "mean"),
        valbf1=("valbf1", "mean"),
        valbf1sd=("valbf1", "std"),
        testdsc=("testdsc", "mean"),
        testbf1=("testbf1", "mean"),
        testbf1sd=("testbf1", "std"),
    )
    .to_string()
)
