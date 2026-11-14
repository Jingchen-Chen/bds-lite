# ruff: noqa: E501
"""Independently verify central PeerJ submission numbers from existing per-case CSVs.

Reimplements aggregation, uncertainty and Holm correction using pandas/NumPy/SciPy;
imports expansion_statistics only for path resolution, never its statistical helpers.
Does not read Experiment C, run inference/training, or rewrite frozen results.
Usage: PYTHONPATH=src:. .venv/bin/python analysis/revision/verify_editorial_numbers.py
Output: revision/validation/editorial_numerical_verification.csv (absolute tolerance 1e-10).
"""

import csv
import importlib
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "analysis/revision"))
paths = importlib.import_module("expansion_statistics")  # path resolution only

metrics = ["dsc", "iou", "boundary_f1", "hd95", "assd"]
checks = []


@lru_cache(None)
def raw(path, ds, label=None):
    frame = pd.read_csv(path)
    if label is not None:
        frame = frame[frame["label"] == label].copy()
    both = frame["empty_prediction"].astype(str).str.lower().eq("true") & frame[
        "empty_target"
    ].astype(str).str.lower().eq("true")
    frame.loc[both, metrics] = np.nan
    sample = frame.groupby("id", sort=True)[metrics].mean()
    if ds != "isic2018":
        unit = sample.index.str.extract(r"^(patient\d+|case\d+)", expand=False)
        sample = sample.groupby(unit, sort=True).mean()
    return sample


def calc(aa, bb, m, pooled=False):
    common = set.intersection(*(set(x.index[x[m].notna()]) for x in aa + bb))
    ids = sorted(common)
    sign = -1 if m in ["hd95", "assd"] else 1
    a = sign * np.array([x.loc[ids, m].to_numpy() for x in aa])
    b = sign * np.array([x.loc[ids, m].to_numpy() for x in bb])
    ra, rb = a.mean(axis=1), b.mean(axis=1)
    na, nb = len(ra), len(rb)
    va, vb = np.var(ra, ddof=1) / na, np.var(rb, ddof=1) / nb
    if pooled:
        df = na + nb - 2
        vrun = ((np.sum((ra - ra.mean()) ** 2) + np.sum((rb - rb.mean()) ** 2)) / df) * (
            1 / na + 1 / nb
        )
    else:
        vrun = va + vb
        df = vrun**2 / (va**2 / (na - 1) + vb**2 / (nb - 1))
    perunit = a.mean(axis=0) - b.mean(axis=0)
    se = math.sqrt(vrun + np.var(perunit, ddof=1) / len(ids))
    d = ra.mean() - rb.mean()
    half = stats.t.ppf(0.975, df) * se
    return dict(
        delta=d,
        estimate=d,
        low=d - half,
        high=d + half,
        p=2 * stats.t.sf(abs(d / se), df),
        se=se,
        df=df,
        n_a=na,
        n_b=nb,
        arm_a_mean=sign * ra.mean(),
        arm_b_mean=sign * rb.mean(),
        arm_a_sd=ra.std(ddof=1),
        arm_b_sd=rb.std(ddof=1),
    )


def record(name, actual, expected, fields, source):
    for field in fields:
        error = abs(float(actual[field]) - float(expected[field]))
        checks.append(
            dict(
                claim=name,
                quantity=field,
                recomputed=float(actual[field]),
                recorded=float(expected[field]),
                absolute_error=error,
                tolerance=1e-10,
                passed=error < 1e-10,
                source=str(source),
            )
        )


conf = root / "revision/acdc_confirm/results"
csvrows = pd.read_csv(conf / "contrasts.csv").to_dict("records")
perclass = pd.read_csv(conf / "per_class.csv").to_dict("records")
prim = json.loads((conf / "results.json").read_text())
cf = {
    arm: [
        raw(
            str(
                root
                / f"results/acdc_confirm_eval/acdc_confirm_{arm}_seed{s}/per_case_official_test.csv"
            ),
            "acdc",
        )
        for s in range(201, 218)
    ]
    for arm in ["kd_band", "decoder"]
}
for m in metrics:
    expected = next(
        r for r in csvrows if r["analysis"] == "replication seeds, primary set" and r["metric"] == m
    )
    got = calc(cf["kd_band"], cf["decoder"], m)
    record(
        "ACDC confirmation " + m,
        got,
        expected,
        [
            "delta",
            "low",
            "high",
            "p",
            "df",
            "n_a",
            "n_b",
            "arm_a_mean",
            "arm_b_mean",
            "arm_a_sd",
            "arm_b_sd",
        ],
        conf / "contrasts.csv",
    )
    if m == "boundary_f1":
        record(
            "ACDC primary JSON",
            got,
            prim["primary"]["contrast"],
            ["delta", "low", "high", "p"],
            conf / "results.json",
        )
        print("Independent ACDC primary:", got)
for label, structure in [(1, "RV"), (2, "myocardium"), (3, "LV")]:
    arms = {
        arm: [
            raw(
                str(
                    root
                    / f"results/acdc_confirm_eval/acdc_confirm_{arm}_seed{s}/per_case_official_test.csv"
                ),
                "acdc",
                label,
            )
            for s in range(201, 218)
        ]
        for arm in ["kd_band", "decoder"]
    }
    for m in ["dsc", "boundary_f1", "hd95"]:
        got = calc(arms["kd_band"], arms["decoder"], m)
        expected = next(r for r in perclass if r["structure"] == structure and r["metric"] == m)
        record(
            "ACDC " + structure + " " + m,
            got,
            expected,
            ["delta", "low", "high", "p"],
            conf / "per_class.csv",
        )
# Ten-run arms, both all seeds and untouched confirmation seeds, all six contrasts and metrics.
expfile = root / "revision/tables/families/expansion_contrasts.csv"
rows = pd.read_csv(expfile).to_dict("records")
exp = {
    (arm, s): raw(str(paths.isic_path(arm, s)), "isic2018")
    for arm in ["unet", "decoder", "full", "gsl"]
    for s in range(1, 11)
}
for m in metrics:
    group = [r for r in rows if r["metric"] == m]
    for subset, seeds in [("all", range(1, 11)), ("fresh", range(4, 11))]:
        gotrows = []
        for r in group:
            got = calc(
                [exp[(r["first"], s)] for s in seeds], [exp[(r["second"], s)] for s in seeds], m
            )
            record(
                "ISIC " + r["first"] + " - " + r["second"] + " " + m + " " + subset,
                got,
                {k: r[subset + "_" + k] for k in ["delta", "low", "high", "p"]},
                ["delta", "low", "high", "p"],
                expfile,
            )
            gotrows.append((r, got))
        adjusted = 0
        for rank, (r, got) in enumerate(sorted(gotrows, key=lambda x: x[1]["p"])):
            adjusted = max(adjusted, min(1.0, (len(gotrows) - rank) * got["p"]))
            record(
                "ISIC Holm " + r["first"] + " - " + r["second"] + " " + m + " " + subset,
                {"p": adjusted},
                {"p": r[subset + "_p_holm"]},
                ["p"],
                expfile,
            )
# Study-1 central contrasts use pooled residual run variance, NOT Welch degrees of freedom.
intfile = root / "revision/tables/families/expansion_interaction.csv"
ints = pd.read_csv(intfile).to_dict("records")
effects = [
    ("2x2", "decoder without surface", "M2", "M0"),
    ("transfer", "distillation+gate without surface", "M3", "M2"),
    ("transfer", "distillation+gate with surface", "M6", "M7"),
]
for mode, effect, first, second in effects:
    for m in ["boundary_f1", "dsc", "hd95"]:
        parts = []
        for ds in ["isic2018", "acdc", "synapse"]:
            aa = [raw(str(paths.cell_path(first, ds, s, False)), ds) for s in range(1, 4)]
            bb = [raw(str(paths.cell_path(second, ds, s, False)), ds) for s in range(1, 4)]
            got = calc(aa, bb, m, pooled=True)
            expected = next(
                r
                for r in ints
                if r["set"] == mode
                and r["effect"] == effect
                and r["metric"] == m
                and r["dataset"] == ds
                and r["seeds"] == "1-3"
            )
            record(
                "Study1 " + effect + " " + ds + " " + m,
                got,
                expected,
                ["estimate", "low", "high", "se", "df"],
                intfile,
            )
            parts.append(got)
        weights = np.array([1 / r["se"] ** 2 for r in parts])
        values = np.array([r["delta"] for r in parts])
        d = np.dot(weights, values) / weights.sum()
        se = math.sqrt(1 / weights.sum())
        got = dict(estimate=d, low=d - 1.96 * se, high=d + 1.96 * se)
        expected = next(
            r
            for r in ints
            if r["set"] == mode
            and r["effect"] == effect
            and r["metric"] == m
            and r["dataset"].startswith("pooled")
            and r["seeds"] == "1-3"
        )
        record(
            "Study1 pooled " + effect + " " + m, got, expected, ["estimate", "low", "high"], intfile
        )
# Exploratory KD: primary dataset family and its Holm correction.
kfile = root / "revision/kd_unified/contrasts.csv"
krows = pd.read_csv(kfile).to_dict("records")
family = []
for ds in ["isic2018", "acdc", "synapse"]:
    aa = [
        raw(str(root / f"results/revision_runs/{ds}_kd_band_seed{s}/per_case_test.csv"), ds)
        for s in range(1, 4)
    ]
    bb = (
        [exp[("decoder", s)] for s in range(1, 11)]
        if ds == "isic2018"
        else [raw(str(paths.cell_path("M2", ds, s, False)), ds) for s in range(1, 4)]
    )
    got = calc(aa, bb, "boundary_f1")
    expected = next(
        r
        for r in krows
        if r["dataset"] == ds
        and r["comparison"] == "KD-band - decoder-only"
        and r["metric"] == "boundary_f1"
        and r["reference_seeds"] == ("1-10" if ds == "isic2018" else "1-3")
    )
    record("Exploratory KD " + ds, got, expected, ["delta", "low", "high", "p"], kfile)
    family.append((ds, got, expected))
adjusted = 0
for rank, (ds, got, expected) in enumerate(sorted(family, key=lambda x: x[1]["p"])):
    adjusted = max(adjusted, min(1.0, (3 - rank) * got["p"]))
    record(
        "Exploratory Holm " + ds,
        {"p": adjusted},
        {"p": expected["p_holm_over_datasets"]},
        ["p"],
        kfile,
    )
# Store audit independently of frozen result directories.
out = root / "revision/validation/editorial_numerical_verification.csv"
with out.open("w") as f:
    w = csv.DictWriter(f, fieldnames=list(checks[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(checks)
bad = [r for r in checks if not r["passed"]]
print(
    len(checks),
    "quantities;",
    len(bad),
    "mismatches; max absolute error",
    max(r["absolute_error"] for r in checks),
)
for r in bad:
    print(r)
raise SystemExit(bool(bad))
