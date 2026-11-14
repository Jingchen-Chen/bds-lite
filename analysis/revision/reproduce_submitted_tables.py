# ruff: noqa: E501
"""Reproduce the originally submitted Table 2 (ACDC/Synapse) and Table 3 from the submitted
checkpoints with the revision evaluator, using the original conventions.

Table 2 convention (scripts/reeval_main_seeds.py "foreground_macro_mean"): per-class mean over
samples, then mean over foreground classes; see table2_value. Table 3 convention:
analysis/generate_rescue_analysis.py (both-empty undefined; per-sample class mean; cluster means
of oriented paired deltas). Inputs: revision/extra_cohorts/famO_{unet,full}; references:
revision/sources/github_release/{main_test_means.csv, cluster_level_statistics_submitted.csv}.
Output: revision/validation/submitted_tables_reproduction.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from analysis.revision.family_statistics import compare

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "revision/sources/github_release"
COHORT = {"unet": "famO_unet", "bds_lite_full": "famO_full"}


def table2_value(cohort: str, dataset: str, seed: int, metric: str) -> float:
    """Original reeval_main_seeds.py aggregation: per-class nanmean over samples, then mean over
    foreground classes; classes absent from both prediction and reference score 1 for
    DSC/IoU/Boundary F1 and are undefined for HD95/ASSD."""
    rows = list(
        csv.DictReader(
            (
                ROOT / f"revision/extra_cohorts/{cohort}/{dataset}_seed{seed}/test/per_case.csv"
            ).open()
        )
    )
    by_class: dict[str, list[float]] = {}
    for r in rows:
        both_empty = r["empty_prediction"] == "True" and r["empty_target"] == "True"
        if metric in ("dsc", "iou", "boundary_f1"):
            value = 1.0 if both_empty else float(r[metric])
        else:
            value = float(r[metric]) if r[metric] not in ("", "None") else np.nan
        by_class.setdefault(r["label"], []).append(value)
    return float(np.mean([np.nanmean(v) for v in by_class.values()]))


def main() -> None:
    out = []
    for r in csv.DictReader((SRC / "main_test_means.csv").open()):
        if r["dataset"] == "isic2018" or r["method"] not in COHORT:
            continue
        published = [float(x) for x in r["seed_values"].split("|")]
        mine = [table2_value(COHORT[r["method"]], r["dataset"], s, r["metric"]) for s in (1, 2, 3)]
        out.append(
            dict(
                table="2",
                dataset=r["dataset"],
                method=r["method"],
                metric=r["metric"],
                published=";".join(f"{x:.6f}" for x in published),
                reproduced=";".join(f"{x:.6f}" for x in mine),
                max_abs_diff=max(abs(a - b) for a, b in zip(published, mine, strict=True)),
            )
        )
    for r in csv.DictReader((SRC / "cluster_level_statistics_submitted.csv").open()):
        if r["dataset"] == "isic2018":
            continue
        mine = compare("S6", "S0", r["dataset"], r["metric"])
        out.append(
            dict(
                table="3",
                dataset=r["dataset"],
                method="BDS-Lite vs U-Net",
                metric=r["metric"],
                published=f"{float(r['mean_oriented_delta']):.6f} W={r['wilcoxon_statistic']} p={float(r['p_value_two_sided']):.6g}",
                reproduced=f"{mine['mean_oriented_delta']:.6f} W={mine['wilcoxon_W']} p={mine['p_raw']:.6g}",
                max_abs_diff=abs(float(r["mean_oriented_delta"]) - mine["mean_oriented_delta"]),
            )
        )
    path = ROOT / "revision/validation/submitted_tables_reproduction.csv"
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(out[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(out)
    for r in out:
        print(
            r["table"],
            r["dataset"],
            r["method"][:12],
            r["metric"],
            f"{r['max_abs_diff']:.2e}",
            "|",
            r["published"],
            "|",
            r["reproduced"],
        )


if __name__ == "__main__":
    main()
