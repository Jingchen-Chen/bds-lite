#!/usr/bin/env python
"""Prepare publication-first evidence tables for the current BDS-Lite scope."""

from __future__ import annotations

import csv
from pathlib import Path

CHECKPOINTS_DIR = Path("results/checkpoints")
OUTPUT_CSV = Path("results/tables/publication_core_results_v1.csv")
OUTPUT_MD = Path("docs/publication_strategy_report.md")

METRICS = ["dsc", "iou", "hd95", "assd", "boundary_f1"]

ENTRIES = [
    {
        "dataset": "ISIC2018",
        "section": "positive_core",
        "method": "U-Net",
        "experiment": "isic2018_unet_baseline",
        "supports_main_claim": "reference",
        "notes": "Primary same-backbone baseline for ISIC2018.",
    },
    {
        "dataset": "ISIC2018",
        "section": "positive_core",
        "method": "BDS-Lite gate+surface",
        "experiment": "isic2018_bdslite_boundary_gate_damped_surface",
        "supports_main_claim": "yes",
        "notes": "Improves DSC, IoU, HD95, ASSD, and Boundary F1 versus U-Net.",
    },
    {
        "dataset": "ISIC2018",
        "section": "positive_core",
        "method": "MobileViT-UNet",
        "experiment": "isic2018_mobilevit_unet",
        "supports_main_claim": "complementary_baseline",
        "notes": "Efficient backbone baseline; not a boundary-distillation variant.",
    },
    {
        "dataset": "ACDC",
        "section": "positive_core",
        "method": "U-Net",
        "experiment": "acdc_unet_baseline",
        "supports_main_claim": "reference",
        "notes": "Primary same-backbone baseline for ACDC.",
    },
    {
        "dataset": "ACDC",
        "section": "positive_core",
        "method": "BDS-Lite gate+surface class13-mild",
        "experiment": "acdc_bdslite_boundary_gate_damped_surface_class13_mild",
        "supports_main_claim": "mostly_yes",
        "notes": "Improves DSC, IoU, HD95, and ASSD; Boundary F1 is effectively tied.",
    },
    {
        "dataset": "Synapse",
        "section": "diagnostic_limitation",
        "method": "U-Net",
        "experiment": "synapse_unet_baseline",
        "supports_main_claim": "reference",
        "notes": "Acceptance baseline for the one architectural rescue.",
    },
    {
        "dataset": "Synapse",
        "section": "diagnostic_limitation",
        "method": "BDS-Lite gate+surface",
        "experiment": "synapse_bdslite_boundary_gate_damped_surface",
        "supports_main_claim": "mixed_limitation",
        "notes": "Improves DSC, IoU, and Boundary F1 but worsens HD95 and ASSD.",
    },
    {
        "dataset": "Synapse",
        "section": "diagnostic_limitation",
        "method": "BDS-Lite HR rescue",
        "experiment": "synapse_bdslite_hr_rescue",
        "supports_main_claim": "pending",
        "notes": (
            "Only allowed Synapse architectural rescue; accepted only if HD95 and ASSD beat "
            "U-Net and DSC drop is no more than 0.005."
        ),
    },
]


def _read_mean(eval_csv: Path) -> dict[str, str] | None:
    if not eval_csv.exists():
        return None
    with eval_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    mean_rows = [row for row in rows if row.get("label") == "mean"]
    row = mean_rows[0] if mean_rows else rows[-1]
    return {metric: row.get(metric, "missing") for metric in METRICS}


def _metric_row(entry: dict[str, str]) -> dict[str, str]:
    eval_csv = CHECKPOINTS_DIR / entry["experiment"] / "eval_val.csv"
    metrics = _read_mean(eval_csv)
    if metrics is None:
        metrics = {metric: "missing" for metric in METRICS}
    return {
        "dataset": entry["dataset"],
        "section": entry["section"],
        "method": entry["method"],
        "experiment": entry["experiment"],
        "split": "val",
        **metrics,
        "supports_main_claim": entry["supports_main_claim"],
        "notes": entry["notes"],
        "missing_command": (
            "" if eval_csv.exists() else _missing_eval_command(entry["experiment"])
        ),
    }


def _missing_eval_command(experiment: str) -> str:
    return (
        "PYTHONPATH=src .venv/bin/python scripts/evaluate.py "
        f"--checkpoint results/checkpoints/{experiment}/best.pt --split val"
    )


def _as_float(row: dict[str, str], metric: str) -> float | None:
    value = row.get(metric, "missing")
    if value in {"", "missing", "nan"}:
        return None
    return float(value)


def _hr_rescue_decision(rows: list[dict[str, str]]) -> str:
    by_experiment = {row["experiment"]: row for row in rows}
    baseline = by_experiment.get("synapse_unet_baseline")
    hr = by_experiment.get("synapse_bdslite_hr_rescue")
    if baseline is None or hr is None or _as_float(hr, "dsc") is None:
        return "not evaluated"

    baseline_dsc = _as_float(baseline, "dsc")
    baseline_hd95 = _as_float(baseline, "hd95")
    baseline_assd = _as_float(baseline, "assd")
    hr_dsc = _as_float(hr, "dsc")
    hr_hd95 = _as_float(hr, "hd95")
    hr_assd = _as_float(hr, "assd")
    if None in {baseline_dsc, baseline_hd95, baseline_assd, hr_dsc, hr_hd95, hr_assd}:
        return "not evaluated"

    accepted = (
        hr_hd95 < baseline_hd95 and hr_assd < baseline_assd and hr_dsc >= baseline_dsc - 0.005
    )
    return "accepted" if accepted else "rejected"


def _fmt(value: str, metric: str) -> str:
    if value == "missing":
        return "missing"
    precision = 2 if metric in {"hd95", "assd"} else 4
    return f"{float(value):.{precision}f}"


def write_outputs(rows: list[dict[str, str]]) -> None:
    hr_decision = _hr_rescue_decision(rows)
    for row in rows:
        if row["experiment"] == "synapse_bdslite_hr_rescue" and hr_decision != "not evaluated":
            row["supports_main_claim"] = hr_decision

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "section",
        "method",
        "experiment",
        "split",
        *METRICS,
        "supports_main_claim",
        "notes",
        "missing_command",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Publication Strategy Report",
        "",
        "Date: 2026-04-27",
        "",
        "## Strategy",
        "",
        (
            "- Main paper claim: training-time boundary distillation for lightweight "
            "2D medical image segmentation with minimal deploy overhead."
        ),
        "- Positive core datasets: ISIC2018 and ACDC.",
        (
            "- Synapse status: diagnostic limitation unless the HR rescue beats U-Net "
            "on both HD95 and ASSD without more than 0.005 DSC drop."
        ),
        (
            "- Exhausted Synapse paths: loss-only, checkpoint sweep, "
            "LCC post-processing, hard class weights, init-from-U-Net, "
            "and two-stage encoder freezing."
        ),
        "",
        "## Positive Core",
        "",
        "| Dataset | Method | DSC | IoU | HD95 | ASSD | BF1 | Claim |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        if row["section"] != "positive_core":
            continue
        lines.append(
            "| {dataset} | {method} | {dsc} | {iou} | {hd95} | {assd} | {bf1} | {claim} |".format(
                dataset=row["dataset"],
                method=row["method"],
                dsc=_fmt(row["dsc"], "dsc"),
                iou=_fmt(row["iou"], "iou"),
                hd95=_fmt(row["hd95"], "hd95"),
                assd=_fmt(row["assd"], "assd"),
                bf1=_fmt(row["boundary_f1"], "boundary_f1"),
                claim=row["supports_main_claim"],
            )
        )

    lines.extend(
        [
            "",
            "## Diagnostic Limitation",
            "",
            "| Dataset | Method | DSC | IoU | HD95 | ASSD | BF1 | Claim |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        if row["section"] != "diagnostic_limitation":
            continue
        lines.append(
            "| {dataset} | {method} | {dsc} | {iou} | {hd95} | {assd} | {bf1} | {claim} |".format(
                dataset=row["dataset"],
                method=row["method"],
                dsc=_fmt(row["dsc"], "dsc"),
                iou=_fmt(row["iou"], "iou"),
                hd95=_fmt(row["hd95"], "hd95"),
                assd=_fmt(row["assd"], "assd"),
                bf1=_fmt(row["boundary_f1"], "boundary_f1"),
                claim=row["supports_main_claim"],
            )
        )

    lines.extend(
        [
            "",
            "## Synapse HR Rescue Gate",
            "",
            f"- HR rescue decision: {hr_decision}.",
            (
                "- Acceptance threshold: HD95 lower than Synapse U-Net, ASSD lower "
                "than Synapse U-Net, and DSC no more than 0.005 below U-Net."
            ),
            (
                "- If rejected or not evaluated, Synapse remains a controlled limitation "
                "and manuscript work should proceed around ISIC2018 plus ACDC."
            ),
            "",
            "## Missing Metrics",
            "",
        ]
    )
    missing_rows = [row for row in rows if any(row[metric] == "missing" for metric in METRICS)]
    if not missing_rows:
        lines.append("- None.")
    else:
        for row in missing_rows:
            lines.append(f"- {row['experiment']}: `{row['missing_command']}`")

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = [_metric_row(entry) for entry in ENTRIES]
    write_outputs(rows)
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
