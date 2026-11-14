#!/usr/bin/env python
"""Build a cross-dataset BDS-Lite status report from consolidated CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

TIE_TOLERANCE = 5e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare cross-dataset status report.")
    parser.add_argument(
        "--cross-dataset",
        default="results/tables/cross_dataset_bdslite_status.csv",
        help="Consolidated U-Net vs BDS-Lite status CSV.",
    )
    parser.add_argument(
        "--cost-ablation",
        default="results/tables/resource_profile_isic2018_boundary_branch_ablation.csv",
        help="Boundary-branch cost ablation CSV.",
    )
    parser.add_argument(
        "--output-csv",
        default="results/tables/cross_dataset_report_v1.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-md",
        default="docs/cross_dataset_report.md",
        help="Output Markdown report path.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def metric_status(delta: float, lower_is_better: bool = False) -> str:
    if abs(delta) <= TIE_TOLERANCE:
        return "tie"
    improved = delta < 0 if lower_is_better else delta > 0
    return "improved" if improved else "worse"


def pct_delta(value: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return 100.0 * (value - baseline) / baseline


def summarize_cost(rows: list[dict[str, str]]) -> dict[str, float]:
    by_variant = {row["variant"]: row for row in rows}
    baseline = by_variant["unet_baseline"]
    deploy = by_variant["bdslite_boundary_disabled"]
    retained = by_variant["bdslite_boundary_retained"]
    return {
        "deploy_params_over_unet_pct": pct_delta(
            float(deploy["active_params"]),
            float(baseline["active_params"]),
        ),
        "deploy_flops_over_unet_pct": pct_delta(float(deploy["flops"]), float(baseline["flops"])),
        "deploy_latency_over_unet_pct": pct_delta(
            float(deploy["latency_ms"]),
            float(baseline["latency_ms"]),
        ),
        "retained_params_over_unet_pct": pct_delta(
            float(retained["active_params"]),
            float(baseline["active_params"]),
        ),
        "retained_flops_over_unet_pct": pct_delta(
            float(retained["flops"]),
            float(baseline["flops"]),
        ),
        "retained_latency_over_deploy_pct": pct_delta(
            float(retained["latency_ms"]),
            float(deploy["latency_ms"]),
        ),
    }


def summarize_dataset(row: dict[str, str], cost: dict[str, float]) -> dict[str, Any]:
    dsc_status = metric_status(as_float(row, "delta_dsc"))
    iou_status = metric_status(as_float(row, "delta_iou"))
    hd95_status = metric_status(as_float(row, "delta_hd95"), lower_is_better=True)
    assd_status = metric_status(as_float(row, "delta_assd"), lower_is_better=True)
    boundary_status = metric_status(as_float(row, "delta_boundary_f1"))

    overlap_supported = dsc_status == "improved" and iou_status == "improved"
    distance_supported = hd95_status == "improved" and assd_status == "improved"
    boundary_supported = boundary_status in {"improved", "tie"}
    efficiency_supported = (
        cost["deploy_params_over_unet_pct"] < 1.0 and cost["deploy_flops_over_unet_pct"] < 1.0
    )

    if overlap_supported and distance_supported and boundary_supported and efficiency_supported:
        decision = "supports_claim"
    elif overlap_supported and boundary_supported and efficiency_supported:
        decision = "partial_support"
    else:
        decision = "does_not_support_claim"

    return {
        "dataset": row["dataset"],
        "variant": row["method_variant"],
        "dsc_status": dsc_status,
        "iou_status": iou_status,
        "hd95_status": hd95_status,
        "assd_status": assd_status,
        "boundary_f1_status": boundary_status,
        "source_claim_status": row["claim_status"],
        "report_decision": decision,
        "deploy_params_over_unet_pct": f"{cost['deploy_params_over_unet_pct']:.2f}",
        "deploy_flops_over_unet_pct": f"{cost['deploy_flops_over_unet_pct']:.2f}",
        "deploy_latency_over_unet_pct": f"{cost['deploy_latency_over_unet_pct']:.2f}",
        "notes": row["notes"],
    }


def write_markdown_report(
    path: Path,
    rows: list[dict[str, Any]],
    cost: dict[str, float],
) -> None:
    supported = sum(row["report_decision"] == "supports_claim" for row in rows)
    partial = sum(row["report_decision"] == "partial_support" for row in rows)
    failed = sum(row["report_decision"] == "does_not_support_claim" for row in rows)
    ready = failed == 0 and partial == 0 and supported == len(rows)

    lines = [
        "# Cross-Dataset BDS-Lite Report",
        "",
        "Date: 2026-04-27",
        "",
        "## Summary",
        "",
        f"- Datasets supporting the current claim: {supported}",
        f"- Datasets with partial support: {partial}",
        f"- Datasets not supporting the current claim: {failed}",
        f"- Ready for a full cross-dataset boundary-distance claim: {'yes' if ready else 'no'}",
        "",
        "The current evidence supports a resource-efficiency claim, but not yet a full",
        "cross-dataset boundary-distance claim because Synapse HD95 and ASSD regress.",
        "",
        "## Dataset Decisions",
        "",
        "| Dataset | DSC | IoU | HD95 | ASSD | Boundary F1 | Decision |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {dsc_status} | {iou_status} | {hd95_status} | {assd_status} "
            "| {boundary_f1_status} | {report_decision} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Efficiency",
            "",
            f"- Deploy params versus U-Net: {cost['deploy_params_over_unet_pct']:.2f}%",
            f"- Deploy FLOPs versus U-Net: {cost['deploy_flops_over_unet_pct']:.2f}%",
            f"- Deploy latency versus U-Net: {cost['deploy_latency_over_unet_pct']:.2f}%",
            f"- Retained-branch params versus U-Net: {cost['retained_params_over_unet_pct']:.2f}%",
            f"- Retained-branch FLOPs versus U-Net: {cost['retained_flops_over_unet_pct']:.2f}%",
            (
                "- Retained-branch latency versus deploy path: "
                f"{cost['retained_latency_over_deploy_pct']:.2f}%"
            ),
            "",
            "## Next Action",
            "",
            "Prioritize Synapse distance-metric failures before claiming consistent",
            "cross-dataset boundary improvement. The label 6/8 diagnostics should guide",
            "the next targeted loss or post-processing experiment.",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(
    cross_dataset_path: Path,
    cost_ablation_path: Path,
    output_csv_path: Path,
    output_md_path: Path,
) -> list[dict[str, Any]]:
    cross_rows = read_csv(cross_dataset_path)
    cost_rows = read_csv(cost_ablation_path)
    cost = summarize_cost(cost_rows)
    report_rows = [summarize_dataset(row, cost) for row in cross_rows]

    fieldnames = [
        "dataset",
        "variant",
        "dsc_status",
        "iou_status",
        "hd95_status",
        "assd_status",
        "boundary_f1_status",
        "source_claim_status",
        "report_decision",
        "deploy_params_over_unet_pct",
        "deploy_flops_over_unet_pct",
        "deploy_latency_over_unet_pct",
        "notes",
    ]
    write_csv(output_csv_path, fieldnames, report_rows)
    write_markdown_report(output_md_path, report_rows, cost)
    return report_rows


def main() -> None:
    args = parse_args()
    rows = build_report(
        cross_dataset_path=Path(args.cross_dataset),
        cost_ablation_path=Path(args.cost_ablation),
        output_csv_path=Path(args.output_csv),
        output_md_path=Path(args.output_md),
    )
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_md}")
    print(f"Summarized {len(rows)} datasets")


if __name__ == "__main__":
    main()
