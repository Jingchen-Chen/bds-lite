#!/usr/bin/env python
"""Prepare a final ablation status snapshot from completed experiment artifacts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare final ablation status report.")
    parser.add_argument(
        "--month4-summary",
        default="results/tables/month4_ablation_summary.csv",
        help="Month 4 cross-dataset ablation summary CSV.",
    )
    parser.add_argument(
        "--cost-ablation",
        default="results/tables/resource_profile_isic2018_boundary_branch_ablation.csv",
        help="Boundary-branch cost ablation CSV.",
    )
    parser.add_argument(
        "--synapse-unet",
        default="results/checkpoints/synapse_unet_baseline/eval_val.csv",
        help="Synapse U-Net validation CSV.",
    )
    parser.add_argument(
        "--synapse-bds",
        default="results/checkpoints/synapse_bdslite_boundary_gate_damped_surface/eval_val.csv",
        help="Synapse BDS-Lite validation CSV.",
    )
    parser.add_argument(
        "--synapse-unet-lcc",
        default="results/checkpoints/synapse_unet_baseline/eval_val_lcc.csv",
        help="Synapse U-Net largest-component validation CSV.",
    )
    parser.add_argument(
        "--synapse-bds-lcc",
        default="results/checkpoints/synapse_bdslite_boundary_gate_damped_surface/eval_val_lcc.csv",
        help="Synapse BDS-Lite largest-component validation CSV.",
    )
    parser.add_argument(
        "--synapse-init-from-unet",
        default="results/checkpoints/synapse_bdslite_init_from_unet/eval_val.csv",
        help="Optional Synapse BDS-Lite init-from-UNet validation CSV.",
    )
    parser.add_argument(
        "--synapse-two-stage",
        default="results/checkpoints/synapse_bdslite_two_stage/eval_val.csv",
        help="Optional Synapse BDS-Lite two-stage validation CSV.",
    )
    parser.add_argument(
        "--synapse-hr-rescue",
        default="results/checkpoints/synapse_bdslite_hr_rescue/eval_val.csv",
        help="Optional Synapse BDS-Lite high-resolution rescue validation CSV.",
    )
    parser.add_argument(
        "--output-csv",
        default="results/tables/final_ablation_status_v1.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-md",
        default="docs/final_ablation_report.md",
        help="Output Markdown path.",
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


def read_mean_metrics(path: Path) -> dict[str, float]:
    for row in read_csv(path):
        if row["label"] == "mean":
            return {
                "dsc": float(row["dsc"]),
                "iou": float(row["iou"]),
                "hd95": float(row["hd95"]),
                "assd": float(row["assd"]),
                "boundary_f1": float(row["boundary_f1"]),
            }
    raise ValueError(f"{path} has no mean row")


def pct(value: float) -> str:
    return f"{value:.2f}%"


def delta(candidate: float, baseline: float) -> float:
    return candidate - baseline


def summarize_cost(cost_rows: list[dict[str, str]]) -> dict[str, float]:
    by_variant = {row["variant"]: row for row in cost_rows}
    deploy = by_variant["bdslite_boundary_disabled"]
    retained = by_variant["bdslite_boundary_retained"]
    return {
        "deploy_params_over_unet": float(deploy["active_params_over_baseline_pct"]),
        "deploy_flops_over_unet": float(deploy["flops_over_baseline_pct"]),
        "retained_latency_over_deploy": float(retained["latency_over_disabled_pct"]),
    }


def build_rows(
    month4_rows: list[dict[str, str]],
    cost_rows: list[dict[str, str]],
    synapse_unet: dict[str, float],
    synapse_bds: dict[str, float],
    synapse_unet_lcc: dict[str, float],
    synapse_bds_lcc: dict[str, float],
    synapse_init_from_unet: dict[str, float] | None = None,
    synapse_two_stage: dict[str, float] | None = None,
    synapse_hr_rescue: dict[str, float] | None = None,
) -> list[dict[str, str]]:
    cost = summarize_cost(cost_rows)
    positive_datasets = [
        row["dataset"]
        for row in month4_rows
        if row["claim_status"] in {"positive", "mostly_positive"}
    ]
    mixed_datasets = [row["dataset"] for row in month4_rows if row["claim_status"] == "mixed"]

    lcc_delta_hd95 = delta(synapse_bds_lcc["hd95"], synapse_unet_lcc["hd95"])
    lcc_delta_assd = delta(synapse_bds_lcc["assd"], synapse_unet_lcc["assd"])

    rows = [
        {
            "category": "module_ablation",
            "status": "completed_snapshot",
            "evidence": (
                "Month 4 summary covers baseline, boundary branch/gate, and accepted full variants."
            ),
            "decision": (
                f"Supported on {', '.join(positive_datasets)}; "
                f"mixed on {', '.join(mixed_datasets)}."
            ),
        },
        {
            "category": "loss_ablation",
            "status": "completed_snapshot",
            "evidence": (
                "Surface-only variants beat direct distance-error variants "
                "in accepted Month 4 report."
            ),
            "decision": (
                "Keep signed surface loss; exclude direct SDF distance-error loss "
                "for current method."
            ),
        },
        {
            "category": "inference_cost_ablation",
            "status": "completed",
            "evidence": (
                f"Deploy overhead: params {pct(cost['deploy_params_over_unet'])}, "
                f"FLOPs {pct(cost['deploy_flops_over_unet'])}; retained branch latency "
                f"{pct(cost['retained_latency_over_deploy'])} over deploy path."
            ),
            "decision": "Disable boundary branch at inference.",
        },
        {
            "category": "targeted_synapse_postprocess_ablation",
            "status": "completed_failed",
            "evidence": (
                "Largest-component post-processing gives BDS-Lite minus U-Net deltas of "
                f"HD95 {lcc_delta_hd95:+.3f} and ASSD {lcc_delta_assd:+.3f}."
            ),
            "decision": (
                "Does not fix Synapse distance failure; remote-error cause needs model/loss change."
            ),
        },
        {
            "category": "paper_grade_repeats",
            "status": "deferred",
            "evidence": "Current artifacts are single-seed validation snapshots.",
            "decision": "Run repeated seeds only after Synapse distance metrics improve.",
        },
        {
            "category": "synapse_current_delta",
            "status": "tracked",
            "evidence": (
                "No-postprocess BDS-Lite minus U-Net deltas are "
                f"DSC {delta(synapse_bds['dsc'], synapse_unet['dsc']):+.3f}, "
                f"HD95 {delta(synapse_bds['hd95'], synapse_unet['hd95']):+.3f}, "
                f"ASSD {delta(synapse_bds['assd'], synapse_unet['assd']):+.3f}."
            ),
            "decision": "Treat Synapse as partial support only.",
        },
    ]
    if synapse_init_from_unet is not None:
        rows.append(
            {
                "category": "synapse_init_from_unet",
                "status": "completed_failed",
                "evidence": (
                    "Init-from-UNet BDS-Lite minus U-Net deltas are "
                    f"DSC {delta(synapse_init_from_unet['dsc'], synapse_unet['dsc']):+.3f}, "
                    f"HD95 {delta(synapse_init_from_unet['hd95'], synapse_unet['hd95']):+.3f}, "
                    f"ASSD {delta(synapse_init_from_unet['assd'], synapse_unet['assd']):+.3f}."
                ),
                "decision": (
                    "Improves over the failed full Synapse BDS-Lite distance metrics "
                    "but still misses the U-Net HD95/ASSD threshold."
                ),
            }
        )
    if synapse_two_stage is not None:
        rows.append(
            {
                "category": "synapse_two_stage",
                "status": "completed_failed",
                "evidence": (
                    "Two-stage BDS-Lite minus U-Net deltas are "
                    f"DSC {delta(synapse_two_stage['dsc'], synapse_unet['dsc']):+.3f}, "
                    f"HD95 {delta(synapse_two_stage['hd95'], synapse_unet['hd95']):+.3f}, "
                    f"ASSD {delta(synapse_two_stage['assd'], synapse_unet['assd']):+.3f}."
                ),
                "decision": (
                    "Rejected; encoder freezing worsens overlap and distance metrics on Synapse."
                ),
            }
        )
    if synapse_hr_rescue is not None:
        passed = (
            synapse_hr_rescue["hd95"] < synapse_unet["hd95"]
            and synapse_hr_rescue["assd"] < synapse_unet["assd"]
            and synapse_hr_rescue["dsc"] >= synapse_unet["dsc"] - 0.005
        )
        rows.append(
            {
                "category": "synapse_hr_rescue",
                "status": "completed_accepted" if passed else "completed_failed",
                "evidence": (
                    "High-resolution BDS-Lite minus U-Net deltas are "
                    f"DSC {delta(synapse_hr_rescue['dsc'], synapse_unet['dsc']):+.3f}, "
                    f"HD95 {delta(synapse_hr_rescue['hd95'], synapse_unet['hd95']):+.3f}, "
                    f"ASSD {delta(synapse_hr_rescue['assd'], synapse_unet['assd']):+.3f}."
                ),
                "decision": (
                    "Accepted as qualified Synapse support."
                    if passed
                    else "Rejected; Synapse remains a controlled limitation for this paper."
                ),
            }
        )
    return rows


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        "# Final Ablation Status Report",
        "",
        "Date: 2026-04-27",
        "",
        "This report summarizes the current completed ablation evidence. It is a",
        "single-seed validation snapshot. Synapse is retained as a diagnostic",
        "limitation for the current manuscript scope.",
        "",
        "| Category | Status | Decision |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        lines.append(f"| {row['category']} | {row['status']} | {row['decision']} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The inference-cost claim is ready for the current BDS-Lite variant. The",
            "paper-facing boundary-distance claim should be centered on ISIC2018 and",
            "ACDC because Synapse remains partial support. The largest-component check",
            "shows that simple post-processing is not enough to repair the Synapse",
            "HD95/ASSD regression. The init-from-UNet, two-stage, and high-resolution",
            "Synapse fixes also fail the U-Net HD95/ASSD threshold, so Synapse should",
            "be written as a limitation rather than chased with another shallow tweak.",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(
    month4_summary_path: Path,
    cost_ablation_path: Path,
    synapse_unet_path: Path,
    synapse_bds_path: Path,
    synapse_unet_lcc_path: Path,
    synapse_bds_lcc_path: Path,
    output_csv_path: Path,
    output_md_path: Path,
    synapse_init_from_unet_path: Path | None = None,
    synapse_two_stage_path: Path | None = None,
    synapse_hr_rescue_path: Path | None = None,
) -> list[dict[str, str]]:
    synapse_init = (
        read_mean_metrics(synapse_init_from_unet_path)
        if synapse_init_from_unet_path is not None and synapse_init_from_unet_path.exists()
        else None
    )
    synapse_two_stage = (
        read_mean_metrics(synapse_two_stage_path)
        if synapse_two_stage_path is not None and synapse_two_stage_path.exists()
        else None
    )
    synapse_hr_rescue = (
        read_mean_metrics(synapse_hr_rescue_path)
        if synapse_hr_rescue_path is not None and synapse_hr_rescue_path.exists()
        else None
    )
    rows = build_rows(
        month4_rows=read_csv(month4_summary_path),
        cost_rows=read_csv(cost_ablation_path),
        synapse_unet=read_mean_metrics(synapse_unet_path),
        synapse_bds=read_mean_metrics(synapse_bds_path),
        synapse_unet_lcc=read_mean_metrics(synapse_unet_lcc_path),
        synapse_bds_lcc=read_mean_metrics(synapse_bds_lcc_path),
        synapse_init_from_unet=synapse_init,
        synapse_two_stage=synapse_two_stage,
        synapse_hr_rescue=synapse_hr_rescue,
    )
    write_csv(output_csv_path, ["category", "status", "evidence", "decision"], rows)
    write_markdown(output_md_path, rows)
    return rows


def main() -> None:
    args = parse_args()
    rows = build_report(
        month4_summary_path=Path(args.month4_summary),
        cost_ablation_path=Path(args.cost_ablation),
        synapse_unet_path=Path(args.synapse_unet),
        synapse_bds_path=Path(args.synapse_bds),
        synapse_unet_lcc_path=Path(args.synapse_unet_lcc),
        synapse_bds_lcc_path=Path(args.synapse_bds_lcc),
        output_csv_path=Path(args.output_csv),
        output_md_path=Path(args.output_md),
        synapse_init_from_unet_path=Path(args.synapse_init_from_unet),
        synapse_two_stage_path=Path(args.synapse_two_stage),
        synapse_hr_rescue_path=Path(args.synapse_hr_rescue),
    )
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_md}")
    print(f"Summarized {len(rows)} ablation categories")


if __name__ == "__main__":
    main()
