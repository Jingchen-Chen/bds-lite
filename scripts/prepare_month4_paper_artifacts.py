#!/usr/bin/env python
"""Prepare paper-facing Month 4 ablation tables from result CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Month 4 paper tables.")
    parser.add_argument(
        "--cross-dataset",
        default="results/tables/cross_dataset_bdslite_status.csv",
        help="Cross-dataset BDS-Lite status CSV.",
    )
    parser.add_argument(
        "--cost-ablation",
        default="results/tables/resource_profile_isic2018_boundary_branch_ablation.csv",
        help="Boundary-branch cost ablation CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="paper/tables",
        help="Directory for paper-facing tables.",
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


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def fmt_delta(value: float, digits: int = 3, lower_is_better: bool = False) -> str:
    sign = "+" if value > 0 else ""
    marker = "improves" if (value < 0 if lower_is_better else value > 0) else "worse"
    if abs(value) < 5e-4:
        marker = "tie"
    return f"{sign}{value:.{digits}f} ({marker})"


def prepare_cross_dataset_table(
    rows: list[dict[str, str]],
) -> tuple[list[str], list[dict[str, str]]]:
    fieldnames = [
        "Dataset",
        "BDS-Lite variant",
        "U-Net DSC",
        "BDS-Lite DSC",
        "Delta DSC",
        "U-Net HD95",
        "BDS-Lite HD95",
        "Delta HD95",
        "U-Net ASSD",
        "BDS-Lite ASSD",
        "Delta ASSD",
        "U-Net Boundary F1",
        "BDS-Lite Boundary F1",
        "Delta Boundary F1",
        "Claim status",
    ]
    output_rows = []
    for row in rows:
        output_rows.append(
            {
                "Dataset": row["dataset"],
                "BDS-Lite variant": row["method_variant"],
                "U-Net DSC": fmt(as_float(row, "baseline_dsc")),
                "BDS-Lite DSC": fmt(as_float(row, "bds_dsc")),
                "Delta DSC": fmt_delta(as_float(row, "delta_dsc")),
                "U-Net HD95": fmt(as_float(row, "baseline_hd95")),
                "BDS-Lite HD95": fmt(as_float(row, "bds_hd95")),
                "Delta HD95": fmt_delta(as_float(row, "delta_hd95"), lower_is_better=True),
                "U-Net ASSD": fmt(as_float(row, "baseline_assd")),
                "BDS-Lite ASSD": fmt(as_float(row, "bds_assd")),
                "Delta ASSD": fmt_delta(as_float(row, "delta_assd"), lower_is_better=True),
                "U-Net Boundary F1": fmt(as_float(row, "baseline_boundary_f1")),
                "BDS-Lite Boundary F1": fmt(as_float(row, "bds_boundary_f1")),
                "Delta Boundary F1": fmt_delta(as_float(row, "delta_boundary_f1")),
                "Claim status": row["claim_status"],
            }
        )
    return fieldnames, output_rows


def prepare_cost_table(rows: list[dict[str, str]]) -> tuple[list[str], list[dict[str, str]]]:
    fieldnames = [
        "Variant",
        "Boundary branch",
        "Active params (M)",
        "FLOPs (G)",
        "FPS",
        "Latency (ms)",
        "Peak memory (MB)",
        "Params vs U-Net",
        "FLOPs vs U-Net",
        "Latency vs disabled",
    ]
    labels = {
        "unet_baseline": "U-Net baseline",
        "bdslite_boundary_disabled": "BDS-Lite deploy",
        "bdslite_boundary_retained": "BDS-Lite retained branch",
    }
    output_rows = []
    for row in rows:
        params_delta = row["active_params_over_baseline_pct"]
        flops_delta = row["flops_over_baseline_pct"]
        latency_delta = row["latency_over_disabled_pct"]
        output_rows.append(
            {
                "Variant": labels.get(row["variant"], row["variant"]),
                "Boundary branch": "retained" if row["boundary_branch"] == "True" else "disabled",
                "Active params (M)": fmt(float(row["active_params"]) / 1_000_000),
                "FLOPs (G)": fmt(float(row["flops"]) / 1_000_000_000),
                "FPS": fmt(float(row["fps"]), digits=2),
                "Latency (ms)": fmt(float(row["latency_ms"]), digits=3),
                "Peak memory (MB)": fmt(float(row["peak_memory_mb"]), digits=2),
                "Params vs U-Net": "" if not params_delta else f"+{float(params_delta):.2f}%",
                "FLOPs vs U-Net": "" if not flops_delta else f"+{float(flops_delta):.2f}%",
                "Latency vs disabled": "" if not latency_delta else f"+{float(latency_delta):.2f}%",
            }
        )
    return fieldnames, output_rows


def write_boundary_cost_svg(rows: list[dict[str, str]], output_path: Path) -> None:
    labels = {
        "unet_baseline": "U-Net",
        "bdslite_boundary_disabled": "BDS deploy",
        "bdslite_boundary_retained": "BDS retained",
    }
    colors = {
        "unet_baseline": "#4b5563",
        "bdslite_boundary_disabled": "#2563eb",
        "bdslite_boundary_retained": "#dc2626",
    }
    metrics = [
        ("active_params", "Params (M)", 1_000_000.0),
        ("flops", "FLOPs (G)", 1_000_000_000.0),
        ("latency_ms", "Latency (ms)", 1.0),
    ]
    values = {
        metric: [float(row[metric]) / scale for row in rows] for metric, _label, scale in metrics
    }
    maxima = {metric: max(metric_values) for metric, metric_values in values.items()}

    width = 920
    height = 360
    margin_left = 150
    group_gap = 88
    bar_height = 20
    bar_gap = 10
    chart_width = 620
    y = 72
    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="34" font-family="Arial, sans-serif" font-size="20" '
        'font-weight="700" fill="#111827">Boundary branch inference-cost ablation</text>',
        '<text x="24" y="56" font-family="Arial, sans-serif" font-size="12" '
        'fill="#4b5563">ISIC2018, batch size 1, CUDA profile. '
        "Lower FLOPs and latency are better.</text>",
    ]

    for metric, metric_label, _scale in metrics:
        svg.append(
            f'<text x="24" y="{y + 15}" font-family="Arial, sans-serif" '
            f'font-size="13" font-weight="700" fill="#111827">{metric_label}</text>'
        )
        for index, row in enumerate(rows):
            variant = row["variant"]
            value = values[metric][index]
            bar_width = chart_width * value / maxima[metric]
            bar_y = y + index * (bar_height + bar_gap)
            value_text = f"{value:.2f}" if metric == "flops" else f"{value:.3f}"
            svg.extend(
                [
                    f'<text x="{margin_left - 10}" y="{bar_y + 15}" '
                    'font-family="Arial, sans-serif" font-size="12" text-anchor="end" '
                    f'fill="#374151">{labels.get(variant, variant)}</text>',
                    f'<rect x="{margin_left}" y="{bar_y}" width="{bar_width:.1f}" '
                    f'height="{bar_height}" rx="2" fill="{colors.get(variant, "#111827")}"/>',
                    f'<text x="{margin_left + bar_width + 8:.1f}" y="{bar_y + 15}" '
                    'font-family="Arial, sans-serif" font-size="12" '
                    f'fill="#111827">{value_text}</text>',
                ]
            )
        y += group_gap

    svg.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    cross_source_rows = read_csv(Path(args.cross_dataset))
    cost_source_rows = read_csv(Path(args.cost_ablation))
    cross_fieldnames, cross_rows = prepare_cross_dataset_table(cross_source_rows)
    cost_fieldnames, cost_rows = prepare_cost_table(cost_source_rows)

    cross_output = output_dir / "month4_cross_dataset_ablation_paper.csv"
    cost_output = output_dir / "month4_boundary_branch_cost_paper.csv"
    figure_output = Path("paper/figures/month4_boundary_branch_cost.svg")
    write_csv(cross_output, cross_fieldnames, cross_rows)
    write_csv(cost_output, cost_fieldnames, cost_rows)
    write_boundary_cost_svg(cost_source_rows, figure_output)
    print(f"Wrote {cross_output}")
    print(f"Wrote {cost_output}")
    print(f"Wrote {figure_output}")


if __name__ == "__main__":
    main()
