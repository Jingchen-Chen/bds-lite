#!/usr/bin/env python
"""Generate the main paper comparison table (baselines vs BDS-Lite) from eval CSVs.

Outputs a rounded CSV and a Markdown table suitable for paper draft review.

Usage:
    python scripts/prepare_paper_main_table.py
"""

from __future__ import annotations

import csv
from pathlib import Path

CHECKPOINTS_DIR = Path("results/checkpoints")
OUTPUT_DIR = Path("paper/tables")

# Each entry: (display_name, exp_name, dataset, split)
TABLE_ENTRIES = [
    # --- ISIC2018 ---
    ("U-Net", "isic2018_unet_baseline", "ISIC2018", "val"),
    ("Attention U-Net", "isic2018_attention_unet_baseline", "ISIC2018", "val"),
    ("MobileViT-UNet", "isic2018_mobilevit_unet", "ISIC2018", "val"),
    ("BDS-Lite (boundary+distill)", "isic2018_bdslite_unet", "ISIC2018", "val"),
    (
        "BDS-Lite (gate+surface)",
        "isic2018_bdslite_boundary_gate_damped_surface",
        "ISIC2018",
        "val",
    ),
    # --- ACDC ---
    ("U-Net", "acdc_unet_baseline", "ACDC", "val"),
    ("Attention U-Net", "acdc_attention_unet_baseline", "ACDC", "val"),
    ("BDS-Lite (boundary+distill)", "acdc_bdslite_unet", "ACDC", "val"),
    (
        "BDS-Lite (gate+surface+cw)",
        "acdc_bdslite_boundary_gate_damped_surface_class13_mild",
        "ACDC",
        "val",
    ),
    # --- Synapse ---
    ("U-Net", "synapse_unet_baseline", "Synapse", "val"),
    ("Attention U-Net", "synapse_attention_unet_baseline", "Synapse", "val"),
    (
        "BDS-Lite (gate+surface)",
        "synapse_bdslite_boundary_gate_damped_surface",
        "Synapse",
        "val",
    ),
    ("BDS-Lite (init from U-Net)", "synapse_bdslite_init_from_unet", "Synapse", "val"),
]

METRICS = ["dsc", "iou", "hd95", "assd", "boundary_f1"]
METRIC_LABELS = {"dsc": "DSC", "iou": "IoU", "hd95": "HD95", "assd": "ASSD", "boundary_f1": "BF1"}


def _read_mean(csv_path: Path) -> dict[str, float] | None:
    if not csv_path.exists():
        return None
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    mean_rows = [r for r in rows if r.get("label", "") == "mean"]
    row = mean_rows[0] if mean_rows else rows[-1]
    return {m: float(row[m]) for m in METRICS if m in row}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    out_rows: list[dict[str, str]] = []
    md_lines: list[str] = []
    md_lines.append("| Dataset | Method | DSC | IoU | HD95 | ASSD | BF1 |")
    md_lines.append("|---------|--------|-----|-----|------|------|-----|")

    for display_name, exp_name, dataset, split in TABLE_ENTRIES:
        csv_path = CHECKPOINTS_DIR / exp_name / f"eval_{split}.csv"
        data = _read_mean(csv_path)
        if data is None:
            row = {"dataset": dataset, "method": display_name, "experiment": exp_name}
            for m in METRICS:
                row[m] = "n/a"
        else:
            row = {"dataset": dataset, "method": display_name, "experiment": exp_name}
            for m in METRICS:
                precision = 2 if m in ("hd95", "assd") else 4
                row[m] = f"{data[m]:.{precision}f}"

        out_rows.append(row)
        vals = [row[m] for m in METRICS]
        md_lines.append(f"| {dataset} | {display_name} | {' | '.join(vals)} |")

    csv_out = OUTPUT_DIR / "paper_main_table.csv"
    fieldnames = ["dataset", "method", "experiment", *METRICS]
    with csv_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    md_out = OUTPUT_DIR / "paper_main_table.md"
    md_out.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote {csv_out}")
    print(f"Wrote {md_out}")
    print()
    print("\n".join(md_lines))


if __name__ == "__main__":
    main()
