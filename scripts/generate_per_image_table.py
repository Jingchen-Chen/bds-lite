#!/usr/bin/env python
"""Generate manuscript_v2/tables/per_image_paired_wilcoxon.tex from the CSV.

Reads
``results/final_manuscript_audit_v2/per_image_multiseed/per_image_paired_wilcoxon.csv``
and emits a LaTeX table containing the three pairwise comparisons:
BDS-Lite full vs vanilla U-Net, BDS-Lite full vs EGE-UNet, and EGE-UNet
vs vanilla U-Net.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = (
    ROOT
    / "results"
    / "final_manuscript_audit_v2"
    / "per_image_multiseed"
    / "per_image_paired_wilcoxon.csv"
)
TABLE_PATH = ROOT / "manuscript_v2" / "tables" / "per_image_paired_wilcoxon.tex"

DATASET_LABEL = {"isic2018": "ISIC2018 (val)", "acdc": "ACDC (test)", "synapse": "Synapse (test)"}
METRIC_LABEL = {
    "dsc": "Dice",
    "iou": "IoU",
    "hd95": "HD95 (px)",
    "assd": "ASSD (px)",
    "boundary_f1": "Boundary F1",
}
COMPARISON_LABEL = {
    "bds_lite_full_vs_unet": ("U-Net", "BDS-Lite full"),
    "bds_lite_full_vs_ege_unet": ("EGE-UNet", "BDS-Lite full"),
    "ege_unet_vs_unet": ("U-Net", "EGE-UNet"),
}

# Order of comparisons (sections) and metrics (rows within section)
COMPARISONS = ("bds_lite_full_vs_unet", "bds_lite_full_vs_ege_unet", "ege_unet_vs_unet")
METRICS = ("dsc", "iou", "boundary_f1", "hd95", "assd")
DATASETS = ("isic2018", "acdc", "synapse")


def fmt_diff(value: float) -> str:
    return f"${value:+.4f}$" if abs(value) < 1.0 else f"${value:+.3f}$"


def fmt_p(value: float) -> str:
    if value >= 0.001:
        return f"{value:.3g}"
    # Scientific notation
    mantissa, exp = f"{value:.1e}".split("e")
    return f"${mantissa}\\!\\times\\!10^{{{int(exp)}}}$"


def decision_word(decision: str, diff: float, metric: str) -> str:
    if decision != "significant":
        return "ns"
    higher_is_better = metric in ("dsc", "iou", "boundary_f1")
    favours_b = (diff > 0) == higher_is_better
    word = r"\textbf{better}" if favours_b else r"\textbf{worse}"
    # Flag statistically significant but substantively negligible overlap effects.
    if metric in ("dsc", "iou", "boundary_f1") and abs(diff) < 1e-3:
        word += r"$^{\dagger}$"
    return word


def main() -> None:
    rows = list(csv.DictReader(CSV_PATH.open()))
    by_cmp_ds = {(r["comparison"], r["dataset"], r["metric"]): r for r in rows}

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Multi-seed per-image paired Wilcoxon analysis across three "
        r"pairwise comparisons: BDS-Lite full vs vanilla U-Net, BDS-Lite full "
        r"vs EGE-UNet \cite{ruan2023egeunet}, and EGE-UNet vs vanilla U-Net. "
        r"For each image, the three-seed mean per-image metric is computed; "
        r"paired Wilcoxon then compares the candidate to the reference. "
        r"Holm correction is applied within (dataset, comparison) over the "
        r"five metrics. Means and medians are computed over the paired "
        r"subset (the same $n$ pairs used by the test). HD95 and ASSD are in "
        r"pixels because spacing metadata is not stored in the processed "
        r"sample arrays. ISIC2018 uses the validation split (image-disjoint "
        r"from train); ACDC and Synapse use their test splits. Decisions "
        r"refer to the candidate (column 4): \textbf{better} means the "
        r"candidate is Holm-significantly better; \textbf{worse} means "
        r"Holm-significantly worse; ns is not significant at $\alpha=0.05$. A "
        r"$\dagger$ marks a Holm-significant overlap result whose absolute "
        r"mean difference is below $10^{-3}$ and is therefore substantively "
        r"tied despite the small $p$-value. "
        r"Source artifact: \texttt{results/final\_manuscript\_audit\_v2/"
        r"per\_image\_multiseed/per\_image\_paired\_wilcoxon.csv}.}"
    )
    lines.append(r"\label{tab:per-image-paired-wilcoxon}")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{llrrrll}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Metric & $n$ & Ref mean & Cand mean & Holm $p$ & Decision \\")

    for cmp_key in COMPARISONS:
        ref_label, cand_label = COMPARISON_LABEL[cmp_key]
        lines.append(r"\midrule")
        lines.append(
            r"\multicolumn{7}{l}{\textbf{"
            + f"{cand_label}"
            + r"} (candidate) vs \textbf{"
            + f"{ref_label}"
            + r"} (reference)} \\"
        )
        for ds in DATASETS:
            for m in METRICS:
                key = (cmp_key, ds, m)
                if key not in by_cmp_ds:
                    continue
                r = by_cmp_ds[key]
                holm_p = float(r["p_value_holm"])
                diff = float(r["mean_diff_b_minus_a"])
                row_cells = [
                    DATASET_LABEL[ds],
                    METRIC_LABEL[m],
                    r["n_pairs"],
                    f"{float(r['mean_a']):.4f}"
                    if m not in ("hd95", "assd")
                    else f"{float(r['mean_a']):.3f}",
                    f"{float(r['mean_b']):.4f}"
                    if m not in ("hd95", "assd")
                    else f"{float(r['mean_b']):.3f}",
                    fmt_p(holm_p),
                    decision_word(r["decision_alpha_0_05"], diff, m),
                ]
                lines.append(" & ".join(str(c) for c in row_cells) + r" \\")
            lines.append(r"\addlinespace[2pt]")
        # Trim trailing addlinespace before next midrule
        if lines[-1] == r"\addlinespace[2pt]":
            lines.pop()
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    TABLE_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {TABLE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
