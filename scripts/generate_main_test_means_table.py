#!/usr/bin/env python
"""Generate manuscript_v2/tables/main_test_means.tex from the seed JSONs.

Reads ``outputs/evaluations/<dataset>_<method>_seed<n>_test.json`` for the
present methods and writes a LaTeX table with three-seed means under the
patched evaluation pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "outputs" / "evaluations"
TABLE_PATH = ROOT / "manuscript_v2" / "tables" / "main_test_means.tex"

DATASETS = (
    ("isic2018", "ISIC2018 (val)"),
    ("acdc", "ACDC (test)"),
    ("synapse", "Synapse (test)"),
)
METHODS = (
    ("unet", "U-Net"),
    ("ege_unet", "EGE-UNet"),
    ("bds_lite_full", "BDS-Lite full"),
)
SEEDS = (1, 2, 3)
METRICS = ("dsc", "iou", "hd95", "assd", "boundary_f1")


def load_mean(dataset: str, method: str, metric: str) -> float | None:
    vals = []
    for s in SEEDS:
        p = EVAL / f"{dataset}_{method}_seed{s}_test.json"
        if not p.exists():
            return None
        j = json.loads(p.read_text())
        vals.append(j["metrics"][metric]["foreground_macro_mean"])
    return float(np.mean(vals))


def fmt(x: float | None, digits: int) -> str:
    if x is None:
        return r"\textemdash"
    if digits == 4:
        return f"{x:.4f}"
    return f"{x:.3f}"


def main() -> None:
    rows: list[str] = []
    rows.append(r"\begin{table}[htbp]")
    rows.append(r"\centering")
    rows.append(
        r"\caption{Three-seed evaluation means for compact U-Net, EGE-Unet "
        r"\cite{ruan2023egeunet}, and BDS-Lite full, recomputed under the "
        r"patched ``boundary\_f1'' (disjoint non-empty boundaries return $0$ "
        r"instead of $1$) on the current locked splits. ISIC2018 is evaluated "
        r"on the validation split because the historical test partition was "
        r"not re-created by the current converter. ACDC and Synapse are "
        r"evaluated on their test splits. Distances are in pixels because "
        r"spacing metadata is not recorded in the processed sample arrays. "
        r"Dice, IoU, and Boundary F1 are higher-is-better; HD95 and ASSD "
        r"are lower-is-better. No comparison is Holm-significant at the seed "
        r"level under n=3 (smallest achievable two-sided exact Wilcoxon "
        r"$p$-value is $0.25$). Source artifact: "
        r"\texttt{results/final\_manuscript\_audit\_v2/recomputed\_main\_test\_means.csv}.}"
    )
    rows.append(r"\label{tab:main-test-means}")
    rows.append(r"\setlength{\tabcolsep}{4pt}")
    rows.append(r"\begin{tabular}{llccccc}")
    rows.append(r"\toprule")
    rows.append(r"Dataset & Method & Dice & IoU & HD95 (px) & ASSD (px) & Boundary F1 \\")
    rows.append(r"\midrule")
    for ds_name, ds_label in DATASETS:
        first = True
        for m_name, m_label in METHODS:
            vals = {met: load_mean(ds_name, m_name, met) for met in METRICS}
            if all(v is None for v in vals.values()):
                continue
            row = [ds_label if first else "", m_label]
            first = False
            row.append(fmt(vals["dsc"], 4))
            row.append(fmt(vals["iou"], 4))
            row.append(fmt(vals["hd95"], 3))
            row.append(fmt(vals["assd"], 3))
            row.append(fmt(vals["boundary_f1"], 4))
            rows.append(" & ".join(row) + r" \\")
        rows.append(r"\addlinespace")
    # remove last addlinespace
    if rows[-1] == r"\addlinespace":
        rows.pop()
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    rows.append(r"\end{table}")

    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text("\n".join(rows) + "\n")
    print(f"Wrote {TABLE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
