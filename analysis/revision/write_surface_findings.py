"""Generate descriptive CDF tables and qualified interpretation from actual CSVs."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    directory = ROOT / "revision/surface_cdf"
    summaries = defaultdict(list)
    for row in csv.DictReader((directory / "summary.csv").open()):
        summaries[row["dataset"], row["family"]].append(row)
    thresholds = defaultdict(list)
    for row in csv.DictReader((directory / "threshold_cdf.csv").open()):
        thresholds[row["dataset"], row["family"], int(row["threshold_px"])].append(
            float(row["equal_image_class_cdf"])
        )
    table, tex = (
        [],
        [
            r"\begin{tabular}{llrrr}",
            r"Dataset & Family & CDF $\leq2$ & Tail $>20$ & Spurious tail share \\",
            r"\hline",
        ],
    )
    for (dataset, family), rows in sorted(summaries.items()):
        near = float(np.mean(thresholds[dataset, family, 2]))
        tail = 1 - float(np.mean(thresholds[dataset, family, 20]))
        spurious = float(np.mean([float(r["spurious_share_pred_tail20"]) for r in rows]))
        table.append(
            dict(
                dataset=dataset,
                family=family,
                cdf_le2=near,
                tail_gt20=tail,
                spurious_share_pred_tail20=spurious,
                valid_rows_per_seed="/".join(r["valid_image_classes"] for r in rows),
                one_empty_per_seed="/".join(r["one_empty"] for r in rows),
                seeds="/".join(r["seed"] for r in rows),
            )
        )
        family_tex = family.replace("_", r"\_")
        tex.append(f"{dataset} & {family_tex} & {near:.4f} & {tail:.4f} & {spurious:.4f} " + r"\\")
    tex.append(r"\end{tabular}")
    with (directory / "family_summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    (ROOT / "revision/tables/surface_cdf_summary.tex").write_text("\n".join(tex) + "\n")
    validation = json.loads((ROOT / "revision/validation/surface_cdf.json").read_text())
    observed = {(r["dataset"], r["family"]): r for r in table}
    pairs = [
        ("isic2018", "U", "F"),
        ("acdc", "main_p1b_unet", "main_p1b_full"),
        ("synapse", "main_p1b_unet", "main_p1b_full"),
    ]
    if not all(
        observed[d, f]["cdf_le2"] > observed[d, u]["cdf_le2"]
        and observed[d, f]["tail_gt20"] > observed[d, u]["tail_gt20"]
        for d, u, f in pairs
    ):
        raise AssertionError("Narrative signs differ; revise interpretation")
    text = [
        "# Surface-error distribution findings (descriptive)",
        "",
        (
            f"{validation['cdf_cohorts']} saved-test cohorts; "
            f"{validation['image_class_rows']} image/class rows; "
            f"{validation['valid_surfaces']} valid surfaces. Saved HD95/missingness joins pass; "
            f"maximum rounding delta={validation['max_saved_hd95_delta']:.3g}px. "
            f"Independent KDTree checks: {len(validation['independent_kdtree_checks'])}. "
            "Empties remain separate, without invented distances."
        ),
        "",
        "| Dataset | Family | CDF <=2px | Tail >20px | Spurious share of predicted tail |",
        "|---|---|---:|---:|---:|",
    ]
    for row in table:
        text.append(
            f"| {row['dataset']} | {row['family']} | {row['cdf_le2']:.4f} | "
            f"{row['tail_gt20']:.4f} | {row['spurious_share_pred_tail20']:.4f} |"
        )
    text += [
        "",
        "CDFs give each eligible image/class equal weight within a seed, then average seeds. "
        "Point-pooled CDFs are also saved. Eligibility differs between methods when masks "
        "are empty; these curves are descriptive rather than matched-unit effect estimates. "
        "Attention uses seed2026; legacy GSL retains its separate provenance caveat.",
        "",
        "For the main full recipe versus U-Net, all three datasets have more probability "
        "within2px alongside more tail probability beyond20px in these conditional curves. "
        "This mixture does not support describing every change as a uniform contour shift. "
        "ACDC/Synapse remote predicted-surface points frequently belong to components with "
        "no GT intersection. The spurious-share column concerns only the predicted-to-GT "
        "direction; it is not a fraction of all bidirectional errors or causal attribution "
        "of an HD95 difference. ISIC has a different tail pattern. Never generalize this "
        "association to an established mechanism.",
        "",
        "The20px cutoff was chosen for this exploratory descriptive analysis; it is not "
        "a preregistered clinical threshold. All distances use the processed224 grid. "
        "Numeric Synapse classes remain unrelabelled. No physical or clinical conclusion "
        "is inferred. Sources: surface_cdf/*.csv and validation/surface_cdf.json.",
    ]
    (ROOT / "revision/SURFACE_ERROR_FINDINGS.md").write_text("\n".join(text) + "\n")
    (directory / "README.md").write_text(
        "Reproduce with surface_distance_cdf.py (refuses existing output), then "
        "plot_surface_cdf.py and validate_surface_cdf.py. Table/prose generation "
        "uses make_tables.py. --run-list accepts dataset/split/family/seed/classes/path "
        "for other saved methods. Existing main/reference test cohorts only; gate "
        "interventions and future missing comparators are separate. The corrected "
        "figure uses right-continuous steps; initial linear renders are preserved. "
        "Per-class empty cases are explicit in per_image_class.csv and summary.csv. "
        "Original GT provenance is in audit/inventory.csv and sources/isic_acquisition.json.\n"
    )
    (
        ROOT / "revision/manuscript/surface_errors_draft.tex"
    ).write_text(r"""% Generated evidence fragment; author integration and compilation pending.
\paragraph{Surface-distance distributions.}
The descriptive processed-grid CDFs show a mixture of near-contour and distant
errors rather than a uniform displacement. Component-tail associations are
conditional on valid masks and do not establish a causal mechanism.
\begin{figure}[ht]
\centering
\includegraphics[width=\linewidth]{../surface_cdf/surface_distance_cdf.pdf}
\caption{Right-continuous distance CDFs. Each eligible image/class has equal
weight within a seed; curves average seeds. Empty-mask counts are reported
separately. Legacy GSL and Attention seed2026 are separate reference families.
Distances are processed-grid pixels. Curves are descriptive, not independent
pixel-level statistical comparisons.}
\end{figure}
""")
    print("Generated CDF evidence fragment and", len(table), "family summaries")


if __name__ == "__main__":
    main()
