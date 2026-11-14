"""Build descriptive absolute-score comparison from existing test CSVs only.

No inference, training, model selection or hypothesis testing. Three seed means
are independently aggregated from the original class-level files, reconciled to
the frozen method summary, and recorded with paths/hashes for audit. Published
values retain their different populations, aggregation and distance definitions.
Usage: PYTHONPATH=src:. .venv/bin/python analysis/revision/make_comparison_table.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from analysis.revision.family_statistics import per_case_path

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "revision/manuscript/generated"
VALIDATION = ROOT / "revision/validation"
DATASETS = {"isic2018": "ISIC2018", "acdc": "ACDC", "synapse": "Synapse"}
METHODS = {
    "M0": "U-Net",
    "M2": "Decoder-only",
    "M6": "BDS-Lite (full)",
    "GSL": "U-Net + GSL",
    "C1": "U-Net + Boundary Loss",
    "C3": "Attention U-Net",
    "C4": "LB-UNet",
}
METRICS = ("dsc", "boundary_f1", "hd95")
# Swin-Unet is cited in its ECCV 2022 Workshops version (Cao2022SwinUnet); the values were
# verified against the arXiv v1 text of the same paper (revision/lit_numbers.csv).
PUBLISHED = [
    ("ISIC2018", "EGE-UNet", "Ruan2023EGEUNet", "2307.08473v1.pdf"),
    ("ISIC2018", "LB-UNet", "Xu2024LBUNet", "lbunet_miccai2024.pdf"),
    ("ACDC", "TransUNet", "Chen2021TransUNetArxiv", "2102.04306v1.pdf"),
    ("ACDC", "Swin-Unet", "Cao2022SwinUnet", "2105.05537v1.pdf"),
    ("Synapse", "TransUNet", "Chen2021TransUNetArxiv", "2102.04306v1.pdf"),
    ("Synapse", "Swin-Unet", "Cao2022SwinUnet", "2105.05537v1.pdf"),
]


def read(path: Path) -> list[dict]:
    with path.open() as stream:
        return list(csv.DictReader(stream))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_means(path: Path) -> dict[str, tuple[float, int]]:
    """Foreground-class macro per image/slice, then mean of defined samples."""
    samples = defaultdict(lambda: defaultdict(list))
    seen = set()
    for row in read(path):
        key = (row["id"], row.get("label", "1"))
        if key in seen:
            raise ValueError(f"duplicate image/class in {path}: {key}")
        seen.add(key)
        both_empty = all(
            row.get(k, "").lower() == "true" for k in ("empty_prediction", "empty_target")
        )
        for metric in METRICS:
            value = row.get(metric, "")
            number = math.nan if both_empty or value in ("", "None", "nan") else float(value)
            if math.isfinite(number):
                if number < 0 or (metric != "hd95" and number > 1):
                    raise ValueError(f"invalid absolute {metric} in {path}: {number}")
                samples[row["id"]][metric].append(number)
    out = {}
    for metric in METRICS:
        values = [statistics.mean(v[metric]) for v in samples.values() if v[metric]]
        if not values:
            raise ValueError(f"no defined {metric} in {path}")
        out[metric] = (statistics.mean(values), len(values))
    return out


def build() -> dict:
    frozen = {
        (r["dataset"], r["method"]): r
        for r in read(ROOT / "revision/tables/families/method_summary.csv")
    }
    provenance, matched, errors = [], [], []
    for dataset in DATASETS:
        for method in METHODS:
            values = defaultdict(list)
            for seed in (1, 2, 3):
                path = per_case_path(method, dataset, seed)
                if path is None:
                    raise FileNotFoundError(
                        f"missing original test CSV: {dataset} {method} seed {seed}"
                    )
                raw = run_means(path)
                stored = frozen[(dataset, method)]
                sha = digest(path)
                for metric in METRICS:
                    value, n = raw[metric]
                    expected = json.loads(stored[f"{metric}_seeds"])[seed - 1]
                    errors.append(abs(value - expected))
                    values[metric].append(value)
                    provenance.append(
                        dict(
                            dataset=dataset,
                            method=method,
                            model=METHODS[method],
                            seed=seed,
                            metric=metric,
                            absolute_run_mean=value,
                            defined_images_or_slices=n,
                            frozen_run_mean=expected,
                            absolute_error=abs(value - expected),
                            source=path.relative_to(ROOT).as_posix(),
                            sha256=sha,
                        )
                    )
            row = dict(dataset=dataset, method=method, model=METHODS[method])
            for metric in METRICS:
                mean, sd = statistics.mean(values[metric]), statistics.stdev(values[metric])
                errors.extend(
                    [
                        abs(mean - float(stored[f"{metric}_mean"])),
                        abs(sd - float(stored[f"{metric}_sd"])),
                    ]
                )
                row[f"{metric}_mean"], row[f"{metric}_sd"] = mean, sd
            matched.append(row)
    if max(errors) > 1e-10:
        raise ValueError(
            f"absolute scores do not reproduce frozen summary: max error {max(errors)}"
        )

    literature = read(ROOT / "revision/lit_numbers.csv")
    published, receipts = [], []
    for dataset, model, citation, filename in PUBLISHED:
        row = dict(dataset=dataset, model=model, citation=citation)
        for metric in ("DSC", "HD95"):
            candidates = [
                r
                for r in literature
                if r["dataset"] == dataset and r["method"] == model and r["metric"] == metric
            ]
            if not candidates:
                if metric == "DSC":
                    raise ValueError(f"missing published DSC: {dataset} {model}")
                row[metric] = None
                continue
            if len(candidates) != 1:
                raise ValueError(f"ambiguous published source: {dataset} {model} {metric}")
            source = candidates[0]
            pdf = ROOT / "revision/sources/literature" / filename
            if digest(pdf) != source["local_pdf_sha256"]:
                raise ValueError(f"published source hash mismatch: {pdf}")
            # DSC alone is converted from percent; distances remain as reported.
            row[metric] = float(source["value"]) / (100 if metric == "DSC" else 1)
            receipts.append(
                dict(
                    source,
                    local_pdf=pdf.relative_to(ROOT).as_posix(),
                    paper_metric_label="HD" if metric == "HD95" else metric,
                )
            )
        published.append(row)

    # Use the same six-model reporting scope on every dataset, irrespective of
    # results. Retain all adapted LB-UNet rows in the provenance output and S7.
    main_rows = [r for r in matched if r["method"] != "C4"]
    best_by, rankings = {}, []
    for dataset in DATASETS:
        candidates = [r for r in main_rows if r["dataset"] == dataset]
        for metric in METRICS:
            best = (min if metric == "hd95" else max)(r[f"{metric}_mean"] for r in candidates)
            winners = [
                r["method"]
                for r in candidates
                if math.isclose(r[f"{metric}_mean"], best, rel_tol=0, abs_tol=1e-12)
            ]
            best_by[(dataset, metric)] = winners
            rankings.append(dict(dataset=dataset, metric=metric, mean=best, methods=winners))

    def scores(row: dict, bold: bool = False) -> list[str]:
        cells = []
        for metric in METRICS:
            precision = 2 if metric == "hd95" else 4
            mean = f"{row[f'{metric}_mean']:.{precision}f}"
            if bold and row["method"] in best_by[(row["dataset"], metric)]:
                mean = r"\mathbf{" + mean + "}"
            cells.append(f"${mean} \\pm {row[f'{metric}_sd']:.{precision}f}$")
        return cells

    lines = [
        "% descriptive absolute scores; generated by analysis/revision/make_comparison_table.py",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"\multicolumn{5}{l}{\textbf{A. Matched protocol: three runs, mean $\pm$ SD}} \\",
        r"\midrule",
        r"Dataset & Model & DSC $\uparrow$ & Boundary F1 $\uparrow$ & HD95 (px) $\downarrow$ \\",
        r"\midrule",
    ]
    for dataset in DATASETS:
        for i, row in enumerate(r for r in main_rows if r["dataset"] == dataset):
            model = row["model"] + (" (adapted)" if row["method"] == "C4" else "")
            lines.append(
                f"{DATASETS[dataset] if i == 0 else ''} & {model} & "
                + " & ".join(scores(row, bold=True))
                + r" \\"
            )
        lines.append(r"\addlinespace")
    lines.extend(
        [
            r"\midrule",
            r"\multicolumn{5}{l}{\textbf{B. Published references: different protocols}} \\",
            r"\midrule",
            (
                "Dataset & Model & DSC $\\uparrow$ & Boundary F1 $\\uparrow$ & "
                "HD$^\\dagger$ $\\downarrow$ \\\\"
            ),
            r"\midrule",
        ]
    )
    last = None
    for row in published:
        hd = "--" if row["HD95"] is None else f"{row['HD95']:.2f}"
        label = row["dataset"] if row["dataset"] != last else ""
        lines.append(
            f"{label} & {row['model']} \\citep{{{row['citation']}}} & {row['DSC']:.4f} & -- & {hd}"
            + r" \\"
        )
        last = row["dataset"]
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    adaptations, multiclass = [], []
    for row in matched:
        if row["method"] == "C4":
            values = scores(row)
            text = (
                f"{DATASETS[row['dataset']]}: DSC {values[0]}, Boundary F1 {values[1]}, "
                f"HD95 {values[2]}\\,px."
            )
            adaptations.append(text)
            if row["dataset"] != "isic2018":
                multiclass.append(text)
    (GEN / "lb_adapted_scores.tex").write_text("\n".join(adaptations) + "\n")
    (GEN / "lb_multiclass_scores.tex").write_text("\n".join(multiclass) + "\n")
    (GEN / "tab_lit_main.tex").write_text("\n".join(lines) + "\n")
    with (VALIDATION / "comparison_absolute_scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(provenance[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(provenance)
    report = dict(
        matched_models=matched,
        main_matched_rows=len(main_rows),
        numerical_best=rankings,
        bold_policy=(
            "Unrounded means; exact ties within 1e-12; numerical rank only, not significance."
        ),
        adapted_lbunet_rows_preserved=(
            "All three datasets in S7 generated/lb_adapted_scores.tex, "
            "generated/tab_comparator_means.tex and full contrast tables; "
            "original generated/lb_multiclass_scores.tex also retained unchanged"
        ),
        main_reporting_scope=(
            "The same six models on every dataset: M0 M2 M6 GSL C1 C3. "
            "All C4 adaptations remain in S7 regardless of performance; "
            "published LB-UNet remains in panel B."
        ),
        published_values=receipts,
        source_run_files=63,
        absolute_run_metric_means=len(provenance),
        summary_checks=len(errors),
        tolerance=1e-10,
        max_error_vs_frozen_summary=max(errors),
        aggregation=(
            "Within each run: exclude classes absent in both masks; class macro "
            "within image/slice, then mean of defined images/slices. Across runs:"
            " mean and sample SD (ddof=1). HD95 is a positive distance, never a "
            "signed contrast."
        ),
        published_dsc_conversion="percent / 100; no distance conversion or metric imputation",
        scope=(
            "Descriptive display of existing outputs only; no evaluation, "
            "inference, training or significance analysis."
        ),
        architecture_sources=[
            "https://arxiv.org/abs/2105.05537",
            "https://arxiv.org/abs/2201.01266",
        ],
    )
    (VALIDATION / "comparison_table_verification.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


if __name__ == "__main__":
    result = build()
    print(
        f"{result['source_run_files']} runs; "
        f"{result['absolute_run_metric_means']} absolute means; "
        f"max reconciliation error {result['max_error_vs_frozen_summary']:.3g}"
    )
