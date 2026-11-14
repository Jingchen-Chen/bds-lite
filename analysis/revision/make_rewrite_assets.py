# ruff: noqa: E501
"""Assets for the reconstructed manuscript: run-level noise, sensitivity analyses, merged tables.

Everything is derived from existing per-case CSVs, checkpoints and summary tables; no model is
trained or re-run. Output is kept apart from make_manuscript_assets.py so that the two generators
cannot overwrite one another:

  revision/manuscript/generated/numbers_extra.tex, keys_extra.json   \\R{} keys
  revision/manuscript/generated/tab_two_sets.tex                     original vs retrained U-Net/BDS-Lite
  revision/manuscript/generated/tab_all_arms.tex                     every arm vs retrained U-Net
  revision/manuscript/generated/tab_datasets.tex                     datasets and structure
  revision/manuscript/generated/tab_lit_compact.tex                  published context + our rows
  revision/manuscript/generated/tab_s_all_case.tex, tab_s_run_level.tex, tab_s_selected_epochs.tex
  revision/manuscript/generated/rl_*.tex                             SESOI-relabelled copies
  revision/tables/families/run_level_noise.csv, all_case_sensitivity.csv
  revision/figures/run_level.pdf/.png

Added after the results were known (post hoc, not part of prereg_revision.md):
run-level margin. Per dataset and metric, the within-arm variance of the three run means is pooled
over all arms except LB-UNet (which is on a different scale on ACDC and Synapse); the margin
t(0.975, df) * SD * sqrt(2/3) is the half-width of a 95% interval for the difference between two
three-run means, treating runs of two arms as independent. Same-numbered seeds are not treated as
pairs because arms differ in architecture and random-number consumption.

Registered in prereg_revision.md but not reported until now: all-case distance sensitivity
(one empty mask = image diagonal, both empty = 0). The approximate minimum detectable effect is
2.80 * SD(cluster deltas) / sqrt(n) (two-sided alpha 0.05, 80% power, normal approximation).
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis/revision"))
import family_statistics as fs  # noqa: E402

GEN = ROOT / "revision/manuscript/generated"
FAM = ROOT / "revision/tables/families"
FIG = ROOT / "revision/figures"
DATASETS = ["isic2018", "acdc", "synapse"]
NAMES = {"isic2018": "ISIC2018", "acdc": "ACDC", "synapse": "Synapse"}
METRICS = fs.METRICS
MLABEL = {"dsc": "DSC", "iou": "IoU", "boundary_f1": "BF1", "hd95": "HD95", "assd": "ASSD"}
ARMS = ["M0", "M1", "M2", "M3", "M4", "M5", "M6", "GSL", "C1", "C3", "C4"]
POOLED = [a for a in ARMS if a != "C4"] + ["S0", "S6"]
LABEL = {
    "S0": "U-Net (original)",
    "S6": "BDS-Lite (original)",
    "M0": "U-Net",
    "M6": "BDS-Lite",
    "M1": "U-Net + surface loss",
    "M2": "U-Net + boundary decoder",
    "M3": "BDS-Lite $-$ surface loss",
    "M4": "BDS-Lite $-$ distillation",
    "M5": "BDS-Lite $-$ gate",
    "GSL": "U-Net + GSL",
    "C1": "U-Net + Boundary Loss",
    "C3": "Attention U-Net",
    "C4": "LB-UNet",
}
KEYS: dict[str, str] = {}


def put(key: str, value: str) -> None:
    KEYS[key] = value


def num(value: float, digits: int, sign: bool = False) -> str:
    text = f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"
    return text.replace("-", "$-$")


def digits(metric: str, delta: bool = False) -> int:
    return 2 if metric in fs.DISTANCE else 3


def read(path: Path) -> list[dict]:
    with path.open() as stream:
        return list(csv.DictReader(stream))


def run_means(method: str, dataset: str) -> dict[str, list[float]]:
    """Per run: mean over clusters of the cluster mean of the label-macro sample value."""
    result: dict[str, list[float]] = {m: [] for m in METRICS}
    for seed in fs.SEEDS:
        path = fs.per_case_path(method, dataset, seed)
        samples: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in read(path):
            for metric in METRICS:
                samples[row["id"]][metric].append(fs._value(row, metric))
        for metric in METRICS:
            clusters: dict[str, list[float]] = defaultdict(list)
            for sample, metrics in samples.items():
                arr = np.array(metrics[metric], float)
                arr = arr[np.isfinite(arr)]
                if arr.size:
                    clusters[fs.cluster_of(dataset, sample)].append(float(arr.mean()))
            result[metric].append(float(np.mean([np.mean(v) for v in clusters.values()])))
    return result


def run_level() -> dict:
    rows, data = [], {}
    for dataset in DATASETS:
        means = {arm: run_means(arm, dataset) for arm in ARMS + ["S0", "S6"]}
        data[dataset] = {"means": means, "margin": {}, "sd": {}}
        for metric in METRICS:
            sign = -1.0 if metric in fs.DISTANCE else 1.0
            ss = sum(np.var(means[a][metric], ddof=1) * 2 for a in POOLED)
            df = 2 * len(POOLED)
            sd = math.sqrt(ss / df)
            margin = stats.t.ppf(0.975, df) * sd * math.sqrt(2 / 3)
            data[dataset]["margin"][metric], data[dataset]["sd"][metric] = margin, sd
            d = digits(metric)
            put(f"RL.{dataset}.{metric}.sd", num(sd, d + 1 if d == 3 else d))
            put(f"RL.{dataset}.{metric}.margin", num(margin, d))

            def diff(
                a: str, b: str, means: dict = means, metric: str = metric, sign: float = sign
            ) -> float:
                return sign * (np.mean(means[a][metric]) - np.mean(means[b][metric]))

            put(f"RL.{dataset}.{metric}.ugap", num(diff("S0", "M0"), d, True))
            put(f"RL.{dataset}.{metric}.bgap", num(diff("S6", "M6"), d, True))
            put(f"RL.{dataset}.{metric}.S", num(diff("S6", "S0"), d, True))
            put(f"RL.{dataset}.{metric}.M", num(diff("M6", "M0"), d, True))
            for tag, (a, b) in {"S": ("S6", "S0"), "M": ("M6", "M0")}.items():
                seeds = [
                    sign * (x - y) for x, y in zip(means[a][metric], means[b][metric], strict=True)
                ]
                put(
                    f"RL.{dataset}.{metric}.{tag}.perseed",
                    ", ".join(num(s, d, True) for s in seeds),
                )
            for arm in ARMS + ["S0", "S6"]:
                rows.append(
                    dict(
                        dataset=dataset,
                        metric=metric,
                        arm=arm,
                        run1=means[arm][metric][0],
                        run2=means[arm][metric][1],
                        run3=means[arm][metric][2],
                        run_mean=float(np.mean(means[arm][metric])),
                        run_sd=float(np.std(means[arm][metric], ddof=1)),
                        pooled_sd=sd,
                        margin=margin,
                        delta_vs_M0=diff(arm, "M0"),
                        delta_vs_M6=diff(arm, "M6"),
                        beyond_margin_vs_M0=abs(diff(arm, "M0")) > margin,
                        beyond_margin_vs_M6=abs(diff(arm, "M6")) > margin,
                    )
                )
        above = [a for a in POOLED if a not in ("S0", "S6", "M0")]
        above = [a for a in above if np.mean(means[a]["dsc"]) > np.mean(means["S0"]["dsc"])]
        put(f"RL.{dataset}.dsc.above.S0", str(len(above)))
        put(f"RL.{dataset}.dsc.S0.mean", num(float(np.mean(means["S0"]["dsc"])), 3))
        put(f"RL.{dataset}.dsc.M0.mean", num(float(np.mean(means["M0"]["dsc"])), 3))
        boundary_arms = [a for a in ARMS if a not in ("M0", "C4")]
        for ref in ("M0", "M6"):
            arms = [a for a in boundary_arms if a != ref]
            hits = [
                r
                for r in rows
                if r["dataset"] == dataset and r["arm"] in arms and r[f"beyond_margin_vs_{ref}"]
            ]
            put(f"RL.{dataset}.{ref}.hits", str(len(hits)))
            put(f"RL.{dataset}.{ref}.tests", str(len(arms) * len(METRICS)))
            favour = [r for r in hits if r[f"delta_vs_{ref}"] > 0]
            put(f"RL.{dataset}.{ref}.hits.favour", str(len(favour)))
    with (FAM / "run_level_noise.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return data


def selected_epochs() -> list[dict]:
    import torch

    def from_checkpoint(path: Path) -> int:
        return int(torch.load(path, map_location="cpu", weights_only=False)["epoch"])

    def run(dataset: str, arm: str, seed: int) -> int | None:
        final = ROOT / f"results/revision_runs/{dataset}_{arm}_seed{seed}/final_metrics.json"
        return json.loads(final.read_text())["selected_epoch"] if final.exists() else None

    out = []
    for dataset in DATASETS:
        rows = {}
        rows["S0"] = [
            from_checkpoint(ROOT / f"outputs/runs/{dataset}_unet_seed{s}/best.pt") for s in fs.SEEDS
        ]
        rows["S6"] = [
            from_checkpoint(ROOT / f"outputs/runs/{dataset}_bds_lite_full_seed{s}/best.pt")
            for s in fs.SEEDS
        ]
        rows["M0"] = [
            from_checkpoint(ROOT / f"results/checkpoints/{dataset}_p1b_unet_seed{s}/best.pt")
            for s in fs.SEEDS
        ]
        if dataset == "acdc":
            rows["M6"] = [run(dataset, "m6_full_unweighted", s) for s in fs.SEEDS]
        else:
            rows["M6"] = [
                from_checkpoint(
                    ROOT / f"results/checkpoints/{dataset}_p1b_bdslite_full_current_seed{s}/best.pt"
                )
                for s in fs.SEEDS
            ]
        for arm, values in rows.items():
            put(f"SEL.{dataset}.{arm}", "/".join(str(v) for v in values))
            out.append(dict(dataset=dataset, arm=arm, epochs=values))
        original, retrained = rows["S0"] + rows["S6"], rows["M0"] + rows["M6"]
        put(f"SEL.{dataset}.original.median", f"{np.median(original):.0f}")
        put(f"SEL.{dataset}.retrained.median", f"{np.median(retrained):.0f}")
    return out


def error_shares() -> None:
    rows = read(FAM / "error_bands.csv")
    for row in rows:
        near = float(row["fp_near_per_image"]) + float(row["fn_near_per_image"])
        far = float(row["fp_far_per_image"]) + float(row["fn_far_per_image"])
        put(f"ERR.{row['dataset']}.{row['method']}.near", f"{100 * near / (near + far):.0f}")


def component_stats() -> None:
    rows = [r for r in read(ROOT / "revision/structural_descriptors/per_sample_class.csv")]
    values = np.array(
        [int(r["components_4connected"]) for r in rows if r["dataset"] == "isic2018"], float
    )
    put("CMP.isic2018.median", f"{np.median(values):.0f}")
    put("CMP.isic2018.single", f"{100 * np.mean(values == 1):.0f}")


def mde() -> None:
    for dataset in ("acdc", "synapse"):
        for metric in METRICS:
            values = np.array(list(fs.cluster_deltas("S6", "S0", dataset, metric).values()))
            effect = 2.80 * values.std(ddof=1) / math.sqrt(values.size)
            put(f"MDE.{dataset}.{metric}", num(effect, digits(metric)))


def penalised_sample(method: str, dataset: str, metric: str) -> dict[str, float]:
    per_seed = [read(fs.per_case_path(method, dataset, s)) for s in fs.SEEDS]
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for table in per_seed:
        for row in table:
            acc[row["id"]][row.get("label", "1")].append(float(row[metric]))
    return {
        sample: float(np.mean([np.mean(v) for v in labels.values()]))
        for sample, labels in acc.items()
    }


def all_case() -> list[dict]:
    rows = []
    pairs = [("S6", "S0", "original"), ("M6", "M0", "retrained")]
    for dataset in DATASETS:
        for a, b, tag in pairs:
            for metric, column in (("hd95", "hd95_penalized"), ("assd", "assd_penalized")):
                va = penalised_sample(a, dataset, column)
                vb = penalised_sample(b, dataset, column)
                clusters: dict[str, list[float]] = defaultdict(list)
                for sample in va:
                    clusters[fs.cluster_of(dataset, sample)].append(vb[sample] - va[sample])
                values = np.array([np.mean(v) for v in clusters.values()])
                low, high = fs.bootstrap(values)
                primary = np.array(list(fs.cluster_deltas(a, b, dataset, metric).values()))
                rows.append(
                    dict(
                        dataset=dataset,
                        set=tag,
                        metric=metric,
                        n=values.size,
                        primary_delta=float(primary.mean()),
                        all_case_delta=float(values.mean()),
                        ci95_low=low,
                        ci95_high=high,
                    )
                )
                key = f"AC.{dataset}.{tag}.{metric}"
                put(f"{key}.delta", num(float(values.mean()), 2, True))
                put(f"{key}.ci", f"[{num(low, 2, True)}, {num(high, 2, True)}]")
    with (FAM / "all_case_sensitivity.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def volume_keys() -> None:
    for row in read(FAM / "volume_method_summary.csv"):
        key = f"VOL.{row['dataset']}.{row['method']}"
        put(f"{key}.dsc", f"{100 * float(row['dsc']):.1f}")
        put(f"{key}.hd95", f"{float(row['hd95']):.1f}")


def stats_rows() -> dict[tuple, dict]:
    return {
        (r["dataset"], r["family"], r["method"], r["reference"], r["metric"]): r
        for r in read(FAM / "family_statistics.csv")
    }


def comparator_counts(stats_by: dict) -> None:
    cells = [stats_by[(d, "C-vs-M6", "GSL", "M6", m)] for d in DATASETS for m in METRICS]
    put("GSLM6.cells", str(len(cells)))
    put("GSLM6.positive", str(sum(float(r["mean_oriented_delta"]) > 0 for r in cells)))
    put("GSLM6.holm", str(sum(r["decision"] == "favourable" for r in cells)))
    put("GSLM6.against", str(sum(r["decision"] == "unfavourable" for r in cells)))
    loo = [
        stats_by[(d, "A", a, "M6", m)]
        for d in DATASETS
        for a in ("M3", "M4", "M5")
        for m in METRICS
    ]
    put("LOO.tests", str(len(loo)))
    put("LOO.worse", str(sum(r["decision"] == "unfavourable" for r in loo)))
    put("LOO.better", str(sum(r["decision"] == "favourable" for r in loo)))


def tab_two_sets(stats_by: dict, runs: dict) -> None:
    summary = {(r["dataset"], r["method"]): r for r in read(FAM / "method_summary.csv")}
    lines = [
        "% generated by analysis/revision/make_rewrite_assets.py",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Dataset, model set & Arm & DSC $\\uparrow$ & IoU $\\uparrow$ & BF1 $\\uparrow$ & HD95 (px) $\\downarrow$ & ASSD (px) $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for dataset in DATASETS:
        for family, unet, bds, tag in (
            ("P", "S0", "S6", "original"),
            ("P-matched", "M0", "M6", "retrained"),
        ):
            for arm, label in ((unet, "U-Net"), (bds, "BDS-Lite")):
                cells = []
                for metric in METRICS:
                    d = digits(metric)
                    row = summary[(dataset, arm)]
                    mean, sd = float(row[f"{metric}_mean"]), float(row[f"{metric}_sd"])
                    cells.append(f"{mean:.{d}f} $\\pm$ {sd:.{d}f}")
                head = f"{NAMES[dataset]}, {tag}" if arm == unet else ""
                lines.append(f"{head} & {label} & " + " & ".join(cells) + " \\\\")
            cells = []
            for metric in METRICS:
                d = digits(metric)
                r = stats_by[(dataset, family, bds, unet, metric)]
                star = "$^{*}$" if r["decision"] != "no decision" else ""
                delta = num(float(r["mean_oriented_delta"]), d, True)
                low = num(float(r["ci95_low"]), d, True)
                high = num(float(r["ci95_high"]), d, True)
                cells.append(f"{delta}{star} [{low}, {high}]")
            lines.append(" & $\\Delta$ [95\\% CI] & " + " & ".join(cells) + " \\\\")
            lines.append("\\addlinespace")
        margins = [num(runs[dataset]["margin"][m], digits(m)) for m in METRICS]
        lines.append(" & Run-level margin & " + " & ".join(f"$\\pm${m}" for m in margins) + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    (GEN / "tab_two_sets.tex").write_text("\n".join(lines) + "\n")


def tab_all_arms(stats_by: dict, runs: dict) -> None:
    show = ["dsc", "boundary_f1", "hd95"]
    summary = {(r["dataset"], r["method"]): r for r in read(FAM / "method_summary.csv")}
    head = " & ".join(f"\\multicolumn{{3}}{{c}}{{{NAMES[d]}}}" for d in DATASETS)
    sub = " & ".join(" & ".join(MLABEL[m] for m in show) for _ in DATASETS)
    rules = "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}"
    lines = [
        "% generated by analysis/revision/make_rewrite_assets.py",
        "\\begin{tabular}{lrrrrrrrrr}",
        "\\toprule",
        f" & {head} \\\\",
        rules,
        f"Arm & {sub} \\\\",
        "\\midrule",
    ]
    cells = []
    for dataset in DATASETS:
        for metric in show:
            value = float(summary[(dataset, "M0")][f"{metric}_mean"])
            cells.append(f"{value:.{digits(metric)}f}")
    lines.append("U-Net (absolute) & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    groups = [
        [("M6", "P-matched")],
        [("M1", "A0"), ("M2", "A0"), ("M3", "A0"), ("M4", "A0"), ("M5", "A0")],
        [("GSL", "C"), ("C1", "C"), ("C3", "C"), ("C4", "C")],
    ]
    for group in groups:
        for arm, family in group:
            cells = []
            for dataset in DATASETS:
                for metric in show:
                    r = stats_by[(dataset, family, arm, "M0", metric)]
                    delta = float(r["mean_oriented_delta"])
                    marks = "*" if r["decision"] != "no decision" else ""
                    means = runs[dataset]["means"]
                    gap = np.mean(means[arm][metric]) - np.mean(means["M0"][metric])
                    if abs(gap) > runs[dataset]["margin"][metric]:
                        marks += "\\dagger"
                    mark = f"$^{{{marks}}}$" if marks else ""
                    cells.append(num(delta, digits(metric), True) + mark)
            lines.append(f"{LABEL[arm]} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    cells = []
    for dataset in DATASETS:
        for metric in show:
            cells.append(f"$\\pm${num(runs[dataset]['margin'][metric], digits(metric))}")
    lines.append("Run-level margin & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (GEN / "tab_all_arms.tex").write_text("\n".join(lines) + "\n")


def tab_datasets() -> None:
    rows = read(ROOT / "revision/structural_descriptors/summary.csv")
    info = {
        "isic2018": ("Dermoscopy", "1", "2,075 / 519 / 1,000 images", "image (1,000)"),
        "acdc": ("Cine MRI", "3", "1,350 / 186 / 366 slices", "patient (20)"),
        "synapse": ("Abdominal CT", "8", "1,738 / 473 / 1,568 slices", "case (12)"),
    }
    lines = [
        "% generated by analysis/revision/make_rewrite_assets.py from structural_descriptors/summary.csv",
        "\\begin{tabular}{lllllll}",
        "\\toprule",
        "Dataset & Modality & Classes & Train / val / test & Test unit ($n$) & Class absent from image & DSC loss per 1\\,px shift \\\\",
        "\\midrule",
    ]
    for dataset in DATASETS:
        part = [r for r in rows if r["dataset"] == dataset]
        empty = [100 * float(r["empty_class_fraction"]) for r in part]
        drop = [float(r["first_order_dsc_drop_per_1px"]) for r in part]
        modality, classes, split, unit = info[dataset]
        empty_text = (
            f"{min(empty):.0f}\\%" if len(part) == 1 else f"{min(empty):.0f}--{max(empty):.0f}\\%"
        )
        drop_text = f"{drop[0]:.3f}" if len(part) == 1 else f"{min(drop):.3f}--{max(drop):.3f}"
        lines.append(
            f"{NAMES[dataset]} & {modality} & {classes} & {split} & {unit} & {empty_text} & {drop_text} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    (GEN / "tab_datasets.tex").write_text("\n".join(lines) + "\n")


def tab_lit_compact() -> None:
    lit = {
        (r["method"], r["dataset"], r["metric"]): r for r in read(ROOT / "revision/lit_numbers.csv")
    }
    vol = {(r["dataset"], r["method"]): r for r in read(FAM / "volume_method_summary.csv")}
    jac = {r["method"]: r for r in read(FAM / "native_jaccard.csv")}

    def value(method: str, dataset: str, metric: str) -> str:
        return lit[(method, dataset, metric)]["value"]

    def pct(x: str) -> str:
        return f"{100 * float(x):.1f}"

    lines = [
        "% generated by analysis/revision/make_rewrite_assets.py from lit_numbers.csv and family tables",
        "\\begin{tabular}{lllrr}",
        "\\toprule",
        "Task and protocol & Method & Source & \\multicolumn{2}{c}{Result} \\\\",
        "\\midrule",
        " & & & 3D DSC (\\%) & HD95 \\\\",
    ]
    syn = "Synapse"
    for method, cite in (
        ("TransUNet", "\\citet{Chen2021TransUNetArxiv}"),
        ("Swin-Unet", "\\citet{Cao2022SwinUnet}"),
        ("R50 U-Net (TransUNet table)", "\\citet{Chen2021TransUNetArxiv}"),
    ):
        name = method.split(" (")[0]
        head = "Synapse, 18 training cases" if method == "TransUNet" else ""
        lines.append(
            f"{head} & {name} & {cite} & {value(method, syn, 'DSC')} & {value(method, syn, 'HD95')} \\\\"
        )
    for arm, label in (("S0", "U-Net, original"), ("S6", "BDS-Lite, original")):
        r = vol[("synapse", arm)]
        head = "Synapse, 14 training cases" if arm == "S0" else ""
        lines.append(
            f"{head} & {label} & this study & {pct(r['dsc'])} & {float(r['hd95']):.1f} \\\\"
        )
    lines.append("\\midrule")
    for method, cite in (
        ("TransUNet", "\\citet{Chen2021TransUNetArxiv}"),
        ("Swin-Unet", "\\citet{Cao2022SwinUnet}"),
    ):
        head = "ACDC, published split" if method == "TransUNet" else ""
        lines.append(f"{head} & {method} & {cite} & {value(method, 'ACDC', 'DSC')} & -- \\\\")
    for arm, label in (("S0", "U-Net, original"), ("S6", "BDS-Lite, original")):
        r = vol[("acdc", arm)]
        head = "ACDC, 70 training patients" if arm == "S0" else ""
        lines.append(
            f"{head} & {label} & this study & {pct(r['dsc'])} & {float(r['hd95']):.1f} \\\\"
        )
    lines += ["\\midrule", " & & & Thresh.\\ Jaccard & Jaccard \\\\"]
    isic = "ISIC2018 official test (1000)"
    for method, label in (
        ("ISIC2018 challenge rank 1", "Challenge rank 1"),
        ("ISIC2018 challenge rank 5", "Challenge rank 5"),
    ):
        head = "ISIC2018 official test" if "rank 1" in method else ""
        lines.append(
            f"{head} & {label} & \\citet{{Codella2019Skin}} & {value(method, isic, 'thresholded Jaccard')} & {value(method, isic, 'Jaccard')} \\\\"
        )
    for arm, label in (
        ("S0", "U-Net, original"),
        ("S6", "BDS-Lite, original"),
        ("M0", "U-Net, retrained"),
        ("M6", "BDS-Lite, retrained"),
        ("GSL", "U-Net + GSL"),
        ("C4", "LB-UNet"),
    ):
        r = jac[arm]
        lines.append(
            f" & {label} & this study & {float(r['thresholded_jaccard']):.3f} & {float(r['jaccard']):.3f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    (GEN / "tab_lit_compact.tex").write_text("\n".join(lines) + "\n")


def tab_s_run_level(runs: dict) -> None:
    lines = [
        "% generated by analysis/revision/make_rewrite_assets.py",
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Dataset & Metric & Pooled run SD & Margin & Same-recipe gap, U-Net & Same-recipe gap, BDS-Lite & BDS-Lite $-$ U-Net, original & BDS-Lite $-$ U-Net, retrained \\\\",
        "\\midrule",
    ]
    for dataset in DATASETS:
        for metric in METRICS:
            k = f"RL.{dataset}.{metric}"
            lines.append(
                f"{NAMES[dataset]} & {MLABEL[metric]} & {KEYS[k + '.sd']} & $\\pm${KEYS[k + '.margin']} & {KEYS[k + '.ugap']} & {KEYS[k + '.bgap']} & {KEYS[k + '.S']} & {KEYS[k + '.M']} \\\\"
            )
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    (GEN / "tab_s_run_level.tex").write_text("\n".join(lines) + "\n")


def tab_s_all_case(rows: list[dict]) -> None:
    lines = [
        "% generated by analysis/revision/make_rewrite_assets.py from all_case_sensitivity.csv",
        "\\begin{tabular}{lllrrl}",
        "\\toprule",
        "Dataset & Model set & Metric & Primary $\\Delta$ & All-case $\\Delta$ & All-case 95\\% CI \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(
            f"{NAMES[r['dataset']]} & {r['set']} & {MLABEL[r['metric']]} & {num(r['primary_delta'], 2, True)} & {num(r['all_case_delta'], 2, True)} & [{num(r['ci95_low'], 2, True)}, {num(r['ci95_high'], 2, True)}] \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    (GEN / "tab_s_all_case.tex").write_text("\n".join(lines) + "\n")


def tab_s_selected(rows: list[dict]) -> None:
    lines = [
        "% generated by analysis/revision/make_rewrite_assets.py from best.pt and final_metrics.json",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Dataset & U-Net, original & BDS-Lite, original & U-Net, retrained & BDS-Lite, retrained \\\\",
        "\\midrule",
    ]
    for dataset in DATASETS:
        cells = [KEYS[f"SEL.{dataset}.{arm}"] for arm in ("S0", "S6", "M0", "M6")]
        lines.append(f"{NAMES[dataset]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (GEN / "tab_s_selected_epochs.tex").write_text("\n".join(lines) + "\n")


def relabel() -> None:
    """Copies of generated tables with descriptive threshold labels and model-set names.

    The follow-up family equals the retrained set on ISIC2018 and Synapse, but its ACDC BDS-Lite
    used foreground class weights, so it keeps its own name. Long tables become longtables.
    """
    swaps = [
        ("practically superior", "beyond $+$SESOI"),
        ("practically inferior", "beyond $-$SESOI"),
        ("practically equivalent", "within SESOI"),
        ("BDS-Lite (submitted models) vs U-Net (submitted models)", "BDS-Lite vs U-Net, original"),
        ("(submitted models)", "(original)"),
        ("submitted", "original"),
        ("(replication family)", "(follow-up)"),
        ("replication", "follow-up"),
    ]
    long_tables = {"tab_s_collapse", "tab_s_errorbands", "tab_s_perclass", "tab_s_gradient"}
    # A longtable steps the table counter even without a caption, so every printed one gets one.
    captions = {
        "tab_s_errorbands": (
            "tab:errorbands",
            "Error pixels and components per image by type, seed means. Near: within 2\\,px of "
            "the reference contour. Spurious component: a predicted component with no reference "
            "overlap.",
        ),
        "tab_s_gradient": (
            "tab:gradient",
            "Gradient probes on the shared encoder parameters in the seed-1 diagnostic runs: "
            "cosine between the segmentation gradient and each auxiliary gradient, by training "
            "phase.",
        ),
    }
    for name in (
        "tab_primary_stats",
        "tab_s_duplicates",
        "tab_s_matched",
        "tab_s_replication",
        "tab_s_replication_means",
        "tab_s_comparators_vs_unet",
        "tab_s_comparators_vs_full",
        "tab_volume",
        *sorted(long_tables),
    ):
        text = (GEN / f"{name}.tex").read_text()
        for old, new in swaps:
            text = text.replace(old, new)
        if name in long_tables:
            text = text.replace("\\begin{tabular}", "\\begin{longtable}")
            text = text.replace("\\end{tabular}", "\\end{longtable}")
        if name in captions:
            label, caption = captions[name]
            head, sep, body = text.partition("\\begin{longtable}")
            spec, newline, rest = body.partition("\n")
            text = f"{head}{sep}{spec}{newline}\\caption{{\\label{{{label}}}{caption}}}\\\\\n{rest}"
        (GEN / f"rl_{name}.tex").write_text(text)


def figure(runs: dict) -> None:
    order = ["S0", "S6", "M0", "M6", "M1", "M2", "M3", "M4", "M5", "GSL", "C1", "C3", "C4"]
    group = {a: "bds" for a in ("M6", "M1", "M2", "M3", "M4", "M5")}
    group |= {"S0": "orig", "S6": "orig", "M0": "base"}
    group |= {a: "comp" for a in ("GSL", "C1", "C3", "C4")}
    # Slots 1-3 of the validated categorical palette plus neutral for the reference arm.
    style = {
        "orig": ("#eb6834", "s", "Original models"),
        "base": ("#52514e", "D", "Retrained U-Net"),
        "bds": ("#2a78d6", "o", "Retrained BDS-Lite and ablations"),
        "comp": ("#1baf7a", "^", "Retrained comparators"),
    }
    names = {k: v.replace("$-$", "−") for k, v in LABEL.items()}
    names["M0"], names["M6"] = "U-Net (retrained)", "BDS-Lite (retrained)"
    show = [("dsc", "DSC"), ("boundary_f1", "Boundary F1"), ("hd95", "HD95 (px)")]
    limits = {
        ("acdc", "dsc"): (0.70, 0.80),
        ("acdc", "boundary_f1"): (0.75, 0.84),
        ("acdc", "hd95"): (4.5, 11.0),
        ("synapse", "dsc"): (0.43, 0.58),
        ("synapse", "boundary_f1"): (0.45, 0.60),
        ("synapse", "hd95"): (7.0, 11.0),
    }
    plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "axes.linewidth": 0.5})
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 7.4), sharey=True)
    ys = {arm: len(order) - 1 - i for i, arm in enumerate(order)}
    for r, (metric, title) in enumerate(show):
        for c, dataset in enumerate(DATASETS):
            ax = axes[r, c]
            means = runs[dataset]["means"]
            center = float(np.mean(means["M0"][metric]))
            margin = runs[dataset]["margin"][metric]
            ax.axvspan(center - margin, center + margin, color="#d8d7d2", alpha=0.6, lw=0)
            ax.axvline(center, color="#52514e", lw=0.7)
            low, high = limits.get((dataset, metric), (None, None))
            for arm in order:
                color, marker, _ = style[group[arm]]
                values = np.array(means[arm][metric])
                y = ys[arm]
                if low is not None and (values.max() < low or values.min() > high):
                    worse_low = values.max() < low
                    edge = low if worse_low else high
                    pad = (high - low) * 0.015
                    # Off-scale arms are always worse; every panel has "better" on the right.
                    ax.text(
                        edge + (pad if worse_low else -pad),
                        y,
                        f"\u2190 {values.mean():.2f}",
                        ha="left",
                        va="center",
                        fontsize=6,
                        color="#52514e",
                    )
                    continue
                ax.scatter(values, [y] * 3, s=11, c=color, marker=marker, lw=0, alpha=0.9, zorder=3)
                ax.plot([values.mean()] * 2, [y - 0.32, y + 0.32], color=color, lw=1.1, zorder=3)
            if low is not None:
                ax.set_xlim(low, high)
            ax.set_ylim(-0.7, len(order) - 0.3)
            ax.axhline(ys["M6"] - 0.5 + 1, color="#c3c2b7", lw=0.4)
            ax.axhline(ys["M0"] + 0.5, color="#c3c2b7", lw=0.4)
            ax.axhline(ys["GSL"] + 0.5, color="#c3c2b7", lw=0.4)
            ax.tick_params(length=2, width=0.4, labelsize=6.5)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4, prune="both"))
            ax.grid(False)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if r == 0:
                ax.set_title(NAMES[dataset], fontsize=8, fontweight="bold")
            ax.set_xlabel(title, fontsize=7, labelpad=1.5)
            if metric == "hd95":
                ax.invert_xaxis()
    axes[0, 0].set_yticks([ys[a] for a in order], [names[a] for a in order])
    for r in range(3):
        axes[r, 0].tick_params(axis="y", labelleft=True)
        axes[r, 0].set_yticks([ys[a] for a in order], [names[a] for a in order])
    handles = [
        plt.Line2D([], [], color=c, marker=m, lw=0, markersize=4, label=label)
        for c, m, label in style.values()
    ]
    band = plt.Rectangle(
        (0, 0), 1, 1, color="#d8d7d2", alpha=0.6, label="Retrained U-Net mean ± run-level margin"
    )
    fig.legend(handles=[*handles, band], loc="lower center", ncol=3, frameon=False, fontsize=7)
    fig.subplots_adjust(left=0.2, right=0.985, top=0.965, bottom=0.095, wspace=0.13, hspace=0.24)
    fig.savefig(FIG / "run_level.pdf")
    fig.savefig(FIG / "run_level.png", dpi=300)


def training_cost() -> None:
    """Training-step profile as reported in Table 4 of the submitted manuscript."""
    tex = (ROOT / "recovered/peerj-submission/main.tex").read_text()
    for tag, name in (("U-Net ", "unet"), ("BDS-Lite ", "bds"), ("U-Net+GSL ", "gsl")):
        match = re.search(
            rf"^{re.escape(tag)}\s*&.*\$([\d.]+) \\pm [\d.]+\$\s*& ([\d.]+) \\\\", tex, re.M
        )
        put(f"TRAIN.{name}.ms", f"{float(match.group(1)):.1f}")
        put(f"TRAIN.{name}.mb", f"{float(match.group(2)):.0f}")


def write_keys() -> None:
    lines = ["% generated by analysis/revision/make_rewrite_assets.py; do not edit"]
    for key in sorted(KEYS):
        lines.append(f"\\expandafter\\def\\csname R@{key}\\endcsname{{{KEYS[key]}}}")
    (GEN / "numbers_extra.tex").write_text("\n".join(lines) + "\n")
    (GEN / "keys_extra.json").write_text(json.dumps(sorted(KEYS), indent=0) + "\n")


def main() -> None:
    runs = run_level()
    selected = selected_epochs()
    error_shares()
    component_stats()
    mde()
    case_rows = all_case()
    volume_keys()
    training_cost()
    by = stats_rows()
    comparator_counts(by)
    tab_two_sets(by, runs)
    tab_all_arms(by, runs)
    tab_datasets()
    tab_lit_compact()
    tab_s_run_level(runs)
    tab_s_all_case(case_rows)
    tab_s_selected(selected)
    relabel()
    figure(runs)
    write_keys()
    print(f"{len(KEYS)} keys")


if __name__ == "__main__":
    main()
