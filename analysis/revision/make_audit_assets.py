# ruff: noqa: E501
"""Manuscript assets from the 2026-11-07 project audit. Nothing is trained or evaluated here.

Reads completed runs only. Experiment C runs are excluded by name and by seed; none of its files is
opened. All contrasts use the frozen estimator of analysis/revision/expansion_statistics.py.

Writes:
  results/tables/families/audit_validation_contrasts.csv, audit_development_controls.csv,
  audit_isic_membership.csv
  paper/generated/numbers_audit.tex and tab_s_membership.tex, tab_s_validation.tex,
  tab_s_development.tex, tab_s_ledger.tex
  paper/figures/transfer_mechanism.pdf
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
_VALIDATION = ROOT / "results/historical_reevaluation/validation.json"
# Evidence classes change only once the frozen historical re-evaluation has passed validation.
REEVALUATED = _VALIDATION.exists() and json.loads(_VALIDATION.read_text()).get("status") == "PASSED"
EXPLORATORY_RETESTED = (
    "Exploratory; re-evaluated post hoc on test"
    if REEVALUATED
    else "Exploratory (development images)"
)
sys.path.insert(0, str(ROOT / "analysis/revision"))
import expansion_statistics as ex  # noqa: E402

FAM = ROOT / "results/tables/families"
GEN = ROOT / "paper/generated"
FIG = ROOT / "paper/figures"
RUNS = ROOT / "results/revision_runs"
FOLLOW = ROOT / "recovered/followup/results"
AUDIT = ROOT / "results/audit/project_audit_20261107"
STAGE2 = {
    r["run_id"]
    for r in yaml.safe_load((ROOT / "protocols/confirmation/queue_confirmation.yaml").read_text())[
        "runs"
    ]
}
KEYS: dict[str, str] = {}
METRIC_DIGITS = {"boundary_f1": 3, "dsc": 3, "iou": 3, "hd95": 2, "assd": 2}
BLUE, ORANGE, AQUA, INK, MUTED, GRID = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#2b2b2b",
    "#52514e",
    "#e9e8e4",
)


def num(value: float, digits: int = 3, sign: bool = True) -> str:
    text = f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"
    return text.replace("-", "$-$")


def interval(low: float, high: float, digits: int = 3) -> str:
    return f"[{num(low, digits)}, {num(high, digits)}]"


def run(name: str, seed: int, split: str) -> Path:
    run_id = f"isic2018_{name}_seed{seed}"
    if run_id in STAGE2 or seed > 10:
        raise ValueError(f"Experiment C run requested: {run_id}")
    return RUNS / run_id / f"per_case_{split}.csv"


def validation_path(arm: str, seed: int) -> Path:
    if arm in ("unet", "full") and seed <= 3:
        return (
            FOLLOW
            / f"followup_20261018_v2/isic2018_{'U' if arm == 'unet' else 'F'}_seed{seed}/per_case.csv"
        )
    return run(
        {"unet": "m0_unet", "full": "m6_full", "decoder": "m2_aux_boundary_only", "gsl": "gsl"}[
            arm
        ],
        seed,
        "val",
    )


def development_path(arm: str, seed: int) -> Path:
    roots = {
        "U": "followup_20261018_v2",
        "S": "followup_20261018_v2",
        "B": "followup_20261018_v2",
        "F": "followup_20261018_v2",
        "G": "pathway_controls_20261018",
        "C": "pathway_controls_20261018",
        "D": "boundary_specificity_20261018",
        "UA": "region_baseline_20261018",
    }
    if arm == "M2":
        return run("m2_aux_boundary_only", seed, "val")
    return FOLLOW / f"{roots[arm]}/isic2018_{arm}_seed{seed}/per_case.csv"


def write_csv(name: str, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (FAM / name).open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def validation_contrasts() -> list[dict]:
    values = {
        (a, s): ex.unit_values(validation_path(a, s), "isic2018") for a in ex.NAMES for s in ex.ALL
    }
    rows = []
    for metric in ex.METRICS:
        block = []
        for first, second in ex.PAIRS:
            a, b = [values[(first, s)] for s in ex.ALL], [values[(second, s)] for s in ex.ALL]
            result = ex.contrast(a, b, metric)
            block.append(dict(metric=metric, first=first, second=second, **result))
            digits = METRIC_DIGITS[metric]
            key = f"VAL.{first}.{second}.{metric}"
            KEYS[f"{key}.delta"] = num(result["delta"], digits)
            KEYS[f"{key}.ci"] = interval(result["low"], result["high"], digits)
            KEYS[f"{key}.label"] = result["classification"]
            KEYS[f"VAL.{first}.{metric}.mean"] = num(result["arm_a_mean"], digits, False)
            KEYS[f"VAL.{second}.{metric}.mean"] = num(result["arm_b_mean"], digits, False)
            KEYS[f"VAL.{first}.{metric}.sd"] = num(result["arm_a_sd"], 4, False)
            KEYS[f"VAL.{second}.{metric}.sd"] = num(result["arm_b_sd"], 4, False)
        ex.holm(block)
        rows += block
    write_csv("audit_validation_contrasts.csv", rows)
    return rows


def development_controls() -> list[dict]:
    arms = ("U", "S", "B", "F", "G", "C", "D", "UA", "M2")
    values = {
        a: [ex.unit_values(development_path(a, s), "isic2018") for s in (1, 2, 3)] for a in arms
    }
    ten = [ex.unit_values(run("m2_aux_boundary_only", s, "val"), "isic2018") for s in ex.ALL]
    pairs = [
        ("B", "C"),
        ("C", "G"),
        ("G", "U"),
        ("D", "G"),
        ("B", "D"),
        ("C", "M2"),
        ("B", "M2"),
        ("UA", "U"),
    ]
    rows = []
    for metric in ex.METRICS:
        digits = METRIC_DIGITS[metric]
        for first, second in pairs:
            result = ex.contrast(values[first], values[second], metric)
            rows.append(dict(metric=metric, first=first, second=second, **result))
            KEYS[f"DEV.{first}.{second}.{metric}.delta"] = num(result["delta"], digits)
            KEYS[f"DEV.{first}.{second}.{metric}.ci"] = interval(
                result["low"], result["high"], digits
            )
            KEYS[f"DEV.{first}.{metric}.mean"] = num(result["arm_a_mean"], digits, False)
            KEYS[f"DEV.{second}.{metric}.mean"] = num(result["arm_b_mean"], digits, False)
        result = ex.contrast(values["C"], ten, metric)
        rows.append(dict(metric=metric, first="C", second="M2 ten seeds", **result))
        KEYS[f"DEV.C.M2ten.{metric}.delta"] = num(result["delta"], digits)
    for arm in ("U", "UA"):
        count = 0
        for seed in (1, 2, 3):
            count += int(
                pd.read_csv(development_path(arm, seed))["empty_prediction"].astype(bool).sum()
            )
        KEYS[f"DEV.{arm}.empty"] = str(count)
    write_csv("audit_development_controls.csv", rows)
    return rows


def membership() -> list[dict]:
    split = json.loads((ROOT / "splits/isic2018_split.json").read_text())
    train, val, test = ({Path(i).name for i in split[k]} for k in ("train", "val", "test"))
    frame = pd.read_csv(ROOT / "results/midterm_q2_rescue/analysis/per_case_metrics.csv")
    frame = frame[frame.dataset == "isic2018"].copy()
    frame["subset"] = frame.sample_id.map(
        lambda i: "train" if i in train else "sel" if i in val else "held" if i in test else "?"
    )
    if (frame.subset == "?").any() or len(frame) != 519:
        raise ValueError("membership of the 519 images is incomplete")
    rng = np.random.default_rng(20261107)
    rows = []
    for subset, part in [*frame.groupby("subset"), ("all", frame)]:
        row = dict(subset=subset, n=len(part))
        KEYS[f"LK.{subset}.n"] = str(len(part))
        for metric, sign in (("dsc", 1), ("boundary_f1", 1), ("hd95", -1), ("assd", -1)):
            delta = (
                (sign * (part[f"bds_lite_full_{metric}"] - part[f"unet_{metric}"]))
                .dropna()
                .to_numpy()
            )
            draws = rng.choice(delta, (5000, delta.size)).mean(1)
            low, high = np.percentile(draws, [2.5, 97.5])
            digits = 2 if metric in ("hd95", "assd") else 3
            row |= {metric: delta.mean(), f"{metric}_low": low, f"{metric}_high": high}
            KEYS[f"LK.{subset}.{metric}.delta"] = num(delta.mean(), digits)
            KEYS[f"LK.{subset}.{metric}.ci"] = interval(low, high, digits)
        rows.append(row)
    arms = {}
    for arm in ("unet", "bds_lite_full"):
        parts = []
        for seed in (1, 2, 3):
            table = pd.read_csv(ROOT / f"outputs/evaluations/isic2018_{arm}_seed{seed}_test.csv")
            parts.append(
                table[table.class_id > 0].set_index("sample_id")[
                    ["dsc", "boundary_f1", "hd95", "assd"]
                ]
            )
        arms[arm] = sum(parts) / 3
    row = dict(subset="clean390", n=len(arms["unet"]))
    KEYS["LK.clean.n"] = str(len(arms["unet"]))
    for metric, sign in (("dsc", 1), ("boundary_f1", 1), ("hd95", -1), ("assd", -1)):
        delta = (sign * (arms["bds_lite_full"][metric] - arms["unet"][metric])).dropna().to_numpy()
        draws = rng.choice(delta, (5000, delta.size)).mean(1)
        low, high = np.percentile(draws, [2.5, 97.5])
        digits = 2 if metric in ("hd95", "assd") else 3
        row |= {metric: delta.mean(), f"{metric}_low": low, f"{metric}_high": high}
        KEYS[f"LK.clean.{metric}.delta"] = num(delta.mean(), digits)
        KEYS[f"LK.clean.{metric}.ci"] = interval(low, high, digits)
    rows.append(row)
    write_csv("audit_isic_membership.csv", rows)
    return rows


def context_keys() -> None:
    screen = json.loads((FOLLOW / "acdc_context_audit_20261019/screen.json").read_text())[
        "contrasts"
    ]
    dice = [100 * c["dice_delta"] for c in screen]
    assd = [-100 * c["assd_relative"] for c in screen]
    hd95 = [-100 * c["hd95_relative"] for c in screen]
    for name, values, digits in (("dsc", dice, 1), ("assd", assd, 0), ("hd95", hd95, 0)):
        KEYS[f"CTX.{name}.low"] = f"{min(values):.{digits}f}"
        KEYS[f"CTX.{name}.high"] = f"{max(values):.{digits}f}"


def training_loss_keys() -> None:
    for dataset in ("isic2018", "acdc", "synapse"):
        old, new = [], []
        for seed in (1, 2, 3):
            old.append(
                pd.read_csv(
                    ROOT / f"outputs/runs/{dataset}_unet_seed{seed}/train_log.csv"
                ).seg_dice.iloc[-1]
            )
            lines = (
                (ROOT / f"results/runs/{dataset}_p1b_unet_seed{seed}/metrics.jsonl")
                .read_text()
                .splitlines()
            )
            new.append(json.loads(lines[-1])["train"]["dice"])
        KEYS[f"TL.{dataset}.S"] = num(float(np.mean(old)), 3, False)
        KEYS[f"TL.{dataset}.M"] = num(float(np.mean(new)), 3, False)


def legacy_bf1_keys() -> None:
    """Historical against present Boundary F1 for the same models and images."""
    corr, gap = [], []
    for dataset in ("acdc", "synapse"):
        for cohort, arm in (("famO_unet", "unet"), ("famO_full", "bds_lite_full")):
            parts = []
            for seed in (1, 2, 3):
                old = pd.read_csv(ROOT / f"outputs/evaluations/{dataset}_{arm}_seed{seed}_test.csv")
                old = old[old.class_id > 0]
                new = pd.read_csv(
                    ROOT / f"results/extra_cohorts/{cohort}/{dataset}_seed{seed}/test/per_case.csv"
                )
                both = old.merge(
                    new,
                    left_on=["sample_id", "class_id"],
                    right_on=["id", "label"],
                    suffixes=("_old", "_new"),
                )
                parts.append(both[~((both.dsc_old >= 0.999999) & (both.hd95_old.fillna(0) == 0))])
            merged = pd.concat(parts)
            corr.append(merged.boundary_f1_old.corr(merged.boundary_f1_new))
            gap.append((merged.boundary_f1_old - merged.boundary_f1_new).abs().mean())
    KEYS["LEG.r.low"], KEYS["LEG.r.high"] = f"{min(corr):.3f}", f"{max(corr):.3f}"
    KEYS["LEG.gap.low"], KEYS["LEG.gap.high"] = f"{min(gap):.3f}", f"{max(gap):.3f}"


def ledger() -> None:
    frame = pd.read_csv(AUDIT / "experiment_ledger.csv", dtype=str).fillna("")
    done = frame[frame.status.isin(["complete", "complete; checkpoints absent"])]
    # Reductions are derived in Section 3.1 of the audit report.
    distinct = len(done) - 28 - 12 - 6
    independent = distinct - 9 - 9 - 3
    KEYS["LED.completed"] = str(len(done))
    KEYS["LED.distinct"] = str(distinct)
    KEYS["LED.independent"] = str(independent)
    order = [
        (
            "single-seed exploration",
            "Apr",
            "Recipe development on validation data",
            "Exploratory",
            "No",
        ),
        (
            "family O reproducibility grid",
            "Apr--May",
            "13-arm grid; U-Net and BDS-Lite are the original set",
            (
                "Validated (U-Net, BDS-Lite); re-evaluated post hoc (4 ablation arms); "
                "unvalidated (7 other arms)"
            )
            if REEVALUATED
            else "Validated (U-Net, BDS-Lite); unvalidated (11 other arms)",
            "Original set; S15",
        ),
        ("p1 gate pilot (single seed)", "May", "Gate and stop-gradient pilot", "Exploratory", "No"),
        (
            "p1b gate-free multi-seed",
            "May",
            "Follow-up family; 18 of these are gate variants of the same checkpoints",
            "Validated",
            "Retrained set",
        ),
        (
            "p2 recent-loss baselines (GSL, Boundary DoU)",
            "May",
            "Earlier GSL and Boundary DoU runs",
            "Unvalidated; superseded",
            "No",
        ),
        (
            "p2b BDS-Lite(no gate)+GSL",
            "May",
            "BDS-Lite path combined with GSL",
            "Unvalidated",
            "No",
        ),
        (
            "EGE-UNet compact baseline",
            "May",
            "Compact comparator, dropped before submission",
            "Unvalidated",
            "No",
        ),
        (
            "Phase 16 matched GSL",
            "Jun",
            "U-Net+GSL rows of the original submission",
            "Validated by exact retraining",
            "Via retrain",
        ),
        (
            "ISIC U/S/B/F factorial",
            "Oct",
            "Development study",
            "Validated (re-evaluated on test)",
            "Four arms reused on ISIC2018",
        ),
        (
            "ISIC pathway controls",
            "Oct",
            "Gate only; gate with decoder",
            EXPLORATORY_RETESTED,
            "S15",
        ),
        (
            "ISIC boundary-specificity control",
            "Oct",
            "Feature matching without boundary supervision",
            EXPLORATORY_RETESTED,
            "S15",
        ),
        (
            "ISIC D4 augmentation baseline",
            "Oct",
            "U-Net with flips and rotations",
            "Exploratory (development images)",
            "S15",
        ),
        (
            "ACDC index-grid region vs paper-GSL",
            "Oct",
            "Development data, 10 patients",
            "Exploratory",
            "No",
        ),
        (
            "ACDC three-index context control",
            "Oct",
            "Three adjacent slices as input; development data",
            "Exploratory",
            "S15",
        ),
        (
            "ACDC phase-reduction control",
            "Oct",
            "Loss reduction unit; development data",
            "Exploratory",
            "No",
        ),
        ("WP2/WP4 queue A", "Nov", "Ablations, Attention U-Net", "Validated (prespecified)", "Yes"),
        ("WP4 queue B", "Nov", "GSL, Boundary Loss, LB-UNet", "Validated (prespecified)", "Yes"),
        (
            "WP2/WP3 Tier B",
            "Nov",
            "Decoder-only arm; $\\alpha$ and $\\lambda$ sensitivity",
            "Validated (prespecified)",
            "Yes",
        ),
        (
            "Stage 1 expansion",
            "Nov",
            "Seeds 4--10 on ISIC2018; decoder plus surface; ACDC correction",
            "Validated (frozen); source of both confirmed contrasts",
            "Yes",
        ),
    ]
    counts = Counter(done.family)
    lines = [
        r"\begin{tabular}{p{4.0cm}lrp{4.2cm}p{3.6cm}p{2.6cm}}",
        r"\toprule",
        r"Family & 2026 & Runs & Purpose & Evidence class & Use here \\",
        r"\midrule",
    ]
    total = 0
    for family, when, purpose, evidence, used in order:
        total += counts[family]
        name = (
            family.replace("&", r"\&")
            .replace("single-seed exploration", "Single-seed exploration")
            .replace("WP2/WP4 queue A", "Prespecified ablations")
            .replace("WP4 queue B", "Prespecified comparators")
            .replace("WP2/WP3 Tier B", "Prespecified sensitivity")
            .replace("Stage 1 expansion", "Frozen additional trainings")
            .replace("family O reproducibility grid", "Original grid")
            .replace("p1b gate-free multi-seed", "Follow-up family")
            .replace("p1 gate pilot (single seed)", "Gate pilot")
            .replace("p2 recent-loss baselines (GSL, Boundary DoU)", "Earlier loss baselines")
            .replace("p2b BDS-Lite(no gate)+GSL", "BDS-Lite path + GSL")
            .replace("Phase 16 matched GSL", "Original GSL")
        )
        lines.append(
            f"{name} & {when} & {counts[family]} & {purpose} & {evidence} & {used} " + r"\\"
        )
    if total != len(done):
        raise ValueError(f"ledger families do not cover all completed runs: {total} of {len(done)}")
    lines += [
        r"\midrule",
        f"Total completed & & {total} & & & " + r"\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (GEN / "tab_s_ledger.tex").write_text("\n".join(lines) + "\n")


def tab_membership() -> None:
    names = [
        ("all", "All images, as originally reported"),
        ("train", "In these models' training set"),
        ("sel", "In their checkpoint-selection set"),
        ("held", "Held out from them"),
        ("clean", "Their own held-out test set"),
    ]
    lines = [
        r"\begin{tabular}{lrcccc}",
        r"\toprule",
        r"Images & $n$ & HD95 (px) & ASSD (px) & Boundary F1 & DSC \\",
        r"\midrule",
    ]
    for key, label in names:
        cells = [
            f"{KEYS[f'LK.{key}.{m}.delta']} {KEYS[f'LK.{key}.{m}.ci']}"
            for m in ("hd95", "assd", "boundary_f1", "dsc")
        ]
        if key == "clean":
            lines.append(r"\midrule")
        lines.append(f"{label} & {KEYS[f'LK.{key}.n']} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_s_membership.tex").write_text("\n".join(lines) + "\n")


def tab_validation(rows: list[dict]) -> None:
    test = pd.read_csv(FAM / "expansion_contrasts.csv")
    names = {"decoder": "Decoder", "unet": "U-Net", "full": "BDS-Lite", "gsl": "GSL"}
    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Contrast & Development images (519) & Official test (1,000) \\",
        r"\midrule",
    ]
    for first, second in ex.PAIRS:
        value = [
            r
            for r in rows
            if r["metric"] == "boundary_f1" and r["first"] == first and r["second"] == second
        ][0]
        frozen = test[
            (test.metric == "boundary_f1") & (test["first"] == first) & (test["second"] == second)
        ].iloc[0]
        left = f"{num(value['delta'])} {interval(value['low'], value['high'])}"
        right = f"{num(frozen.all_delta)} {interval(frozen.all_low, frozen.all_high)}"
        lines.append(f"{names[first]} vs {names[second]} & {left} & {right} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_s_validation.tex").write_text("\n".join(lines) + "\n")


def tab_development(rows: list[dict]) -> None:
    names = {
        ("B", "C"): "Distillation, given decoder and gate",
        ("C", "G"): "Decoder, given gate",
        ("G", "U"): "Gate and projection alone",
        ("D", "G"): "Feature matching without boundary supervision",
        ("B", "D"): "Boundary supervision, given feature matching",
        ("C", "M2"): "Gate without distillation, against decoder",
        ("B", "M2"): "Gate with distillation, against decoder",
        ("UA", "U"): "D4 augmentation of U-Net",
    }
    lines = [r"\begin{tabular}{lcc}", r"\toprule", r"Contrast & Boundary F1 & DSC \\", r"\midrule"]
    for (first, second), label in names.items():
        cells = []
        for metric in ("boundary_f1", "dsc"):
            value = [
                r
                for r in rows
                if r["metric"] == metric and r["first"] == first and r["second"] == second
            ][0]
            cells.append(f"{num(value['delta'])} {interval(value['low'], value['high'])}")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_s_development.tex").write_text("\n".join(lines) + "\n")


def figure() -> None:
    probe = pd.read_csv(FAM / "boundary_transfer_probe.csv")
    groups = {
        "distilled": ("M3", "M5", "M6"),
        "undistilled": ("M2", "M4"),
        "no decoder": ("M0", "GSL"),
    }
    style = {
        "distilled": dict(color=BLUE, marker="o", facecolor=BLUE, offset=0.24),
        "undistilled": dict(color=ORANGE, marker="s", facecolor="white", offset=0.0),
        "no decoder": dict(color=AQUA, marker="D", facecolor="white", offset=-0.24),
    }
    names = {"isic2018": "ISIC2018", "acdc": "ACDC", "synapse": "Synapse"}
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "font.family": "DejaVu Sans",
            "axes.edgecolor": MUTED,
            "axes.linewidth": 0.6,
        }
    )
    figure_, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True, sharex=True)
    panels = [
        (
            axes[0],
            "head_q_s",
            ("distilled", "undistilled"),
            "Frozen teacher head on the deployed projection $q_s$",
        ),
        (
            axes[1],
            "probe_F_s",
            ("distilled", "undistilled", "no decoder"),
            "Linear readout of the prediction feature $F_s$",
        ),
    ]
    for axis, column, shown, title in panels:
        for index, dataset in enumerate(names):
            y = 2 - index
            part = probe[probe.dataset == dataset]
            for group in shown:
                values = part[part.arm.isin(groups[group])][column].dropna().to_numpy()
                s = style[group]
                jitter = np.linspace(-0.07, 0.07, values.size)
                axis.scatter(
                    values,
                    y + s["offset"] + jitter,
                    s=15,
                    marker=s["marker"],
                    facecolor=s["facecolor"],
                    edgecolor=s["color"],
                    linewidth=0.8,
                    alpha=0.75,
                    zorder=3,
                )
                axis.plot(
                    [values.mean()] * 2,
                    [y + s["offset"] - 0.13, y + s["offset"] + 0.13],
                    color=s["color"],
                    linewidth=2.0,
                    solid_capstyle="round",
                    zorder=4,
                )
            if column == "head_q_s":
                teacher = (
                    part[part.arm.isin(groups["distilled"] + groups["undistilled"])]["head_q_b"]
                    .dropna()
                    .mean()
                )
                axis.plot(
                    [teacher],
                    [y],
                    marker="|",
                    markersize=15,
                    markeredgewidth=1.8,
                    color=INK,
                    linestyle="none",
                    zorder=4,
                )
        axis.set_yticks([2, 1, 0], list(names.values()))
        axis.set_ylim(-0.55, 2.55)
        axis.set_xlim(0, 0.6)
        axis.set_xlabel("Boundary average precision", color=MUTED)
        axis.set_title(title, fontsize=8.5, color=INK, loc="left")
        axis.grid(axis="x", color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
        axis.tick_params(colors=MUTED, length=2.5, width=0.6)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
    legend = [
        Line2D(
            [],
            [],
            marker="o",
            color=BLUE,
            markerfacecolor=BLUE,
            linestyle="none",
            markersize=4.5,
            label="Distilled arms",
        ),
        Line2D(
            [],
            [],
            marker="s",
            color=ORANGE,
            markerfacecolor="white",
            linestyle="none",
            markersize=4.5,
            label="Decoder, not distilled",
        ),
        Line2D(
            [],
            [],
            marker="D",
            color=AQUA,
            markerfacecolor="white",
            linestyle="none",
            markersize=4.5,
            label="No decoder",
        ),
        Line2D(
            [],
            [],
            marker="|",
            color=INK,
            linestyle="none",
            markersize=9,
            markeredgewidth=1.8,
            label="Teacher itself",
        ),
    ]
    figure_.legend(
        handles=legend,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.03),
    )
    figure_.tight_layout(rect=(0, 0.07, 1, 1))
    figure_.savefig(FIG / "transfer_mechanism.pdf")
    figure_.savefig(FIG / "transfer_mechanism.png", dpi=200)
    plt.close(figure_)


def historical() -> None:
    """Keys and tables for HISTORICAL_REEVALUATION_PROTOCOL.md; only after its frozen analysis ran."""
    files = [FAM / f"historical_{n}.csv" for n in ("april", "october", "tolerance")]
    if not all(f.exists() for f in files):
        return
    april, october, tolerance = (pd.read_csv(f) for f in files)
    folder = ROOT / "results/historical_reevaluation"
    validation = json.loads((folder / "validation.json").read_text())
    parity = json.loads((folder / "parity.json").read_text())
    KEYS["HR.status"] = validation["status"]
    KEYS["HR.folders"] = str(validation["folders_checked"])
    KEYS["HR.parity.n"] = str(len(parity))
    KEYS["HR.parity.passed"] = str(sum(bool(v["passed"]) for v in parity.values()))
    KEYS["HR.parity.maxdiff"] = f"{max(v['mean_abs_diff'] for v in parity.values()):.4f}"
    KEYS["HR.parity.maxshare"] = f"{100 * max(v['share_rows_beyond'] for v in parity.values()):.1f}"
    names = {
        "isic2018": "ISIC2018 (official)",
        "isic390": "ISIC2018 (clean 390)",
        "acdc": "ACDC",
        "synapse": "Synapse",
        "pooled": "Pooled",
    }
    for row in april.itertuples():
        key = f"H.{row.id}.{row.dataset}.{row.metric}"
        digits = METRIC_DIGITS[row.metric]
        KEYS[f"{key}.delta"] = num(row.delta, digits)
        KEYS[f"{key}.ci"] = interval(row.low, row.high, digits)
        KEYS[f"{key}.label"] = row.classification
    for row in october.itertuples():
        family = "Ds" if row.family == "D-sensitivity" else "D"
        key = f"{family}.{row.id}.{row.metric}"
        digits = METRIC_DIGITS[row.metric]
        KEYS[f"{key}.delta"] = num(row.delta, digits)
        KEYS[f"{key}.ci"] = interval(row.low, row.high, digits)
        KEYS[f"{key}.label"] = row.classification
    labels = {
        "H1": "Decoder vs U-Net",
        "H2": "Decoder + distillation vs decoder",
        "H3": "Decoder + gate vs decoder",
        "H4": "Full vs decoder + gate + surface",
        "H5": "Full vs decoder",
    }
    cols = ("isic2018", "acdc", "synapse", "pooled", "isic390")
    lines = [
        r"\begin{tabular}{l" + "c" * len(cols) + "}",
        r"\toprule",
        "Contrast & " + " & ".join(names[c] for c in cols) + r" \\",
        r"\midrule",
    ]
    bf1 = april[april.metric == "boundary_f1"].set_index(["id", "dataset"])
    for hid, label in labels.items():
        cells = [
            f"{num(bf1.loc[(hid, c), 'delta'])} {interval(bf1.loc[(hid, c), 'low'], bf1.loc[(hid, c), 'high'])}"
            for c in cols
        ]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_s_hist_april.tex").write_text("\n".join(lines) + "\n")
    dl = {
        "D1": "B vs C: distillation given decoder and gate",
        "D2": "C vs G: decoder given gate",
        "D3": "G vs U-Net: gate and projection",
        "D4": "D vs G: matching without boundary supervision",
        "D5": "B vs D: boundary supervision given matching",
        "D6": "C vs decoder: gate without distillation",
        "D7": "B vs decoder: gate with distillation",
        "D8": "Interaction (B$-$C)$-$(D$-$G)",
    }
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Contrast & Boundary F1 & DSC & Holm $p$ (BF1) \\",
        r"\midrule",
    ]
    main = october[october.family == "D"].set_index(["id", "metric"])
    for did, label in dl.items():
        b, d = main.loc[(did, "boundary_f1")], main.loc[(did, "dsc")]
        lines.append(
            f"{label} & {num(b.delta)} {interval(b.low, b.high)} & {num(d.delta)} {interval(d.low, d.high)} & {b.p_holm:.3f} "
            + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_s_hist_october.tex").write_text("\n".join(lines) + "\n")
    pairs = {
        ("decoder", "unet"): "Decoder vs U-Net",
        ("full", "unet"): "BDS-Lite vs U-Net",
        ("full", "decoder"): "BDS-Lite vs decoder",
        ("gsl", "unet"): "GSL vs U-Net",
        ("gsl", "full"): "GSL vs BDS-Lite",
        ("gsl", "decoder"): "GSL vs decoder",
    }
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Contrast & 1 px & 2 px (frozen) & 4 px \\",
        r"\midrule",
    ]
    for (first, second), label in pairs.items():
        cells = []
        for tol in (1, 2, 4):
            a = tolerance[
                (tolerance.tolerance == tol)
                & (tolerance.seeds == "all")
                & (tolerance["first"] == first)
                & (tolerance["second"] == second)
            ].iloc[0]
            f = tolerance[
                (tolerance.tolerance == tol)
                & (tolerance.seeds == "fresh")
                & (tolerance["first"] == first)
                & (tolerance["second"] == second)
            ].iloc[0]
            confirmed = f.p_holm < 0.05 and (f.low > 0 or f.high < 0) and f.delta * a.delta > 0
            status = (
                "confirmed"
                if confirmed
                else ("supported" if (a.low > 0 or a.high < 0) else "not established")
            )
            KEYS[f"T.{tol}.{first}.{second}.delta"] = num(a.delta)
            KEYS[f"T.{tol}.{first}.{second}.status"] = status
            cells.append(f"{num(a.delta)} ({status})")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_s_tolerance.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    validation = validation_contrasts()
    development = development_controls()
    membership()
    context_keys()
    training_loss_keys()
    legacy_bf1_keys()
    ledger()
    tab_membership()
    tab_validation(validation)
    tab_development(development)
    figure()
    historical()
    lines = ["% generated by analysis/revision/make_audit_assets.py; do not edit"]
    lines += [rf"\expandafter\def\csname R@{k}\endcsname{{{v}}}" for k, v in sorted(KEYS.items())]
    (GEN / "numbers_audit.tex").write_text("\n".join(lines) + "\n")
    print(len(KEYS), "keys")


if __name__ == "__main__":
    main()
