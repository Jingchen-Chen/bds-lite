# ruff: noqa: E501
"""Keys, tables and the ten-seed figure for the expansion and confirmation results.

Reads the frozen outputs in revision/tables/families/ and writes generated/numbers_expansion.tex,
tab_ten_seed.tex, tab_ten_seed_contrasts.tex, tab_components.tex, tab_s_ten_seed_all.tex,
tab_s_interaction.tex, tab_s_seed_subset.tex and revision/figures/ten_seed.pdf. Confirmation keys and
tables are written only when confirmation_contrasts.csv exists.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis/revision"))
import expansion_statistics as ex  # noqa: E402

FAM = ROOT / "revision/tables/families"
GEN = ROOT / "revision/manuscript/generated"
FIG = ROOT / "revision/figures"
METRICS = ["dsc", "iou", "boundary_f1", "hd95", "assd"]
MLABEL = {
    "dsc": "DSC",
    "iou": "IoU",
    "boundary_f1": "Boundary F1",
    "hd95": "HD95 (px)",
    "assd": "ASSD (px)",
}
ARM = {"unet": "U-Net", "decoder": "U-Net + decoder", "full": "BDS-Lite", "gsl": "U-Net + GSL"}
DS = {"isic2018": "ISIC2018", "acdc": "ACDC", "synapse": "Synapse"}
KEYS: dict[str, str] = {}


def read(name: str) -> list[dict]:
    with (FAM / name).open() as stream:
        return list(csv.DictReader(stream))


def f(value, digits: int = 3, sign: bool = True) -> str:
    text = f"{float(value):+.{digits}f}" if sign else f"{float(value):.{digits}f}"
    return text.replace("-", "$-$")


def d(metric: str) -> int:
    return 2 if metric in ("hd95", "assd") else 3


def ci(low, high, digits: int) -> str:
    return f"[{f(low, digits)}, {f(high, digits)}]"


def pval(p) -> str:
    p = float(p)
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def slug(text: str) -> str:
    return text.replace(" ", "-").replace("+", "and")


def contrasts() -> list[dict]:
    rows = read("expansion_contrasts.csv")
    for r in rows:
        k = f"X.{r['first']}.{r['second']}.{r['metric']}"
        n = d(r["metric"])
        KEYS[f"{k}.all.delta"] = f(r["all_delta"], n)
        KEYS[f"{k}.all.ci"] = ci(r["all_low"], r["all_high"], n)
        KEYS[f"{k}.all.label"] = r["all_classification"]
        KEYS[f"{k}.fresh.delta"] = f(r["fresh_delta"], n)
        KEYS[f"{k}.fresh.ci"] = ci(r["fresh_low"], r["fresh_high"], n)
        KEYS[f"{k}.fresh.holm"] = pval(r["fresh_p_holm"])
        KEYS[f"{k}.fresh.label"] = r["fresh_classification"]
        KEYS[f"{k}.mw"] = pval(r["fresh_mann_whitney_p"])
        KEYS[f"{k}.status"] = r["status"].replace(
            "supported by all seeds, not confirmed by fresh seeds", "supported, not confirmed"
        )
        if r["first"] == "decoder" and r["second"] == "unet":
            KEYS[f"XM.decoder.{r['metric']}.mean"] = f(r["all_arm_a_mean"], n, False)
            KEYS[f"XM.decoder.{r['metric']}.sd"] = f(
                r["all_arm_a_sd"], n + 1 if n == 3 else n, False
            )
            KEYS[f"XM.unet.{r['metric']}.mean"] = f(r["all_arm_b_mean"], n, False)
            KEYS[f"XM.unet.{r['metric']}.sd"] = f(r["all_arm_b_sd"], n + 1 if n == 3 else n, False)
        if r["first"] == "gsl" and r["second"] == "full":
            KEYS[f"XM.gsl.{r['metric']}.mean"] = f(r["all_arm_a_mean"], n, False)
            KEYS[f"XM.gsl.{r['metric']}.sd"] = f(r["all_arm_a_sd"], n + 1 if n == 3 else n, False)
            KEYS[f"XM.full.{r['metric']}.mean"] = f(r["all_arm_b_mean"], n, False)
            KEYS[f"XM.full.{r['metric']}.sd"] = f(r["all_arm_b_sd"], n + 1 if n == 3 else n, False)
    return rows


def tab_means() -> None:
    lines = [
        r"\begin{tabular}{l" + "c" * len(METRICS) + "}",
        r"\toprule",
        "Arm (10 runs) & " + " & ".join(MLABEL[m] for m in METRICS) + r" \\",
        r"\midrule",
    ]
    for arm, name in ARM.items():
        cells = [f"{KEYS[f'XM.{arm}.{m}.mean']} $\\pm$ {KEYS[f'XM.{arm}.{m}.sd']}" for m in METRICS]
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_ten_seed.tex").write_text("\n".join(lines) + "\n")


def tab_contrasts(rows: list[dict]) -> None:
    def table(metrics: list[str]) -> str:
        lines = [
            r"\begin{tabular}{llccccl}",
            r"\toprule",
            r"Metric & Contrast & Fresh $\Delta$ (seeds 4--10) & Holm $p$ & All-seed $\Delta$ [95\% interval] (seeds 1--10) & Effect label & Status \\",
            r"\midrule",
        ]
        for m in metrics:
            for r in [x for x in rows if x["metric"] == m]:
                k = f"X.{r['first']}.{r['second']}.{m}"
                lines.append(
                    f"{MLABEL[m]} & {ARM[r['first']]} vs {ARM[r['second']]} & {KEYS[k + '.fresh.delta']} & {KEYS[k + '.fresh.holm']} & {KEYS[k + '.all.delta']} {KEYS[k + '.all.ci']} & {KEYS[k + '.all.label']} & {KEYS[k + '.status']} \\\\"
                )
            if m != metrics[-1]:
                lines.append(r"\midrule")
        return "\n".join([*lines, r"\bottomrule", r"\end{tabular}"]) + "\n"

    (GEN / "tab_ten_seed_contrasts.tex").write_text(table(["boundary_f1"]))
    (GEN / "tab_s_ten_seed_all.tex").write_text(table(METRICS))


def components() -> None:
    rows = read("expansion_components.csv")
    names = {
        "decoder": "Boundary decoder",
        "surface": "Surface loss",
        "distill": "Distillation",
        "gate": "Gate",
    }
    for r in rows:
        ds = "pooled" if r["dataset"].startswith("pooled") else r["dataset"]
        n = d(r["metric"])
        k = f"K.{ds}.{r['metric']}.{r['component']}"
        KEYS[f"{k}.est"] = f(r["estimate"], n)
        KEYS[f"{k}.ci"] = ci(r["low"], r["high"], n)
        KEYS[f"{k}.label"] = r["classification"]
        if ds == "pooled":
            KEYS[f"{k}.q"] = r["dataset"].split("=")[1].rstrip(")")
    lines = [
        r"\begin{tabular}{llllll}",
        r"\toprule",
        "Metric & Component & ISIC2018 & ACDC & Synapse & Pooled (label) \\\\",
        r"\midrule",
    ]
    for m in ("boundary_f1", "dsc", "hd95"):
        for c, name in names.items():

            def cell(ds: str, m: str = m, c: str = c) -> str:
                return f"{KEYS[f'K.{ds}.{m}.{c}.est']} {KEYS[f'K.{ds}.{m}.{c}.ci']}"

            lines.append(
                f"{MLABEL[m]} & {name} & {cell('isic2018')} & {cell('acdc')} & {cell('synapse')} & {cell('pooled')} ({KEYS[f'K.pooled.{m}.{c}.label']}) \\\\"
            )
        if m != "hd95":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_components.tex").write_text("\n".join(lines) + "\n")


def interaction() -> None:
    rows = read("expansion_interaction.csv")
    shown = []
    for r in rows:
        ds = "pooled" if r["dataset"].startswith("pooled") else r["dataset"]
        ten = "ten" if "1-10" in r["seeds"] else "three"
        n = d(r["metric"])
        k = f"I.{slug(r['effect'])}.{ds}.{r['metric']}.{ten}"
        KEYS[f"{k}.est"] = f(r["estimate"], n)
        KEYS[f"{k}.ci"] = ci(r["low"], r["high"], n)
        KEYS[f"{k}.label"] = r["classification"]
        if ds == "pooled":
            KEYS[f"{k}.q"] = r["dataset"].split("=")[1].rstrip(")")
        if r["metric"] == "boundary_f1":
            shown.append((r, ds, ten, k))
    lines = [
        r"\begin{tabular}{lllll}",
        r"\toprule",
        r"Effect (Boundary F1) & ISIC2018 & ACDC & Synapse & Pooled \\",
        r"\midrule",
    ]
    for effect in [
        "interaction",
        "surface without decoder",
        "surface with decoder",
        "decoder without surface",
        "decoder with surface",
        "distillation+gate without surface",
        "distillation+gate with surface",
        "distillation+gate interaction with surface",
    ]:
        cells = []
        for ds in ("isic2018", "acdc", "synapse", "pooled"):
            k = f"I.{slug(effect)}.{ds}.boundary_f1.three"
            cells.append(f"{KEYS[k + '.est']} {KEYS[k + '.ci']}")
        lines.append(effect[0].upper() + effect[1:] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_s_interaction.tex").write_text("\n".join(lines) + "\n")


def seed_subsets() -> None:
    path = FAM / "seed_subset.csv"
    if not path.exists():
        return
    rows = read("seed_subset.csv")
    pct = lambda x: f"{100 * float(x):.0f}"  # noqa: E731
    for r in rows:
        n = d(r["metric"])
        base = f"SS.{r['kind']}.{r['first']}.{r['second']}.{r['k']}.{r['metric']}"
        KEYS[f"{base}.p05"] = pct(r["share_p05"])
        KEYS[f"{base}.beyond"] = pct(r["share_beyond_sesoi"])
        KEYS[f"{base}.abs"] = f(r["mean_abs_delta"], n, False)
        KEYS[f"{base}.range"] = ci(r["low"], r["high"], n)
        if r["kind"] == "contrast":
            KEYS[f"{base}.same"] = pct(r["share_same_sign"])
            KEYS[f"{base}.opp"] = pct(r["share_opposite_p05"])
    same = [r for r in rows if r["kind"] == "same-recipe"]
    for k in ("3", "5"):
        for m in ("boundary_f1", "dsc", "hd95"):
            sub = [r for r in same if r["k"] == k and r["metric"] == m]
            KEYS[f"SS.same.mean.{k}.{m}.p05"] = pct(np.mean([float(r["share_p05"]) for r in sub]))
            KEYS[f"SS.same.mean.{k}.{m}.beyond"] = pct(
                np.mean([float(r["share_beyond_sesoi"]) for r in sub])
            )
            KEYS[f"SS.same.min.{k}.{m}.p05"] = pct(min(float(r["share_p05"]) for r in sub))
            KEYS[f"SS.same.max.{k}.{m}.p05"] = pct(max(float(r["share_p05"]) for r in sub))
    lines = [
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Recipe & Runs per group & \multicolumn{3}{c}{Comparisons with Wilcoxon $p<0.05$ (\%)} \\",
        r" & & Boundary F1 & DSC & HD95 \\",
        r"\midrule",
    ]
    for arm, name in ARM.items():
        for k in ("3", "5"):
            cells = [
                KEYS[f"SS.same-recipe.{arm}.{arm}.{k}.{m}.p05"]
                for m in ("boundary_f1", "dsc", "hd95")
            ]
            lines.append(f"{name} & {k} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_s_seed_subset.tex").write_text("\n".join(lines) + "\n")


def precision_recall() -> None:
    path = FAM / "boundary_precision_recall.csv"
    if not path.exists():
        return
    rows = read("boundary_precision_recall.csv")
    arms = sorted({r["arm"] for r in rows})
    value = {
        a: {
            m: float(np.mean([float(r[m]) for r in rows if r["arm"] == a]))
            for m in ("precision", "recall", "spurious")
        }
        for a in arms
    }
    for a in arms:
        KEYS[f"BP.{a}.precision"] = f(value[a]["precision"], 3, False)
        KEYS[f"BP.{a}.recall"] = f(value[a]["recall"], 3, False)
        KEYS[f"BP.{a}.spurious"] = f(value[a]["spurious"], 2, False)
    for a in arms:
        if a != "M0":
            KEYS[f"BP.{a}.M0.precision"] = f(value[a]["precision"] - value["M0"]["precision"])
            KEYS[f"BP.{a}.M0.recall"] = f(value[a]["recall"] - value["M0"]["recall"])


def mechanism() -> None:
    from scipy import stats

    rows = read("boundary_transfer_probe.csv")
    distilled, plain, none = ("M3", "M5", "M6"), ("M2", "M4"), ("M0", "GSL")
    for ds in DS:
        sub = [r for r in rows if r["dataset"] == ds]
        col = lambda arms, key, sub=sub: np.array([float(r[key]) for r in sub if r["arm"] in arms])  # noqa: E731
        head_d, head_u = col(distilled, "head_q_s"), col(plain, "head_q_s")
        KEYS[f"MECH.{ds}.head.dist"] = f(head_d.mean(), 3, False)
        KEYS[f"MECH.{ds}.head.plain"] = f(head_u.mean(), 3, False)
        KEYS[f"MECH.{ds}.head.ratio"] = f"{head_d.mean() / head_u.mean():.1f}"
        KEYS[f"MECH.{ds}.head.p"] = pval(stats.ttest_ind(head_d, head_u, equal_var=False).pvalue)
        KEYS[f"MECH.{ds}.teacher"] = f(col(distilled + plain, "head_q_b").mean(), 3, False)
        fid = col(distilled, "fidelity")
        KEYS[f"MECH.{ds}.r2.low"], KEYS[f"MECH.{ds}.r2.high"] = (
            f(fid.min(), 2, False),
            f(fid.max(), 2, False),
        )
        fd, fu = col(distilled, "probe_F_s"), col(plain, "probe_F_s")
        var = fd.var(ddof=1) / fd.size + fu.var(ddof=1) / fu.size
        df = var**2 / (
            (fd.var(ddof=1) / fd.size) ** 2 / (fd.size - 1)
            + (fu.var(ddof=1) / fu.size) ** 2 / (fu.size - 1)
        )
        half = stats.t.ppf(0.975, df) * var**0.5
        KEYS[f"MECH.{ds}.fs.diff"] = f(fd.mean() - fu.mean())
        KEYS[f"MECH.{ds}.fs.ci"] = ci(fd.mean() - fu.mean() - half, fd.mean() - fu.mean() + half, 3)
        KEYS[f"MECH.{ds}.fs.none"] = f(col(none, "probe_F_s").mean(), 3, False)
        band = col(("M6",), "gate_band").mean() / col(("M6",), "gate_off").mean()
        KEYS[f"MECH.{ds}.gate.band"] = f"{band:.2f}"
    path = FAM / "gate_structure.csv"
    if path.exists():
        gate = read("gate_structure.csv")
        shares = []
        for ds in DS:
            sub = [r for r in gate if r["dataset"] == ds]
            mean = lambda key, sub=sub: float(np.mean([float(r[key]) for r in sub]))  # noqa: E731
            KEYS[f"GATE.{ds}.abs"] = f(mean("mean_abs"), 3, False)
            KEYS[f"GATE.{ds}.const"] = f(mean("constant"), 3, False)
            KEYS[f"GATE.{ds}.bound"] = f"{100 * mean('mean_abs') / 0.125:.0f}"
            shares.append(mean("labels_changed"))
        KEYS["GATE.labels.max"] = f"{100 * max(shares):.2f}"


def tab_mechanism() -> None:
    label = {
        "head.dist": "Teacher head on deployed $q_s$, distilled arms (AP)",
        "head.plain": "Teacher head on deployed $q_s$, undistilled arms (AP)",
        "teacher": "Teacher decoder itself (AP)",
        "fs.diff": "Readout of $F_s$, distilled minus undistilled (AP)",
        "fs.ci": "\\quad 95\\% interval",
        "fs.none": "Readout of $F_s$, arms without a decoder (AP)",
        "gate.band": "Gate modulation, band / off-band",
    }
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Measure & ISIC2018 & ACDC & Synapse \\",
        r"\midrule",
    ]
    for key, name in label.items():
        lines.append(name + " & " + " & ".join(KEYS[f"MECH.{ds}.{key}"] for ds in DS) + r" \\")
    lines.append(
        r"$R^2$ of $q_s$ for the teacher feature, distilled arms & "
        + " & ".join(f"{KEYS[f'MECH.{ds}.r2.low']} to {KEYS[f'MECH.{ds}.r2.high']}" for ds in DS)
        + r" \\"
    )
    if "GATE.acdc.abs" in KEYS:
        lines.append(
            r"Gate: mean $|$factor$-1|$ & "
            + " & ".join(KEYS[f"GATE.{ds}.abs"] for ds in DS)
            + r" \\"
        )
        lines.append(
            r"Gate: constant per-channel part & "
            + " & ".join(KEYS[f"GATE.{ds}.const"] for ds in DS)
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_s_mechanism.tex").write_text("\n".join(lines) + "\n")


def figure() -> None:
    order = ["unet", "decoder", "full", "gsl"]
    colors = {"unet": "#52514e", "decoder": "#2a78d6", "full": "#eb6834", "gsl": "#1baf7a"}
    markers = {"unet": "D", "decoder": "o", "full": "s", "gsl": "^"}
    metrics = [("boundary_f1", "Boundary F1"), ("dsc", "DSC"), ("hd95", "HD95 (px)")]
    plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "axes.linewidth": 0.5})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 1.9), sharey=True)
    runs = {
        a: [ex.unit_values(ex.isic_path(a, s), "isic2018") for s in range(1, 11)] for a in order
    }
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        for y, arm in enumerate(reversed(order)):
            values = np.array([np.mean(list(r[metric].values())) for r in runs[arm]])
            for seeds, fill in ((slice(0, 3), "none"), (slice(3, 10), colors[arm])):
                ax.scatter(
                    values[seeds],
                    [y] * len(values[seeds]),
                    s=13,
                    marker=markers[arm],
                    facecolors=fill,
                    edgecolors=colors[arm],
                    linewidths=0.6,
                    zorder=3,
                )
            ax.plot([values.mean()] * 2, [y - 0.33, y + 0.33], color=colors[arm], lw=1.3, zorder=3)
        ax.set_xlabel(title, fontsize=7, labelpad=1.5)
        ax.tick_params(length=2, width=0.4, labelsize=6.5)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4, prune="both"))
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if metric == "hd95":
            ax.invert_xaxis()
    axes[0].set_yticks(range(4), [ARM[a] for a in reversed(order)])
    axes[0].set_ylim(-0.6, 3.6)
    fig.subplots_adjust(left=0.135, right=0.99, top=0.97, bottom=0.2, wspace=0.1)
    fig.savefig(FIG / "ten_seed.pdf")
    fig.savefig(FIG / "ten_seed.png", dpi=300)


def main() -> None:
    rows = contrasts()
    tab_means()
    tab_contrasts(rows)
    components()
    interaction()
    seed_subsets()
    precision_recall()
    mechanism()
    tab_mechanism()
    figure()
    lines = ["% generated by analysis/revision/make_expansion_assets.py; do not edit"]
    lines += [f"\\expandafter\\def\\csname R@{k}\\endcsname{{{KEYS[k]}}}" for k in sorted(KEYS)]
    (GEN / "numbers_expansion.tex").write_text("\n".join(lines) + "\n")
    (GEN / "keys_expansion.json").write_text(json.dumps(sorted(KEYS), indent=0) + "\n")
    print(len(KEYS), "keys")


if __name__ == "__main__":
    main()
