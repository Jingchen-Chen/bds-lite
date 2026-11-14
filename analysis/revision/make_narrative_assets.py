# ruff: noqa: E501
"""Table and figure for the central question: what does distillation add to a boundary decoder?

Reads only the frozen outputs revision/tables/families/expansion_interaction.csv and
expansion_contrasts.csv (seeds 1-3 on all datasets, seeds 1-10 on ISIC2018). Nothing is recomputed.

Writes generated/tab_central.tex and revision/figures/central_question.pdf (Fig. 2: a forest plot
with a label column, the intervals and a Delta [95% CI] column formatted as Table 4; drawn at the
text width, 415.13 TeX pt, with the shared settings of analysis/revision/figure_style.py and
included unscaled).

    PYTHONPATH=src:. .venv/bin/python analysis/revision/make_narrative_assets.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import figure_style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from make_distillation_assets import ci, num  # noqa: E402  (the formatter of Table 4)
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Polygon  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FAM = ROOT / "revision/tables/families"
GEN = ROOT / "revision/manuscript/generated"
FIG = ROOT / "revision/figures"
DATASETS = ["isic2018", "acdc", "synapse"]
NAMES = {"isic2018": "ISIC2018", "acdc": "ACDC", "synapse": "Synapse"}
METRICS = [("boundary_f1", "Boundary F1", 3), ("dsc", "DSC", 3), ("hd95", "HD95 (px)", 2)]
EFFECTS = [
    ("decoder without surface", "Decoder vs U-Net"),
    ("distillation+gate without surface", "Transfer, no surface loss"),
    ("distillation+gate with surface", "Transfer, with surface loss"),
]
INK, MUTED, BAND = figure_style.INK, figure_style.MUTED, figure_style.BAND


def rows() -> list[dict]:
    with (FAM / "expansion_interaction.csv").open() as stream:
        return [r for r in csv.DictReader(stream) if r["seeds"] == "1-3"]


def cell(row: dict, digits: int) -> str:
    def fmt(value: str) -> str:
        return f"{float(value):+.{digits}f}".replace("-", "$-$")

    return f"{fmt(row['estimate'])} [{fmt(row['low'])}, {fmt(row['high'])}]"


def pick(data: list[dict], effect: str, metric: str, dataset: str) -> dict:
    found = [
        r
        for r in data
        if r["effect"] == effect and r["metric"] == metric and r["dataset"].startswith(dataset)
    ]
    if len(found) != 1:
        raise ValueError(f"expected one row: {effect} {metric} {dataset}")
    return found[0]


def table(data: list[dict]) -> None:
    lines = [
        r"\begin{tabular}{llllll}",
        r"\toprule",
        r"Metric & Contrast & ISIC2018 & ACDC & Synapse & Pooled (label) \\",
    ]
    for metric, label, digits in METRICS:
        lines.append(r"\midrule")
        for index, (effect, name) in enumerate(EFFECTS):
            cells = [cell(pick(data, effect, metric, d), digits) for d in DATASETS]
            pooled = pick(data, effect, metric, "pooled")
            lines.append(
                f"{label if index == 0 else ''} & {name} & "
                + " & ".join(cells)
                + f" & {cell(pooled, digits)} ({pooled['classification']}) \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_central.tex").write_text("\n".join(lines) + "\n")


def ten_seed(first: str, second: str) -> tuple[float, float, float]:
    with (FAM / "expansion_contrasts.csv").open() as stream:
        for row in csv.DictReader(stream):
            if (row["metric"], row["first"], row["second"]) == ("boundary_f1", first, second):
                return float(row["all_delta"]), float(row["all_low"]), float(row["all_high"])
    raise ValueError(f"missing contrast {first} {second}")


def figure(data: list[dict]) -> None:
    """Fig. 2 as a forest plot in the clinical meta-analysis layout (labels | intervals | values).

    The value column is written with num() and ci() of make_distillation_assets.py, the formatter
    of Table 4 (generated/tab_innetwork_main.tex, written by make_final_assets.py), from the same
    CSV rows, so figure and table show the same digits; only the TeX minus "$-$" is drawn as the
    Unicode minus. STIX digits are tabular and "+" and the minus have the same advance, so the
    right-aligned strings line up at the decimal point.
    """

    def three(effect: str) -> list[tuple]:
        out = []
        for dataset in [*DATASETS, "pooled"]:
            r = pick(data, effect, "boundary_f1", dataset)
            values = (float(r["estimate"]), float(r["low"]), float(r["high"]))
            if dataset == "pooled":
                out.append(("Pooled (3 datasets)", *values, "pooled"))
            else:
                out.append((f"{NAMES[dataset]} (n = 3)", *values, "three"))
        return out

    # Group titles are the contrasts of Table 4 and the text; the second line says what they add.
    groups = [
        (
            "D − U",
            "boundary decoder",
            [
                ("ISIC2018 (n = 10)", *ten_seed("decoder", "unet"), "ten"),
                *three("decoder without surface"),
            ],
        ),
        (
            "D+F+G − D",
            "in-network transfer, no surface loss",
            three("distillation+gate without surface"),
        ),
        (
            "BDS-Lite − D+S",
            "in-network transfer, with surface loss",
            three("distillation+gate with surface"),
        ),
        (
            "BDS-Lite − D",
            "complete recipe (D+F+G+S)",
            [("ISIC2018 (n = 10)", *ten_seed("full", "decoder"), "ten")],
        ),
    ]
    blue, vermilion = figure_style.OKABE_ITO["blue"], figure_style.OKABE_ITO["vermilion"]
    style = {
        "ten": dict(marker="s", color=blue, size=4.0, fill=blue, width=1.2),
        "three": dict(marker="o", color=vermilion, size=4.0, fill="white", width=0.9),
    }

    # Vertical layout in row units (one row = ROW_PT points); y grows downwards.
    title_to_rows, group_gap = 1.85, 0.6
    placed, titles, y = [], [], 0.0
    for title, subtitle, items in groups:
        titles.append((title, subtitle, y))
        y += title_to_rows
        for item in items:
            placed.append((y, *item))
            y += 1
        y += group_gap
    y_top, y_bottom = -0.75, placed[-1][0] + 0.7

    figure_style.apply()
    row_pt, header_pt, bottom_pt = 10.0, 14.0, 41.0
    width = figure_style.W_IN * 72
    plot_h = row_pt * (y_bottom - y_top)
    height = header_pt + plot_h + bottom_pt
    x_title, x_row, ax_left, ax_right, x_value = 0.5, 9.0, 116.0, width - 98.0, width - 0.5
    fig = plt.figure(figsize=(figure_style.W_IN, height / 72))
    ax = fig.add_axes(
        [ax_left / width, bottom_pt / height, (ax_right - ax_left) / width, plot_h / height]
    )
    ax.set_xlim(-0.06, 0.10)
    ax.set_ylim(y_bottom, y_top)
    ax.axvspan(-0.01, 0.01, color=BAND, zorder=0, lw=0)
    ax.axvline(0, color=MUTED, lw=0.6, zorder=1)
    side = blended_transform_factory(fig.transFigure, ax.transData)

    def label(x: float, y: float, text: str, **kw) -> None:
        kw.setdefault("va", "center")
        kw.setdefault("color", INK)
        ax.text(x / width, y, text, transform=side, clip_on=False, **kw)

    def rule(y: float, x0: float, color: str, lw: float) -> None:
        fig.add_artist(Line2D([x0 / width, 1], [y, y], transform=side, color=color, lw=lw))

    header_y = y_top - header_pt / row_pt / 2
    label(x_title, header_y, "Contrast, dataset (n = runs per arm)", fontweight="bold")
    label(x_value, header_y, "Δ [95% CI]", ha="right", fontweight="bold")
    rule(y_top, 0.0, INK, 0.5)
    for title, subtitle, y0 in titles:
        label(x_title, y0, title, fontweight="bold", fontsize=7.5)
        label(x_title, y0 + 0.85, subtitle, color=MUTED, fontsize=7)
    for y0, name, estimate, low, high, kind in placed:
        label(x_row, y0, name)
        value = f"{num(estimate)} {ci(low, high)}".replace("$-$", "−")
        label(x_value, y0, value, ha="right")
        if kind == "pooled":
            rule(y0 - 0.5, x_row, "#9a9a9a", 0.4)
            ax.add_patch(
                Polygon(
                    [(low, y0), (estimate, y0 - 0.34), (high, y0), (estimate, y0 + 0.34)],
                    closed=True,
                    facecolor=INK,
                    edgecolor=INK,
                    lw=0.4,
                    zorder=3,
                )
            )
            continue
        s = style[kind]
        ax.plot(
            [low, high], [y0, y0], color=s["color"], lw=s["width"], zorder=2, solid_capstyle="butt"
        )
        ax.plot(
            estimate,
            y0,
            s["marker"],
            ms=s["size"],
            mfc=s["fill"],
            mec=s["color"],
            mew=0.9,
            zorder=3,
        )

    ax.set_xticks([-0.04, -0.02, 0, 0.02, 0.04, 0.06, 0.08])
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda v, _: "0" if abs(v) < 1e-12 else f"{v:.2f}".replace("-", "−"))
    )
    ax.set_yticks([])
    ax.set_xlabel("Δ Boundary F1 (positive favours the first arm)", labelpad=3)
    ax.tick_params(axis="x", length=2, width=0.4, pad=2)
    for edge in ("top", "right", "left"):
        ax.spines[edge].set_visible(False)
    diamond = [(-1.6, 0), (0, 0.6), (1.6, 0), (0, -0.6), (-1.6, 0)]
    handles = [
        Line2D(
            [], [], marker="s", color=blue, mfc=blue, ms=4.0, lw=1.2, label="n = 10 runs per arm"
        ),
        Line2D(
            [],
            [],
            marker="o",
            color=vermilion,
            mfc="white",
            ms=4.0,
            lw=0.9,
            label="n = 3 runs per arm",
        ),
        Line2D(
            [],
            [],
            marker=diamond,
            color=INK,
            mfc=INK,
            ms=8.0,
            lw=0,
            label="Pooled, fixed effect (width = 95% CI)",
        ),
        Patch(facecolor=BAND, edgecolor="none", label="SESOI (±0.01)"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=4,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.5,
        borderaxespad=0.15,
    )
    fig.savefig(FIG / "central_question.pdf", metadata={"CreationDate": None})
    fig.savefig(FIG / "central_question.png", dpi=300)


def main() -> None:
    data = rows()
    table(data)
    figure(data)


if __name__ == "__main__":
    main()
