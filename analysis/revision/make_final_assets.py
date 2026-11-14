# ruff: noqa: E501
"""Keys, main-text tables and Fig. 3 for the final revision (plan: revision/plan/figures_tables.md).

Reads frozen analysis outputs only; computes no new statistic except arm means of saved per-case
scores for the ISIC2018 ten-run table (each run's own defined images, as in the evidence freeze
section 8.2) and the percentage changes of the saved resource profile.

Inputs (read-only):
  revision/tables/families/{method_summary,expansion_interaction,expansion_contrasts}.csv
  revision/kd_unified/contrasts.csv                      (original exploratory analysis, freeze 10.1)
  revision/acdc_confirm/results/{results.json,per_seed.csv}
  revision/followup/results/{contrasts.csv,per_seed.csv}  (freeze 10.2-10.3)
  revision/evidence_freeze/final_e3_{analysis,per_seed,per_class}.csv  (freeze 10.4)
  revision/tables/{resource_profile,gate_identity_effects}.csv; revision/kd_unified/costs.csv
  results/acdc_confirm_eval_posthoc/teacher_cache_timing.json
  revision/manuscript/generated/tab_lit_main.tex          (only to drop its boldface)
  per-case ISIC2018 files through analysis/revision/expansion_statistics.isic_path
Writes revision/manuscript/generated/{numbers_final.tex,keys_final.json,tab_matched_main.tex,
tab_innetwork_main.tex,tab_acdc_official_main.tex,tab_resource_final.tex,tab_s_e3_all.tex,
tab_s_e3_perseed.tex,tab_s_e3_perclass.tex,tab_ten_seed_final.tex,tab_lit_main_plain.tex}
and revision/figures/kd_acdc.{pdf,png}.

    PYTHONPATH=src:. .venv/bin/python analysis/revision/make_final_assets.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis/revision"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
import figure_style  # noqa: E402
from make_distillation_assets import ci, num, pfmt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

GEN = ROOT / "revision/manuscript/generated"
FIG = ROOT / "revision/figures"
FAM = ROOT / "revision/tables/families"
FU = ROOT / "revision/followup/results"
E3 = ROOT / "revision/evidence_freeze"
KEYS: dict[str, str] = {}
METRICS = ("boundary_f1", "dsc", "iou", "hd95", "assd")
MLABEL = {
    "dsc": "DSC",
    "iou": "IoU",
    "boundary_f1": "Boundary F1",
    "hd95": "HD95 (px)",
    "assd": "ASSD (px)",
}
DATASETS = (("isic2018", "ISIC2018"), ("acdc", "ACDC"), ("synapse", "Synapse"))
FULL_COLOR, PEER_COLOR = "#1baa8c", "#9b5fc0"
KD_ARM = r"D+KD-band"
PEER = r"D+KD-band$_{\mathrm{cmp}}$"  # compact-teacher arm; result-file id kd_band_peer
AUG = r"$_{\mathrm{aug}}$"

# Core design count (evidence freeze section 3): reused references count once.
DESIGN_SLOTS = {
    "matched ablation": 72,
    "matched comparators": 36,
    "ISIC2018 extension": 28,
    "exploratory KD students": 9,
    "external teachers": 3,
    "replication students": 34,
    "control students (whole-image, compact teacher)": 34,
    "compact teacher": 1,
    "augmentation-control students": 20,
}


def put(key: str, value: str) -> None:
    if key in KEYS and KEYS[key] != value:
        raise ValueError(f"key {key} defined twice with different values")
    KEYS[key] = value


def d(metric: str) -> int:
    return 2 if metric in ("hd95", "assd") else 4


def read_csv(path: Path) -> list[dict]:
    with path.open() as stream:
        return list(csv.DictReader(stream))


def one(rows: list[dict], **match) -> dict:
    found = [r for r in rows if all(r[k] == v for k, v in match.items())]
    if len(found) != 1:
        raise ValueError(f"expected one row for {match}, found {len(found)}")
    return found[0]


def write(name: str, lines: list[str]) -> None:
    (GEN / name).write_text("\n".join(lines) + "\n")


def holm_reading(reading: str, p_holm: float | None) -> str:
    """Reading of a contrast inside a Holm family: the interval reading only if Holm p < 0.05."""
    if p_holm is None:
        return reading
    return reading if p_holm < 0.05 else "unresolved"


def ndash(a: float, b: float, digits: int = 4) -> str:
    """Interval for running text: 'a to b' with signs only where negative."""

    def f(x: float) -> str:
        return f"{x:.{digits}f}".replace("-", "$-$")

    return f"{f(a)} to {f(b)}"


# ---------------------------------------------------------------- T3 matched absolute scores
def matched_table() -> None:
    rows = read_csv(FAM / "method_summary.csv")
    # Two row groups per dataset: the arms of this study, then the reference methods.
    groups = [
        [("M0", "U"), ("M2", "D"), ("M3", "D+F+G"), ("M6", "BDS-Lite")],
        [("GSL", "U-Net+GSL"), ("C1", "U-Net+Boundary Loss"), ("C3", "Attention U-Net")],
    ]
    # The adapted LB-UNet is reported in Supplementary S7 (lb_adapted_scores.tex); its keys are
    # still computed here, but it has no row in the lean main-text table.
    for ds, _ in DATASETS:
        for code in ("M0", "M2", "M3", "M6", "GSL", "C1", "C3", "C4"):
            r = one(rows, dataset=ds, method=code)
            for m in ("dsc", "boundary_f1", "hd95"):
                put(f"MT.{ds}.{code}.{m}", f"{float(r[f'{m}_mean']):.{d(m)}f}")
    # siunitx S columns parse "mean +- sd" and align both parts on the decimal marker
    # (uncertainty-mode = separate, set in main.tex). Digits are fixed here, as before; siunitx
    # does not round (round-mode = none). No best-value bold: run-to-run SDs are of the same order
    # as the differences. The reference arm D is shaded instead.
    lines = [
        # No @{} at the outer edges, so that the shaded row spans exactly the width of the rules.
        r"\begin{tabular}{l S[table-format=1.4(3)] S[table-format=1.4(3)] S[table-format=2.2(3)]}",
        r"\toprule",
        r"Model & {DSC $\uparrow$} & {Boundary F1 $\uparrow$} & {HD95 (px) $\downarrow$} \\",
        r"& \multicolumn{3}{c}{\footnotesize mean $\pm$ SD over three runs} \\",
        r"\midrule",
    ]
    for i, (ds, name) in enumerate(DATASETS):
        if i:
            lines.append(r"\addlinespace[0.6em]")
        lines.append(rf"\multicolumn{{4}}{{l}}{{\textit{{{name}}}}} \\")
        for g, arms in enumerate(groups):
            if g:
                lines.append(r"\cmidrule(l{0.6em}){1-4}")
            for code, label in arms:
                r = one(rows, dataset=ds, method=code)
                cells = []
                for m in ("dsc", "boundary_f1", "hd95"):
                    mean, sd = float(r[f"{m}_mean"]), float(r[f"{m}_sd"])
                    cells.append(f"{mean:.{d(m)}f} +- {sd:.{d(m)}f}")
                shade = r"\rowcolor{refshade}" if code == "M2" else ""
                lines.append(f"{shade}\\quad {label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_matched_main.tex", lines)


# ---------------------------------------------------------------- T4 in-network contrasts
# Contrast names use a true minus with binary-operator spacing (equal space on both sides).
TEX_MINUS = r"${}-{}$"
INNET = [
    ("2x2", "decoder without surface", f"D{TEX_MINUS}U", "dec"),
    ("transfer", "distillation+gate without surface", f"D+F+G{TEX_MINUS}D", "nosurf"),
    ("transfer", "distillation+gate with surface", f"BDS-Lite{TEX_MINUS}D+S", "surf"),
    ("2x2", "surface without decoder", f"U+S{TEX_MINUS}U", "s.nodec"),
    ("2x2", "surface with decoder", f"D+S{TEX_MINUS}D", "s.dec"),
]
TEN = [
    ("decoder", "unet", f"D{TEX_MINUS}U"),
    ("full", "unet", f"BDS-Lite{TEX_MINUS}U"),
    ("full", "decoder", f"BDS-Lite{TEX_MINUS}D"),
    ("gsl", "unet", f"U-Net+GSL{TEX_MINUS}U"),
    ("gsl", "full", f"U-Net+GSL{TEX_MINUS}BDS-Lite"),
    ("gsl", "decoder", f"U-Net+GSL{TEX_MINUS}D"),
]
# Fixed vocabularies, defined in the table notes with the wording of Statistics.
STATUS_TEX = {
    "confirmed": r"\textbf{confirmed}",
    "supported by all seeds, not confirmed by fresh seeds": "supported, not confirmed",
    "not established": "not established",
}
READING_TEX = {
    "gain, size unclear": "gain of unclear size",
    "harm, size unclear": "deficit of unclear size",
    "no relevant effect": "no relevant effect",
    "undetermined": "undetermined",
    "unresolved": "unresolved",  # Holm family, adjusted p >= 0.05 (Statistics)
}
# Row order of Table 5 and Fig. 3(a): replication, controls C1-C4, augmentation.
ACDC_ORDER = ("S3", "C1", "C2", "C3", "C4", "E3")


def s_in(value: float, digits: int = 4) -> str:
    """Signed number as siunitx S-column input; the same digits as num()."""
    return f"{float(value):+.{digits}f}"


def innetwork_table() -> None:
    """Table 4 as two panels, each its own tabular in a threeparttable with its own notes.

    A: per-dataset and pooled Delta on one line, its 95% CI on the line below. B: one line per
    contrast, numbers in S columns aligned on the decimal marker and the sign.
    """
    rows = read_csv(FAM / "expansion_interaction.csv")
    signed = r"S[table-format=+1.4, retain-explicit-plus]"
    lines = [
        "% generated by analysis/revision/make_final_assets.py::innetwork_table; do not edit",
        r"\footnotesize\setlength{\tabcolsep}{4pt}",
        r"\begin{threeparttable}",
        # Extra space before the pooled column separates it from the per-dataset columns.
        rf"\begin{{tabular}}{{@{{}}l {signed} {signed} {signed} @{{\hspace{{2\tabcolsep}}\hspace{{1em}}}}"
        rf"{signed} >{{\raggedright\arraybackslash}}p{{1.55cm}}@{{}}}}",
        r"\multicolumn{6}{@{}l}{\normalsize\textbf{A}\enspace Three runs per arm on each dataset} \\",
        r"\addlinespace[2pt]",
        r"\toprule",
        r"Contrast & {ISIC2018} & {ACDC} & {Synapse} & {Pooled} & Reading \\",
        r"& \multicolumn{4}{c}{$\Delta$ Boundary F1, with its 95\% CI below} & (pooled) \\",
        r"\midrule",
    ]
    for setname, effect, label, tag in INNET:
        cells = []
        for metric in ("boundary_f1", "dsc", "hd95"):
            picked = [
                r
                for r in rows
                if r["set"] == setname
                and r["effect"] == effect
                and r["metric"] == metric
                and r["seeds"] == "1-3"
            ]
            for r in picked:
                ds = "pooled" if r["dataset"].startswith("pooled") else r["dataset"]
                key = f"IN.{tag}.{ds}.{metric}"
                put(key + ".est", num(float(r["estimate"]), d(metric)))
                put(key + ".ci", ci(float(r["low"]), float(r["high"]), d(metric)))
                put(key + ".text", ndash(float(r["low"]), float(r["high"]), d(metric)))
                put(key + ".class", r["classification"])
        cis = []
        for ds in ("isic2018", "acdc", "synapse", "pooled"):
            r = next(
                x
                for x in rows
                if x["set"] == setname
                and x["effect"] == effect
                and x["metric"] == "boundary_f1"
                and x["seeds"] == "1-3"
                and (x["dataset"].startswith("pooled") if ds == "pooled" else x["dataset"] == ds)
            )
            # The ACDC share of the pooled weight is a note on the pooled cell it describes.
            mark = r"\tnote{a}" if (tag, ds) == ("nosurf", "pooled") else ""
            cells.append(s_in(r["estimate"]) + mark)
            cis.append("{" + ci(float(r["low"]), float(r["high"])) + "}")
            if ds == "pooled":
                reading = READING_TEX[r["classification"]]
        lines.append(
            f"{label} & " + " & ".join(cells) + rf" & \multirow[t]{{2}}{{=}}{{{reading}}} \\"
        )
        lines.append("& " + " & ".join(cis) + r" & \\")
        lines.append(r"\addlinespace[3pt]")
    lines[-1] = r"\bottomrule"
    lines += [
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]\footnotesize\raggedright\hyphenpenalty=10000",
        r"\item[a] ACDC carries \R{PW.nosurf.acdc}\% of the inverse-variance weight of this pooled estimate.",
        r"\item[] Reading of the pooled interval against the 0.01 SESOI: gain of unclear size, excludes"
        r" zero but overlaps the SESOI; undetermined, includes zero and extends beyond $\pm$SESOI.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        "",
        r"\bigskip",
        "",
        r"\setlength{\tabcolsep}{2.5pt}",  # panel B carries two intervals
        r"\begin{threeparttable}",
        rf"\begin{{tabular}}{{@{{}}l {signed} @{{\hspace{{2\tabcolsep}}[}}{signed}@{{,\enspace}}{signed}@{{]\hspace{{2\tabcolsep}}}}"
        rf"{signed} @{{\hspace{{2\tabcolsep}}[}}{signed}@{{,\enspace}}{signed}@{{]\hspace{{2\tabcolsep}}}}"
        rf"S[table-format=1.3] l@{{}}}}",
        r"\multicolumn{9}{@{}l}{\normalsize\textbf{B}\enspace ISIC2018 official test, ten runs per arm} \\",
        r"\addlinespace[2pt]",
        r"\toprule",
        r"& \multicolumn{3}{c}{All seeds (1--10)} & \multicolumn{4}{c}{Fresh seeds (4--10)} & \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-8}",
        # \multicolumn drops the "[" separator, which belongs to data rows only.
        r"Contrast & \multicolumn{1}{c@{\hspace{2\tabcolsep}}}{$\Delta$} & \multicolumn{2}{c}{95\% CI}"
        r" & \multicolumn{1}{c@{\hspace{2\tabcolsep}}}{$\Delta$} & \multicolumn{2}{c}{95\% CI}"
        r" & {Holm $p$} & Status \\",
        r"\midrule",
    ]
    contrasts = read_csv(FAM / "expansion_contrasts.csv")
    for first, second, label in TEN:
        for metric in METRICS:
            r = one(contrasts, metric=metric, first=first, second=second)
            key = f"TEN.{first}.{second}.{metric}"
            put(key + ".delta", num(float(r["all_delta"]), d(metric)))
            put(key + ".ci", ci(float(r["all_low"]), float(r["all_high"]), d(metric)))
            put(key + ".text", ndash(float(r["all_low"]), float(r["all_high"]), d(metric)))
            put(key + ".fresh", num(float(r["fresh_delta"]), d(metric)))
            put(
                key + ".fresh.text",
                ndash(float(r["fresh_low"]), float(r["fresh_high"]), d(metric)),
            )
            put(key + ".holm", pfmt(float(r["fresh_p_holm"])))
            put(key + ".status", r["status"])
            put(key + ".mean.a", f"{float(r['all_arm_a_mean']):.{d(metric)}f}")
            put(key + ".mean.b", f"{float(r['all_arm_b_mean']):.{d(metric)}f}")
            put(key + ".sd.a", f"{float(r['all_arm_a_sd']):.{d(metric)}f}")
            put(key + ".sd.b", f"{float(r['all_arm_b_sd']):.{d(metric)}f}")
        r = one(contrasts, metric="boundary_f1", first=first, second=second)
        if float(r["fresh_p_holm"]) < 0.001:
            raise ValueError("Holm p below 0.001 needs a non-S cell")  # pfmt switches notation
        lines.append(
            f"{label} & {s_in(r['all_delta'])} & {s_in(r['all_low'])} & {s_in(r['all_high'])} & "
            f"{s_in(r['fresh_delta'])} & {s_in(r['fresh_low'])} & {s_in(r['fresh_high'])} & "
            f"{pfmt(float(r['fresh_p_holm']))} & {STATUS_TEX[r['status']]} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]\footnotesize\raggedright\hyphenpenalty=10000",
        r"\item[] Status: \textbf{confirmed}, fresh-seed Holm $p<0.05$ and fresh-seed interval excluding"
        r" zero in the direction of the ten-run estimate; supported, not confirmed, only the ten-run"
        r" interval excludes zero; not established, neither.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
    ]
    write("tab_innetwork_main.tex", lines)


# ---------------------------------------------------------------- original exploratory KD (10.1)
def exploratory_kd() -> list[dict]:
    rows = read_csv(ROOT / "revision/kd_unified/contrasts.csv")
    picked = []
    for ds, _ in DATASETS:
        ref = "1-10" if ds == "isic2018" else "1-3"
        for metric in METRICS:
            r = one(
                rows,
                dataset=ds,
                comparison="KD-band - decoder-only",
                reference_seeds=ref,
                metric=metric,
                seed_matched="False" if ds == "isic2018" else "True",
            )
            key = f"KDO.{ds}.{metric}"
            put(key + ".delta", num(float(r["delta"]), d(metric)))
            put(key + ".ci", ci(float(r["low"]), float(r["high"]), d(metric)))
            put(key + ".text", ndash(float(r["low"]), float(r["high"]), d(metric)))
            put(key + ".p", pfmt(float(r["p"])))
            put(key + ".na", r["n_a"])
            put(key + ".nb", r["n_b"])
            if metric == "boundary_f1":
                if r["primary"] != "True":
                    raise ValueError(f"original exploratory row not primary: {ds}")
                put(key + ".holm", pfmt(float(r["p_holm_over_datasets"])))
                picked.append(r)
    return picked


# ---------------------------------------------------------------- T5 ACDC official patients
def acdc_official() -> dict:
    fu = read_csv(FU / "contrasts.csv")
    e3 = read_csv(E3 / "final_e3_analysis.csv")
    for r in e3:
        m, key = r["metric"], f"E3.{r['id']}.{r['metric']}"
        put(key + ".delta", num(float(r["delta"]), d(m)))
        put(key + ".ci", ci(float(r["low"]), float(r["high"]), d(m)))
        put(key + ".text", ndash(float(r["low"]), float(r["high"]), d(m)))
        put(key + ".p", pfmt(float(r["p"])))
        put(key + ".reading", r["reading"])
        put(key + ".na", r["n_a"])
        put(key + ".nb", r["n_b"])
        put(key + ".mean.a", f"{float(r['arm_a_mean']):.{d(m)}f}")
        put(key + ".mean.b", f"{float(r['arm_b_mean']):.{d(m)}f}")
        put(key + ".sd.a", f"{float(r['arm_a_sd']):.{d(m)}f}")
        put(key + ".sd.b", f"{float(r['arm_b_sd']):.{d(m)}f}")
    for r in fu:
        m, key = r["metric"], f"FX.{r['id']}.{r['metric']}"
        p_holm = float(r["p_holm"]) if r["p_holm"] else None
        put(key + ".text", ndash(float(r["low"]), float(r["high"]), d(m)))
        put(key + ".reading", holm_reading(r["reading"], p_holm))
    s3 = one(fu, id="S3", metric="boundary_f1")
    put("FX.S3.pfull", f"{float(s3['p']):.8f}")
    # unsigned forms for running text (the sign is carried by the verb)
    put("FIN.cf.abs", f"{abs(float(s3['delta'])):.4f}")
    put("FIN.cf.text", ndash(float(s3["low"]), float(s3["high"])))
    put("FIN.cf.low", f"{float(s3['low']):.4f}")
    per_seed = read_csv(ROOT / "revision/acdc_confirm/results/per_seed.csv")

    def runs(arm):
        return [
            float(r["value"])
            for r in per_seed
            if r["arm"] == arm and r["metric"] == "boundary_f1" and r["split"] == "official_test"
        ]

    put("FIN.cf.overlap", f"{max(runs('decoder')) - min(runs('kd_band')):.4f}")
    full = one(
        read_csv(FAM / "expansion_contrasts.csv"),
        metric="boundary_f1",
        first="full",
        second="decoder",
    )
    put("FIN.full.decoder.abs", f"{abs(float(full['all_delta'])):.4f}")
    # Groups named by their study number in Table 1, which gives the evidence level; row order shared with Fig. 3(a) (ACDC_ORDER).
    # Each contrast is set on two lines, "first arm -" and the second arm below it, using the
    # second line that every row has for the secondary intervals.
    blocks = [
        ("Replication", "study 4", [("S3", fu, "", KD_ARM, "D", 17)]),
        (
            "Controls",
            "study 5",
            [
                ("C1", fu, "C1", KD_ARM, "D+KD-full", 17),
                ("C2", fu, "C2", "D+KD-full", "D", 17),
                ("C3", fu, "C3", PEER, "D", 17),
                ("C4", fu, "C4", KD_ARM, PEER, 17),
            ],
        ),
        ("Augmentation comparison", "study 5", [("E3", e3, "", f"{KD_ARM}{AUG}", f"D{AUG}", 10)]),
    ]
    if tuple(cid for _, _, items in blocks for cid, *_ in items) != ACDC_ORDER:
        raise ValueError("Table 5 rows are not in the order of Fig. 3(a)")
    signed4 = r"S[table-format=+1.4, retain-explicit-plus]"
    signed2 = r"S[table-format=+1.2, retain-explicit-plus]"
    # Primary endpoint first and complete on the first line of each row; the secondary endpoints
    # follow after a gap, each as Delta over its [CI] on the second line.
    lines = [
        "% generated by analysis/revision/make_final_assets.py::acdc_official; do not edit",
        r"\footnotesize\setlength{\tabcolsep}{1.9pt}",
        r"\begin{threeparttable}",
        rf"\begin{{tabular}}{{@{{}}l c {signed4} @{{\hspace{{2\tabcolsep}}[}}{signed4}@{{,\ }}{signed4}"
        rf"@{{]\hspace{{2\tabcolsep}}}}S[table-format=1.3] >{{\raggedright\arraybackslash}}p{{1.55cm}}"
        rf"@{{\hspace{{2\tabcolsep}}\hspace{{0.3em}}}}{signed4} {signed2}@{{}}}}",
        r"\toprule",
        r"& & \multicolumn{5}{c}{Primary: Boundary F1} & \multicolumn{2}{c}{Secondary (descriptive)} \\",
        r"\cmidrule(lr){3-7}\cmidrule(l){8-9}",
        # Empty primary cells are spanned so that the "[", "," and "]" separators of the CI
        # columns appear in the data lines only.
        r"& {Runs} & \multicolumn{5}{c}{} & & {HD95 $\Delta$ (px),} \\",
        r"Contrast & {per arm} & \multicolumn{1}{c@{\hspace{2\tabcolsep}}}{$\Delta$} & \multicolumn{2}{c}{95\% CI}"
        r" & {$p$\tnote{a}} & Reading & {DSC $\Delta$} & {sign-reversed} \\",
    ]
    for title, level, items in blocks:
        lines += [r"\midrule", rf"\multicolumn{{9}}{{@{{}}l}}{{\emph{{{title}}} ({level})}} \\"]
        for cid, rows, tag, first, second, runs in items:
            prefix = f"{tag}\\enspace " if tag else ""
            indent = f"\\phantom{{{tag}}}\\enspace " if tag else ""
            label, label2 = f"{prefix}{first}{TEX_MINUS}", f"{indent}{second}"
            bf1 = one(rows, id=cid, metric="boundary_f1")
            dsc = one(rows, id=cid, metric="dsc")
            hd = one(rows, id=cid, metric="hd95")
            p_holm = float(bf1["p_holm"]) if bf1.get("p_holm") else None
            p_value = p_holm if p_holm is not None else float(bf1["p"])
            # siunitx input with the digits of pfmt(), the formatter shared with Fig. 3
            # p < 0.001 is set by \num in braces (centred), so that the S column need not
            # reserve exponent space on every row.
            p = (
                f"{p_value:.3f}"
                if p_value >= 0.001
                else "{{\\num{{{:.1f}e{}}}}}".format(*_mant_exp(p_value))
            )
            reading = READING_TEX[holm_reading(bf1["reading"], p_holm)]
            if cid == "C4":
                reading += r"\tnote{b}"
            lines.append(
                f"{label} & {runs} & {s_in(bf1['delta'])} & {s_in(bf1['low'])} & {s_in(bf1['high'])} & "
                f"{p} & \\multirow[t]{{2}}{{=}}{{{reading}}} & {s_in(dsc['delta'])} & {s_in(hd['delta'], 2)} \\\\"
            )
            lines.append(
                f"{label2} & & \\multicolumn{{5}}{{c}}{{}} & {{{ci(float(dsc['low']), float(dsc['high']))}}} & "
                f"{{{ci(float(hd['low']), float(hd['high']), 2)}}} \\\\"
            )
            lines.append(r"\addlinespace[3pt]")  # separates the two-line contrasts
        lines.pop()  # no extra space before the next rule
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]\footnotesize\raggedright\hyphenpenalty=10000",
        r"\item[a] Replication and augmentation: $p$ of the single primary contrast; C1--C4: Holm-adjusted"
        r" $p$ over the four controls.",
        r"\item[b] The unadjusted interval excludes zero, but the Holm-adjusted $p$ is"
        r" \R{FU.C4.boundary_f1.holm}.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
    ]
    write("tab_acdc_official_main.tex", lines)
    return dict(fu=fu, e3=e3)


def _mant_exp(p: float) -> tuple[float, int]:
    """Mantissa and exponent as printed by pfmt() for p < 0.001."""
    mantissa, exponent = f"{p:.1e}".split("e")
    return float(mantissa), int(exponent)


# ---------------------------------------------------------------- E3 supplement tables
ARMNAME = {
    "decoder_aug": f"D{AUG}",
    "kd_band_aug": f"{KD_ARM}{AUG}",
    "decoder": "D",
    "kd_band": KD_ARM,
}


def e3_supplement() -> None:
    e3 = read_csv(E3 / "final_e3_analysis.csv")
    lines = [
        r"\begin{tabular}{@{}llllll@{}}",
        r"\toprule",
        r"Comparison & Metric & $\Delta$ [95\% CI] & $p$ & Reading & Arm means (SD) \\",
        r"\midrule",
    ]
    for cid in ("E3", "E3a", "E3b"):
        for m in METRICS:
            r = one(e3, id=cid, metric=m)
            first = (
                f"{cid}: {ARMNAME[r['first']]} $-$ {ARMNAME[r['second']]} ({r['n_a']} vs {r['n_b']})"
                if m == "boundary_f1"
                else ""
            )
            means = (
                f"{float(r['arm_a_mean']):.{d(m)}f} ({float(r['arm_a_sd']):.{d(m)}f}) / "
                f"{float(r['arm_b_mean']):.{d(m)}f} ({float(r['arm_b_sd']):.{d(m)}f})"
            )
            lines.append(
                f"{first} & {MLABEL[m]} & {num(float(r['delta']), d(m))} {ci(float(r['low']), float(r['high']), d(m))} & "
                f"{pfmt(float(r['p']))} & {r['reading']} & {means} \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    write("tab_s_e3_all.tex", lines)

    per_seed = read_csv(E3 / "final_e3_per_seed.csv")
    lines = [
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"Seed & D$_{\mathrm{aug}}$ BF1 & D+KD-band$_{\mathrm{aug}}$ BF1 & D$_{\mathrm{aug}}$ DSC & D+KD-band$_{\mathrm{aug}}$ DSC \\",
        r"\midrule",
    ]
    lines[0] = r"\begin{tabular}{@{}lrrrr@{}}"
    seeds = sorted({int(r["seed"]) for r in per_seed if r["arm"] == "decoder_aug"})
    for s in seeds:
        cells = [
            f"{float(one(per_seed, arm=a, seed=str(s), metric=m)['value']):.4f}"
            for m in ("boundary_f1", "dsc")
            for a in ("decoder_aug", "kd_band_aug")
        ]
        lines.append(f"{s} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_s_e3_perseed.tex", lines)

    per_class = read_csv(E3 / "final_e3_per_class.csv")
    lines = [
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        r"Structure & DSC $\Delta$ [95\% CI] & Boundary F1 $\Delta$ [95\% CI] & HD95 $\Delta$ (px) [95\% CI] \\",
        r"\midrule",
    ]
    for structure in ("RV", "myocardium", "LV"):
        cells = []
        for m in ("dsc", "boundary_f1", "hd95"):
            r = one(per_class, structure=structure, metric=m)
            cells.append(
                f"{num(float(r['delta']), d(m))} {ci(float(r['low']), float(r['high']), d(m))}"
            )
        lines.append(f"{structure} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_s_e3_perclass.tex", lines)


# ---------------------------------------------------------------- T6 resource profile
def resource_table() -> None:
    rows = [
        r
        for r in read_csv(ROOT / "revision/tables/resource_profile.csv")
        if r["dataset"] == "isic2018"
    ]
    order = [
        ("U-Net", "U-Net"),
        ("BDS-Lite (deployed)", "BDS-Lite, gate retained"),
        ("BDS-Lite (gate-collapsed)", "BDS-Lite, gate collapsed"),
        ("Attention U-Net", "Attention U-Net"),
        ("LB-UNet", "LB-UNet (adapted)"),
    ]
    lines = [
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"Graph & Training params & Inference params & GFLOPs & GPU ms, batch 1 & GPU ms, batch 16 & CPU ms, 1 thread \\",
        r"\midrule",
    ]
    for model, label in order:
        r = one(rows, model=model)
        training = f"{int(r['params_total']):,}" if model != "BDS-Lite (gate-collapsed)" else "--"
        lines.append(
            f"{label} & {training} & {int(r['params_deployed']):,} & {float(r['gflops_b1_released_hook']):.3f} & "
            f"{float(r['gpu_fp32_b1_ms_median']):.2f} & {float(r['gpu_fp32_b16_ms_median']):.2f} & "
            f"{float(r['cpu_ms_b1_1thread']):.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_resource_final.tex", lines)
    u, g = one(rows, model="U-Net"), one(rows, model="BDS-Lite (deployed)")
    put("RF.params.extra", f"{int(g['params_deployed']) - int(u['params_deployed']):,}")
    put(
        "RF.gflops.pct",
        f"{100 * (float(g['gflops_b1_released_hook']) / float(u['gflops_b1_released_hook']) - 1):.2f}",
    )
    for b in ("1", "16"):
        pct = 100 * (
            float(g[f"gpu_fp32_b{b}_ms_median"]) / float(u[f"gpu_fp32_b{b}_ms_median"]) - 1
        )
        put(f"RF.gpu.b{b}.pct", f"{pct:.1f}")
        put(f"RF.gpu.b{b}.u", f"{float(u[f'gpu_fp32_b{b}_ms_median']):.2f}")
        put(f"RF.gpu.b{b}.g", f"{float(g[f'gpu_fp32_b{b}_ms_median']):.2f}")
    put("RF.kernels.u", u["cuda_kernels_b1"])
    put("RF.kernels.g", g["cuda_kernels_b1"])
    gate = read_csv(ROOT / "revision/tables/gate_identity_effects.csv")
    split = {"isic2018": "official_test", "acdc": "test", "synapse": "test"}
    worst = 0.0
    for ds, _ in DATASETS:
        r = one(gate, dataset=ds, split=split[ds], metric="boundary_f1")
        est = float(r["mean_delta_identity_minus_full"])
        worst = max(worst, abs(est))
        put(f"GID.{ds}.bf1", num(est, 6))
        put(f"GID.{ds}.bf1.ci", ci(float(r["ci95_low"]), float(r["ci95_high"]), 6))
    put("GID.maxabs", f"{worst:.4f}")
    timing = json.loads(
        (ROOT / "results/acdc_confirm_eval_posthoc/teacher_cache_timing.json").read_text()
    )
    put("RF.cache.sec", f"{timing['median_seconds']:.1f}")
    costs = read_csv(ROOT / "revision/kd_unified/costs.csv")
    put(
        "RF.teacher.hours",
        f"{float(one(costs, run='acdc_kd_teacher_seed100')['sum_epoch_hours']):.3f}",
    )
    conf = json.loads((ROOT / "revision/acdc_confirm/results/results.json").read_text())
    put("RF.s3.hours.kd", f"{conf['training_epoch_hours']['kd_band']:.2f}")
    put("RF.s3.hours.dec", f"{conf['training_epoch_hours']['decoder']:.2f}")
    put("CF.power.15", f"{conf['achieved_power']['delta_0.015_sd_bound_90']:.2f}")


# ---------------------------------------------------------------- ISIC2018 ten-run means (B-31)
def ten_run_means() -> None:
    import expansion_statistics as es

    labels = (("unet", "U-Net"), ("decoder", "D"), ("full", "BDS-Lite"), ("gsl", "U-Net+GSL"))
    lines = [
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        r"Arm (10 runs) & DSC & IoU & Boundary F1 & HD95 (px) & ASSD (px) \\",
        r"\midrule",
    ]
    for arm, label in labels:
        cells = []
        for metric in METRICS[1:2] + METRICS[2:3] + METRICS[0:1] + METRICS[3:]:
            run_means = []
            for seed in range(1, 11):
                values = es.unit_values(es.isic_path(arm, seed), "isic2018")[metric]
                run_means.append(float(np.mean(list(values.values()))))
            mean, sd = float(np.mean(run_means)), float(np.std(run_means, ddof=1))
            cells.append(f"${mean:.{d(metric)}f} \\pm {sd:.{d(metric)}f}$")
            put(f"TENM.{arm}.{metric}", f"{mean:.{d(metric)}f}")
            put(f"TENM.{arm}.{metric}.sd", f"{sd:.{d(metric)}f}")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_ten_seed_final.tex", lines)


def resource_main_table() -> None:
    """Main-text resource table: the four U-Net-family graphs of the saved ISIC2018 profile."""
    rows = [
        r
        for r in read_csv(ROOT / "revision/tables/resource_profile.csv")
        if r["dataset"] == "isic2018"
    ]
    order = [
        ("U-Net", "U-Net"),
        ("BDS-Lite (deployed)", "BDS-Lite, $P$ and gate kept"),
        ("BDS-Lite (gate-collapsed)", "BDS-Lite, gate as identity"),
        ("Attention U-Net", "Attention U-Net"),
    ]
    lines = [
        "% generated by analysis/revision/make_final_assets.py::resource_main_table; do not edit",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"& Parameters & & \multicolumn{3}{c}{Median latency (ms)} \\",
        r"\cmidrule(l){4-6}",
        r"Inference graph & at inference & GFLOPs & GPU, b1 & GPU, b16 & CPU, 1 thread \\",
        r"\midrule",
    ]
    for model, label in order:
        r = one(rows, model=model)
        lines.append(
            f"{label} & {int(r['params_deployed']):,} & {float(r['gflops_b1_released_hook']):.3f} & "
            f"{float(r['gpu_fp32_b1_ms_median']):.2f} & {float(r['gpu_fp32_b16_ms_median']):.2f} & "
            f"{float(r['cpu_ms_b1_1thread']):.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_resource_text.tex", lines)
    full = one(rows, model="BDS-Lite (deployed)")
    put("RF.params.train.full", f"{int(full['params_total']):,}")


def published_table() -> None:
    """Main-text table of published results: panel B of tab_lit_main.tex, values unchanged."""
    text = (GEN / "tab_lit_main.tex").read_text()
    body = text.split(r"\textbf{B. Published references: different protocols}")[1]
    rows = [
        ln
        for ln in body.splitlines()
        if ln.strip().endswith(r"\\") and "Dataset & Model" not in ln and "&" in ln
    ]
    if len(rows) != 6:
        raise ValueError(f"expected six published rows, found {len(rows)}")
    protocol = {
        "EGE-UNet": "random 7:3 split, $256\\times256$, augmentation",
        "LB-UNet": "random 7:3 split, $256\\times256$, augmentation",
        "TransUNet": "ImageNet-pretrained, volume-level score",
        "Swin-Unet": "ImageNet-pretrained, volume-level score",
    }
    lines = [
        "% generated by analysis/revision/make_final_assets.py::published_table from tab_lit_main.tex",
        r"\begin{tabular}{@{}llrrl@{}}",
        r"\toprule",
        r"Dataset & Model & DSC & HD\tnote{a} & Protocol of the source \\",
        r"\midrule",
    ]
    for ln in rows:
        cells = [c.strip() for c in ln.rstrip("\\").split("&")]
        dataset, model, dsc, _bf1, hd = cells
        name = model.split(r" \citep")[0]
        lines.append(f"{dataset} & {model} & {dsc} & {hd} & {protocol[name]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_published_text.tex", lines)


def plain_lit_table() -> None:
    text = (GEN / "tab_lit_main.tex").read_text()
    text = re.sub(r"\\mathbf\{([^{}]*)\}", r"\1", text)
    text = text.replace(
        "generated by analysis/revision/make_comparison_table.py",
        "boldface removed by analysis/revision/make_final_assets.py from tab_lit_main.tex",
    )
    (GEN / "tab_lit_main_plain.tex").write_text(text)


# ---------------------------------------------------------------- Fig. 3
# One colour and marker per arm in both panels (Okabe-Ito); a contrast in (a) takes those of its
# first arm. Filled markers: no augmentation; open markers: D4 augmentation (the D and D+KD-band
# styles reused).
FIG3_ARMS = {
    "decoder": ("D", "blue", "o"),
    "kd_full": ("D+KD-full", "bluish_green", "s"),
    "kd_band_peer": (PEER, "reddish_purple", "^"),
    "kd_band": (KD_ARM, "vermilion", "v"),
    "decoder_aug": (f"D{AUG}", "blue", "o"),
    "kd_band_aug": (f"{KD_ARM}{AUG}", "vermilion", "v"),
}
MINUS = "−"


def bf1_p(row: dict) -> str:
    """p of Table 5 and Fig. 3: Holm-adjusted inside the C1-C4 family, else the single-contrast p."""
    return pfmt(float(row["p_holm"])) if row.get("p_holm") else pfmt(float(row["p"]))


def beeswarm(x_pt: np.ndarray, diameter: float) -> np.ndarray:
    """Vertical offsets (pt) that keep markers of the given diameter from overlapping.

    The rule of seaborn's swarmplot: points are placed in order of value, each at the free offset
    closest to the centre line. Only the drawing position changes; the values are not moved.
    """
    offsets = np.zeros(len(x_pt))
    placed: list[tuple[float, float]] = []
    for i in np.argsort(x_pt, kind="stable"):
        x = float(x_pt[i])
        near = [(px, py) for px, py in placed if abs(px - x) < diameter]
        candidates = [0.0]
        for px, py in near:
            dy = (diameter**2 - (px - x) ** 2) ** 0.5
            candidates += [py + dy, py - dy]
        for c in sorted(candidates, key=abs):
            if all((px - x) ** 2 + (py - c) ** 2 >= diameter**2 - 1e-9 for px, py in near):
                offsets[i] = c
                break
        placed.append((x, offsets[i]))
    return offsets


def figure(data: dict) -> None:
    """Fig. 3: (a) forest plot of the ACDC contrasts in the layout of Fig. 2 (labels | intervals |
    Delta [95% CI] | p), rows in the order of Table 5; (b) run means of every arm as a beeswarm with
    the arm mean and +-1 SD, unaugmented and augmented arms on two axes of the same scale.

    Every printed number uses the formatter and CSV row of Table 5 (num, ci, bf1_p); arm means and
    SDs are the frozen arm_a/arm_b columns of the contrast files and are checked against the saved
    per-run values. Drawn at the text width with analysis/revision/figure_style.py; included unscaled.
    """
    figure_style.apply()
    oi = figure_style.OKABE_ITO
    ink, muted, band = figure_style.INK, figure_style.MUTED, figure_style.BAND
    fu, e3 = data["fu"], data["e3"]
    width = figure_style.W_IN * 72
    # Columns and type as Fig. 2 (make_narrative_assets.py); a p column is added on the right.
    row_pt, gap, title_rows = 10.0, 0.6, 1.85
    x_title, x_row, x_id = 0.5, 9.0, 22.0
    ax_left, ax_right = 116.0, width - 130.0
    x_delta, x_p = width - 42.0, width - 0.5
    patients = json.loads(
        (ROOT / "revision/acdc_confirm/official_test_integrity.json").read_text()
    )["patients"]

    def colour(arm: str) -> str:
        return oi[FIG3_ARMS[arm][1]]

    # ---- (a) rows: (id, rows, id label, contrast, first arm, augmented); order of Table 5
    def contrast(first: str, second: str) -> str:
        return f"{FIG3_ARMS[first][0]} {MINUS} {FIG3_ARMS[second][0]}"

    groups_a = [
        (
            "Replication",
            "confirmatory primary contrast, n = 17",
            [("S3", fu, "", contrast("kd_band", "decoder"), "kd_band", False)],
        ),
        (
            "Controls",
            "exploratory, Holm-adjusted p, n = 17",
            [
                ("C1", fu, "", contrast("kd_band", "kd_full"), "kd_band", False),
                ("C2", fu, "", contrast("kd_full", "decoder"), "kd_full", False),
                ("C3", fu, "", contrast("kd_band_peer", "decoder"), "kd_band_peer", False),
                ("C4", fu, "", contrast("kd_band", "kd_band_peer"), "kd_band", False),
            ],
        ),
        (
            "Augmentation",
            "exploratory, D4 augmentation, n = 10",
            [("E3", e3, "", contrast("kd_band_aug", "decoder_aug"), "kd_band_aug", True)],
        ),
    ]
    if tuple(item[0] for _, _, items in groups_a for item in items) != ACDC_ORDER:
        raise ValueError("Fig. 3(a) rows are not in the order of Table 5")
    titles_a, rows_a, y = [], [], 0.0
    for title, subtitle, items in groups_a:
        titles_a.append((title, subtitle, y))
        y += title_rows
        for item in items:
            rows_a.append((y, *item))
            y += 1
        y += gap
    a_top, a_bottom = -0.75, rows_a[-1][0] + 0.7

    # ---- (b) arms, frozen means and SDs, and per-run values
    def frozen(rows: list[dict], cid: str, side: str) -> tuple[float, float]:
        r = one(rows, id=cid, metric="boundary_f1")
        return float(r[f"arm_{side}_mean"]), float(r[f"arm_{side}_sd"])

    summary = {
        "decoder": frozen(fu, "S3", "b"),
        "kd_band": frozen(fu, "S3", "a"),
        "kd_full": frozen(fu, "C2", "a"),
        "kd_band_peer": frozen(fu, "C3", "a"),
        "decoder_aug": frozen(e3, "E3", "b"),
        "kd_band_aug": frozen(e3, "E3", "a"),
    }
    s_fu = read_csv(FU / "per_seed.csv")
    s_e3 = read_csv(E3 / "final_e3_per_seed.csv")

    def runs(rows: list[dict], arm: str) -> np.ndarray:
        values = np.array(
            [float(r["value"]) for r in rows if r["arm"] == arm and r["metric"] == "boundary_f1"]
        )
        mean, sd = summary[arm]
        if abs(values.mean() - mean) > 1e-9 or abs(values.std(ddof=1) - sd) > 1e-9:
            raise ValueError(f"{arm}: saved runs do not reproduce the frozen mean and SD")
        return values

    groups_b = []
    for name, rows, arms, xlim in (
        ("No augmentation", s_fu, ["decoder", "kd_full", "kd_band_peer", "kd_band"], (0.76, 0.81)),
        ("D4 augmentation", s_e3, ["decoder_aug", "kd_band_aug"], (0.825, 0.875)),
    ):
        values = {arm: runs(rows, arm) for arm in arms}
        sizes = {len(v) for v in values.values()}
        if len(sizes) != 1:
            raise ValueError(f"{name}: unequal run counts {sizes}")
        groups_b.append((f"{name}, n = {sizes.pop()}", values, xlim))

    # ---- vertical layout in points from the top of the figure
    plot_w = ax_right - ax_left
    header, ticks_only, ticks_label, b_title = 13.0, 13.0, 25.0, 11.0
    a_height = row_pt * (a_bottom - a_top)
    # (b) rows in points, raincloud-like: the swarm on top, the mean and SD bar under it. Both
    # axes span 0.05, so they share the scale; the row height follows the widest swarm.
    diameter, tick = 3.9, 2.5
    swarms = {}
    for _, values, xlim in groups_b:
        if abs((xlim[1] - xlim[0]) - 0.05) > 1e-12:
            raise ValueError("panel (b) axes must share the scale")
        for arm, v in values.items():
            swarms[arm] = beeswarm(v * plot_w / 0.05, diameter)
    up = max(o.max() for o in swarms.values()) + diameter / 2
    down = -min(o.min() for o in swarms.values()) + diameter / 2
    bar_at = up + down + tick + 1.0
    b_row = bar_at + tick + 2.5
    b_heights = [b_row * len(v) for _, v, _ in groups_b]
    legend_h = 36.0
    total = (
        header
        + a_height
        + ticks_label
        + 8.0
        + header
        + sum(b_title + h for h in b_heights)
        + ticks_only
        + ticks_label
        + legend_h
    )
    fig = plt.figure(figsize=(figure_style.W_IN, total / 72))

    def axes_at(top_pt: float, height_pt: float):
        return fig.add_axes(
            [ax_left / width, 1 - (top_pt + height_pt) / total, plot_w / width, height_pt / total]
        )

    def ftext(x_pt: float, top_pt: float, text: str, **kw) -> None:
        kw.setdefault("va", "center")
        kw.setdefault("color", ink)
        fig.text(x_pt / width, 1 - top_pt / total, text, **kw)

    def frule(top_pt: float) -> None:
        fig.add_artist(Line2D([0, 1], [1 - top_pt / total] * 2, color=ink, lw=0.5))

    def style_x(ax, label: str | None) -> None:
        ax.set_yticks([])
        ax.tick_params(axis="x", length=2, width=0.4, pad=2)
        for edge in ("top", "right", "left"):
            ax.spines[edge].set_visible(False)
        if label:
            ax.set_xlabel(label, labelpad=3)

    def fmt(v: float, _=None) -> str:
        return "0" if abs(v) < 1e-12 else f"{v:.2f}".replace("-", MINUS)

    # ---- (a)
    cursor = 0.0
    ftext(x_title, cursor + header / 2, "a", fontweight="bold", fontsize=9)
    ftext(
        x_row,
        cursor + header / 2,
        f"Contrast on the {patients} official ACDC patients (n = runs per arm)",
        fontweight="bold",
    )
    ftext(x_delta, cursor + header / 2, "Δ [95% CI]", ha="right", fontweight="bold")
    ftext(x_p, cursor + header / 2, "$p$", ha="right", fontweight="bold")
    cursor += header
    frule(cursor)
    ax = axes_at(cursor, a_height)
    ax.set_xlim(-0.02, 0.03)
    ax.set_ylim(a_bottom, a_top)
    ax.axvspan(-0.01, 0.01, color=band, zorder=0, lw=0)
    ax.axvline(0, color=muted, lw=0.6, zorder=1)
    side = blended_transform_factory(fig.transFigure, ax.transData)

    def label(x_pt: float, y: float, text: str, **kw) -> None:
        kw.setdefault("va", "center")
        kw.setdefault("color", ink)
        ax.text(x_pt / width, y, text, transform=side, clip_on=False, **kw)

    for title, subtitle, y0 in titles_a:
        label(x_title, y0, title, fontweight="bold", fontsize=7.5)
        label(x_title, y0 + 0.85, subtitle, color=muted, fontsize=7)
    for y0, cid, rows, tag, name, first, hollow in rows_a:
        r = one(rows, id=cid, metric="boundary_f1")
        est, low, high = float(r["delta"]), float(r["low"]), float(r["high"])
        c, marker = colour(first), FIG3_ARMS[first][2]
        if tag:
            label(x_row, y0, tag)
        label(x_id if tag else x_row, y0, name)
        label(x_delta, y0, f"{num(est, 3)} {ci(low, high, 3)}".replace("$-$", MINUS), ha="right")
        label(x_p, y0, bf1_p(r), ha="right")
        ax.plot([low, high], [y0, y0], color=c, lw=1.2, solid_capstyle="butt", zorder=2)
        ax.plot(est, y0, marker, ms=4.6, mfc="white" if hollow else c, mec=c, mew=0.9, zorder=3)
    ax.set_xticks([-0.02, -0.01, 0, 0.01, 0.02, 0.03])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt))
    style_x(ax, "Δ Boundary F1 (positive favours the first arm)")
    cursor += a_height + ticks_label + 8.0

    # ---- (b)
    ftext(x_title, cursor + header / 2, "b", fontweight="bold", fontsize=9)
    ftext(
        x_row,
        cursor + header / 2,
        "Arm: Boundary F1 of each run (mean over the patients)",
        fontweight="bold",
    )
    ftext(x_delta, cursor + header / 2, "Mean ± SD", ha="right", fontweight="bold")
    cursor += header
    frule(cursor)
    for index, ((title, values, xlim), height) in enumerate(zip(groups_b, b_heights, strict=True)):
        ftext(x_title, cursor + b_title / 2 + 1, title, fontweight="bold", fontsize=7.5)
        cursor += b_title
        bx = axes_at(cursor, height)
        n_arms = len(values)
        bx.set_xlim(*xlim)
        bx.set_ylim(n_arms * b_row, 0)
        bside = blended_transform_factory(fig.transFigure, bx.transData)
        for k, (arm, v) in enumerate(values.items()):
            name, _, marker = FIG3_ARMS[arm]
            c, hollow = colour(arm), arm.endswith("_aug")
            mean, sd = summary[arm]
            centre, bar = k * b_row + up, k * b_row + bar_at
            bx.scatter(
                v,
                centre - swarms[arm],
                s=13,
                marker=marker,
                facecolors="white" if hollow else c,
                edgecolors=c,
                linewidths=0.8,
                zorder=3,
            )
            bx.plot(
                [mean - sd, mean + sd],
                [bar, bar],
                color=c,
                lw=1.0,
                zorder=2,
                solid_capstyle="butt",
            )
            bx.plot([mean, mean], [bar - tick, bar + tick], color=c, lw=1.5, zorder=2)
            bx.text(x_row / width, centre, name, transform=bside, va="center", color=ink)
            bx.text(
                x_delta / width,
                centre,
                f"{mean:.3f} ± {sd:.3f}",
                transform=bside,
                va="center",
                ha="right",
                color=ink,
            )
        bx.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.01))
        bx.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt))
        last = index == len(groups_b) - 1
        style_x(bx, f"Boundary F1, run mean over the {patients} patients" if last else None)
        cursor += height + (ticks_label if last else ticks_only)

    # ---- legend: arms (both panels), augmentation fill, (b) summary, (a) SESOI
    handles = [
        Line2D(
            [],
            [],
            ls="none",
            marker=FIG3_ARMS[a][2],
            ms=4.6,
            mfc=colour(a),
            mec=colour(a),
            label=FIG3_ARMS[a][0],
        )
        for a in ("decoder", "kd_full", "kd_band_peer", "kd_band")
    ]
    handles += [
        Line2D(
            [],
            [],
            ls="none",
            marker="o",
            ms=4.6,
            mfc="white",
            mec=muted,
            label="open: D4 augmentation",
        ),
        Line2D([], [], color=muted, lw=1.0, marker="|", ms=6, mew=1.5, label="(b) mean ± SD"),
        Patch(facecolor=band, edgecolor="none", label="(a) ±0.01, preset threshold"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=4,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.5,
        borderaxespad=0.15,
        title="Colour and marker: arm; in (a), the first arm of the contrast",
        title_fontsize=7,
    )
    fig.savefig(FIG / "kd_acdc.pdf", metadata={"CreationDate": None})
    fig.savefig(FIG / "kd_acdc.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- S15 all training records
INVENTORY = ROOT / "revision/evidence_freeze/final_experiment_inventory.csv"
# Families whose inventory rows leave era/use blank; the article uses all of them (S17, S18, S21).
USED_BLANK = {
    "exploratory external KD students": "yes (S17)",
    "exploratory external teacher": "yes (S17)",
    "ACDC confirmation": "yes",
    "follow-up E1/E2 students": "yes",
    "follow-up E3 augmentation": "yes",
    "follow-up compact teacher/cache": "yes",
}
# Display labels for the stopped study (supplement S15 describes it in one sentence).
DISPLAY = {
    "Stage 2 Experiment C": "Stopped study",
    "confirmatory, blind": "stopped, not analysed",
    "no (blind)": "no",
    "withdrawn registration": "withdrawn design slots",
    # internal queue names of the stage-1 launches, shown as in the ledger table (S15)
    "WP2/WP4 queue A": "Prespecified ablations",
    "WP4 queue B": "Prespecified comparators",
    "WP2/WP3 Tier B": "Prespecified sensitivity",
}


def inventory_table() -> None:
    """Every training record and withdrawn design slot of the frozen inventory, by family."""
    rows = [
        r
        for r in read_csv(INVENTORY)
        if r["row_type"] in ("training_record", "withdrawn_design_slot")
    ]
    order: list[str] = []
    for r in rows:
        if r["family"] not in order:
            order.append(r["family"])
    lines = [
        r"\begin{longtable}{@{}p{3.6cm}p{1.9cm}rrp{3.4cm}p{1.9cm}@{}}",
        r"\caption{\label{tab:inventory}All training records by family. $^{a}$Completed: a 150-epoch "
        r"trajectory was read from the run record; other statuses (completion recorded in the ledger "
        r"only, completed runs of the stopped study, partial, diagnostic, failed, not started, "
        r"withdrawn) are listed per run in the archived inventory (S16). $^{b}$Period not recorded in the "
        r"inventory; these runs were trained in November 2026 under the protocols of S17, S18 and "
        r"S21. Evidence class and use are the inventory's fields; ``--'': not recorded.}\\",
        r"\toprule",
        r"Family & Period & Records & Completed$^{a}$ & Evidence class & Used here \\",
        r"\midrule",
        r"\endhead",
    ]
    total = completed = 0
    for fam in order:
        group = [r for r in rows if r["family"] == fam]
        done = sum(r["observed_status"] == "completed_150_epoch_record" for r in group)
        eras = sorted({r["era"] for r in group if r["era"]}) or ["Nov 2026$^{b}$"]
        classes = sorted({r["evidence_class"] for r in group if r["evidence_class"]})
        used = sorted({r["in_current_manuscript"] for r in group if r["in_current_manuscript"]})
        used_text = "; ".join(used) or USED_BLANK.get(fam, "--")
        total += len(group)
        completed += done
        cells = [
            DISPLAY.get(fam, fam),
            "; ".join(re.sub(r" \(commit [0-9a-f]+\)", "", e) for e in eras),
            str(len(group)),
            str(done),
            "; ".join(DISPLAY.get(c, c) for c in classes).replace("registered", "prespecified")
            or "--",
            DISPLAY.get(used_text, used_text),
        ]
        lines.append(" & ".join(c.replace("&", r"\&").replace("_", r"\_") for c in cells) + r" \\")
    lines += [
        r"\midrule",
        f"Total & & {total} & {completed} & & \\\\",
        r"\bottomrule",
        r"\end{longtable}",
    ]
    put("INV.records", str(total))
    put("INV.completed", str(completed))
    write("tab_s_inventory.tex", lines)


# ---------------------------------------------------------------- teacher evaluation (post hoc)
TEACHER = ROOT / "results/teacher_eval/summary.json"
TE_GROUPS = [
    ("D", "D"),
    ("D+KD-band", KD_ARM),
    ("D+KD-full", "D+KD-full"),
    ("D+KD-band_c", PEER),
    ("D_aug", f"D{AUG}"),
    ("D+KD-band_aug", f"{KD_ARM}{AUG}"),
]


def teacher_eval() -> None:
    """Frozen teachers on official ACDC patients 101-150 (post hoc, descriptive).

    Source: results/teacher_eval/summary.json (evidence addendum
    revision/EVIDENCE_ADDENDUM_20261111_teacher_eval.md). Formats saved values only.
    """
    summary = json.loads(TEACHER.read_text())
    teachers, comp = summary["teachers"], summary["comparisons"]
    ext, cmp_ = teachers["external"]["patient_level"], teachers["compact"]["patient_level"]
    put("TE.ext.bf1", f"{ext['boundary_f1']:.4f}")
    put("TE.cmp.bf1", f"{cmp_['boundary_f1']:.4f}")
    put("TE.hashes", str(summary["sanity_checks"]["artifact_hash_checks"]))
    put("TE.recomputed", str(summary["sanity_checks"]["runs_recomputed_equal_to_saved"]))
    put("TE.draws", f"{summary['bootstrap']['draws']:,}")
    put("TE.outcome", summary["interpretation"]["outcome"])
    # Post hoc proportional-transfer check (addendum C16): the share of the teacher-D gap that the
    # registered unaugmented gain closed, applied to the teacher-D_aug gap, against the E3 interval.
    gain = float(one(read_csv(FU / "contrasts.csv"), id="S3", metric="boundary_f1")["delta"])
    share = gain / comp["external"]["D"]["paired_difference_teacher_minus_group"]
    expected = share * comp["external"]["D_aug"]["paired_difference_teacher_minus_group"]
    e3_high = float(
        one(read_csv(E3 / "final_e3_analysis.csv"), id="E3", metric="boundary_f1")["high"]
    )
    put("TE.prop.share", f"{100 * share:.0f}")
    put("TE.prop.expected", num(expected))
    put("TE.prop.upper", num(e3_high))
    put("TE.runs", str(sum(comp["external"][g]["runs"] for g, _ in TE_GROUPS)))
    put(
        "TE.runs.above.ext",
        str(sum(comp["external"][g]["runs_exceeding_teacher"] for g, _ in TE_GROUPS)),
    )
    for who, tag in (("external", "ext"), ("compact", "cmp")):
        for g, _ in TE_GROUPS:
            r = comp[who][g]
            key = f"TE.{tag}.{g.replace('+', '').replace('-', '').replace('_', '')}"
            low, high = r["bootstrap_95ci"]
            put(key + ".delta", num(r["paired_difference_teacher_minus_group"]))
            put(key + ".text", ndash(low, high))
            put(key + ".above", str(r["runs_exceeding_teacher"]))
            put(key + ".runs", str(r["runs"]))
    rows = [
        r"\begin{tabular}{@{}lrccccc@{}}",
        r"\toprule",
        r"Teacher & Parameters & DSC & IoU & Boundary F1 & HD95 (px) & ASSD (px) \\",
        r"\midrule",
    ]
    for who, label in (
        ("external", "External (pretrained ResNet-34 U-Net)"),
        ("compact", "Compact (student architecture)"),
    ):
        t = teachers[who]
        v = t["patient_level"]
        rows.append(
            f"{label} & {t['parameters']:,} & {v['dsc']:.4f} & {v['iou']:.4f} & {v['boundary_f1']:.4f} & "
            f"{v['hd95']:.2f} & {v['assd']:.2f} \\\\"
        )
    rows += [r"\bottomrule", r"\end{tabular}"]
    write("tab_s_teacher_scores.tex", rows)
    rows = [
        r"\begin{tabular}{@{}lrlrlrl@{}}",
        r"\toprule",
        r"Group & Runs & Run means: mean (min--max) & Runs $>$ ext. & External $-$ group [95\% CI] & Runs $>$ comp. & Compact $-$ group [95\% CI] \\",
        r"\midrule",
    ]
    for g, label in TE_GROUPS:
        e, c = comp["external"][g], comp["compact"][g]
        rows.append(
            f"{label} & {e['runs']} & {e['run_mean_bf1_mean']:.4f} ({e['run_mean_bf1_min']:.4f}--{e['run_mean_bf1_max']:.4f}) & "
            f"{e['runs_exceeding_teacher']} & {num(e['paired_difference_teacher_minus_group'])} {ci(*e['bootstrap_95ci'])} & "
            f"{c['runs_exceeding_teacher']} & {num(c['paired_difference_teacher_minus_group'])} {ci(*c['bootstrap_95ci'])} \\\\"
        )
    rows += [r"\bottomrule", r"\end{tabular}"]
    write("tab_s_teacher_groups.tex", rows)


# ---------------------------------------------------------------- T1 studies and design composition
COMPLETED = "completed_150_epoch_record"


def _inventory_runs(family: str) -> list[dict]:
    """Completed training records of one inventory family, with arm and seed parsed from run_id."""
    out = []
    for r in read_csv(INVENTORY):
        if r["family"] != family or r["row_type"] != "training_record":
            continue
        if r["observed_status"] != COMPLETED:
            raise ValueError(f"{r['run_id']}: {r['observed_status']}")
        m = re.fullmatch(rf"{r['dataset']}_(.+)_seed(\d+)", r["run_id"])
        if not m:
            raise ValueError(f"cannot parse {r['run_id']}")
        out.append({"dataset": r["dataset"], "arm": m.group(1), "seed": int(m.group(2))})
    return out


def _seed_range(seeds: set[int]) -> str:
    lo, hi = min(seeds), max(seeds)
    if seeds != set(range(lo, hi + 1)):
        raise ValueError(f"seeds not contiguous: {sorted(seeds)}")
    return f"{lo}--{hi}"


def _per_arm(runs: list[dict]) -> tuple[int, str]:
    """Runs per arm and dataset (must be equal in every cell) and the seed range."""
    cells: dict[tuple[str, str], set[int]] = {}
    for r in runs:
        cells.setdefault((r["dataset"], r["arm"]), set()).add(r["seed"])
    sizes = {len(s) for s in cells.values()}
    seeds = {frozenset(s) for s in cells.values()}
    if len(sizes) != 1 or len(seeds) != 1:
        raise ValueError(f"unequal cells: {cells}")
    return sizes.pop(), _seed_range(set(seeds.pop()))


def design_composition() -> dict:
    """Recount the 237 core design slots from the analysis inputs and the training inventory.

    Study 1 ablation and reference-method runs: the per-case files read by the frozen analyses
    (families/provenance.json; cell M7 from expansion_statistics.cell_path). Study 2: the ISIC2018
    ten-run arms of expansion_statistics.isic_path; runs already counted in study 1 count once.
    Studies 3-5: completed training records of final_experiment_inventory.csv.
    """
    import expansion_statistics as es

    prov = json.loads((FAM / "provenance.json").read_text())["inputs"]
    seeds13 = (1, 2, 3)

    def files(methods: tuple[str, ...]) -> dict[tuple[str, str, int], str]:
        out = {}
        for m in methods:
            for ds, _ in DATASETS:
                for s in seeds13:
                    if m == "M7":
                        path = es.cell_path("M7", ds, s, dry=False).relative_to(ROOT).as_posix()
                    else:
                        path = prov[f"{m}/{ds}/seed{s}"]
                    if not (ROOT / path).exists():
                        raise FileNotFoundError(path)
                    out[(m, ds, s)] = path
        return out

    ablation = files(tuple(es.CELLS))
    comparators = files(("GSL", "C1", "C3", "C4"))
    lbunet = [k for k in comparators if k[0] == "C4"]
    counted = set(ablation.values()) | set(comparators.values())
    if len(counted) != len(ablation) + len(comparators):
        raise ValueError("a study-1 run is listed twice")
    ext = {}
    for arm in ("unet", "decoder", "full", "gsl"):
        for s in range(1, 11):
            path = es.isic_path(arm, s).relative_to(ROOT).as_posix()
            if not (ROOT / path).exists():
                raise FileNotFoundError(path)
            ext[(arm, s)] = path
    ext_reused = {k for k, p in ext.items() if p in counted}
    if {s for _, s in ext_reused} != set(seeds13) or len(ext_reused) != 12:
        raise ValueError("ten-run extension: seeds 1-3 are not exactly the study-1 runs")
    fresh = sorted({s for arm, s in ext if (arm, s) not in ext_reused})

    fam = {
        "kdstudents": _inventory_runs("exploratory external KD students"),
        "teachers": _inventory_runs("exploratory external teacher"),
        "replication": _inventory_runs("ACDC confirmation"),
        "controls": _inventory_runs("follow-up E1/E2 students"),
        "compactteacher": _inventory_runs("follow-up compact teacher/cache"),
        "augmentation": _inventory_runs("follow-up E3 augmentation"),
    }
    counts = {
        "ablation": len(ablation),
        "comparators": len(comparators),
        "extension": len(ext) - len(ext_reused),
        **{k: len(v) for k, v in fam.items()},
    }
    slugs = dict(zip(counts, DESIGN_SLOTS, strict=True))
    for slug, n in counts.items():
        if DESIGN_SLOTS[slugs[slug]] != n:
            raise ValueError(
                f"{slug}: {n} runs in the records, {DESIGN_SLOTS[slugs[slug]]} in freeze"
            )
    rows = [
        (
            "1",
            "ablation",
            "matched ablation: 8 arms x 3 datasets x 3 seeds",
            len(ablation),
            0,
            "families/provenance.json; expansion_statistics.cell_path (M7)",
        ),
        (
            "1",
            "comparators",
            f"reference methods: 4 x 3 datasets x 3 seeds, incl. {len(lbunet)} adapted LB-UNet",
            len(comparators),
            0,
            "families/provenance.json (GSL, C1, C3, C4)",
        ),
        (
            "2",
            "extension",
            "U, D, BDS-Lite, U-Net+GSL on ISIC2018, seeds 1-10",
            len(ext),
            len(ext_reused),
            "expansion_statistics.isic_path; seeds 1-3 are study-1 runs",
        ),
        (
            "3",
            "kdstudents",
            "D+KD-band, 3 datasets x 3 seeds",
            counts["kdstudents"],
            0,
            "inventory: exploratory external KD students",
        ),
        (
            "3",
            "teachers",
            "external pretrained teachers, one per dataset",
            counts["teachers"],
            0,
            "inventory: exploratory external teacher",
        ),
        (
            "4",
            "replication",
            "D and D+KD-band on ACDC",
            counts["replication"],
            0,
            "inventory: ACDC confirmation",
        ),
        (
            "5a",
            "controls",
            "D+KD-full and D+KD-band_c on ACDC",
            counts["controls"],
            0,
            "inventory: follow-up E1/E2 students",
        ),
        (
            "5a",
            "compactteacher",
            "compact teacher",
            counts["compactteacher"],
            0,
            "inventory: follow-up compact teacher/cache",
        ),
        (
            "5b",
            "augmentation",
            "D_aug and D+KD-band_aug on ACDC",
            counts["augmentation"],
            0,
            "inventory: follow-up E3 augmentation",
        ),
    ]
    with (GEN / "design_composition.csv").open("w", newline="") as stream:
        w = csv.writer(stream)
        w.writerow(
            [
                "study",
                "group",
                "description",
                "runs_listed",
                "already_counted",
                "runs_counted",
                "source",
            ]
        )
        for study, slug, text, listed, reused, src in rows:
            w.writerow([study, slug, text, listed, reused, listed - reused, src])
    put("FIN.slot.lbunet", str(len(lbunet)))
    put("FIN.teachers", str(counts["teachers"] + counts["compactteacher"]))
    ext_seeds = {s for _, s in ext}
    return {
        "counts": counts,
        "lbunet": len(lbunet),
        "fresh": fresh,
        "fam": fam,
        "n13": len(seeds13),
        "s13": _seed_range(set(seeds13)),
        "n_ext": len(ext_seeds),
        "s_ext": _seed_range(ext_seeds),
        "rows": rows,
    }


# Table 1 layout: chosen by rendered line count (see PLAN.md phase-20 Table 1 entry).
STUDY3_DATASETS = r"\textit{ISIC2018, ACDC, Synapse}"
STUDY_COL, TEST_COL = "3.3cm", "3.2cm"


def studies_table() -> None:
    """Table 1 (5 columns, tabularx + threeparttable notes) and the S15 design-count table.

    Every count and seed range is recounted by design_composition(); see its docstring.
    """
    comp = design_composition()
    c, fam = comp["counts"], comp["fam"]
    n_kd, s_kd = _per_arm(fam["kdstudents"])
    n_cf, s_cf = _per_arm(fam["replication"])
    n_ctl, s_ctl = _per_arm(fam["controls"])
    n_aug, s_aug = _per_arm(fam["augmentation"])
    fresh = _seed_range(set(comp["fresh"]))
    n13, s13, n_ext, s_ext = comp["n13"], comp["s13"], comp["n_ext"], comp["s_ext"]
    if len({r["dataset"] for r in fam["teachers"]}) != c["teachers"]:
        raise ValueError("expected one external teacher per dataset")
    if s_ctl != s_cf:
        raise ValueError("controls do not use the replication seeds")
    n_abl = c["ablation"] // (len(DATASETS) * n13)
    n_ref = c["comparators"] // (len(DATASETS) * n13)
    rp = r"\newline "
    three = r"\textit{ISIC2018, ACDC, Synapse}"
    rows = [
        (
            r"1. Matched ablation and reference methods" + rp + three,
            f"{n_abl} ablation arms; {n_ref} reference methods" + r"\tnote{a}",
            f"{n13} ({s13})",
            r"ISIC2018 test; ACDC \R{DS.acdc.test.units} internal; Synapse \R{DS.synapse.test.units}",
            "Exploratory",
        ),
        (
            r"2. Ten-run extension" + rp + r"\textit{ISIC2018}",
            "U, D, BDS-Lite, U-Net+GSL",
            f"{n_ext} ({s_ext})" + r"\tnote{b}",
            "ISIC2018 test",
            r"\textbf{Confirmatory}",
        ),
        (
            r"3. External-teacher distillation" + rp + STUDY3_DATASETS,
            r"D+KD-band versus D" + r"\tnote{c}",
            f"{n_kd} ({s_kd})",
            "as study 1",
            "Exploratory",
        ),
        (
            r"4. Replication\tnote{d}" + rp + r"\textit{ACDC}",
            "D+KD-band, D",
            f"{n_cf} ({s_cf})",
            r"\R{OT.patients} official ACDC test patients",
            r"\textbf{Confirmatory}",
        ),
        (
            r"5. Controls\tnote{e}" + rp + r"\textit{ACDC}",
            r"D+KD-full, D+KD-band$_{\mathrm{cmp}}$; D$_{\mathrm{aug}}$, D+KD-band$_{\mathrm{aug}}$",
            f"{n_ctl}; {n_aug}",
            "as study 4",
            "Exploratory",
        ),
    ]
    lines = [
        "% generated by analysis/revision/make_final_assets.py::studies_table; do not edit",
        r"\footnotesize\setlength{\tabcolsep}{3.5pt}\hyphenpenalty=10000\exhyphenpenalty=10000",
        r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{"
        + STUDY_COL
        + r"}>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}p{1.75cm}>{\raggedright\arraybackslash}p{"
        + TEST_COL
        + r"}>{\raggedright\arraybackslash}p{1.75cm}@{}}",
        r"\toprule",
        r"Study & Arms & Runs per arm (seeds) & Test data & Evidence \\",
        r"\midrule",
    ]
    for i, row in enumerate(rows):
        if i:
            lines.append(r"\addlinespace")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabularx}"]
    notes = [
        r"\item[a] U-Net+GSL, U-Net with Boundary Loss, Attention U-Net and the multiclass adaptation of LB-UNet (Supplementary S7).",
        rf"\item[b] Seeds {s13} are the study-1 runs, whose results had been seen; seeds {fresh} are new, and the decision rule was fixed before they were trained.",
        rf"\item[c] {n_kd} D+KD-band runs per dataset; D runs reused from studies 1--2 ({n_ext} on ISIC2018, {n13} on ACDC and Synapse); one external teacher per dataset.",
        r"\item[d] ACDC was chosen for the replication because of the results of study 3; the analysis plan was fixed before the test patients were first used.",
        rf"\item[e] D+KD-full and D+KD-band$_{{\mathrm{{cmp}}}}$: {n_ctl} runs each (seeds {s_ctl}), compared with the study-4 runs; D$_{{\mathrm{{aug}}}}$ and D+KD-band$_{{\mathrm{{aug}}}}$: {n_aug} runs each (seeds {s_aug}). Analysed after study 4, on the same patients.",
    ]
    lines += [r"\begin{tablenotes}\footnotesize"] + notes + [r"\end{tablenotes}"]
    if sum(c.values()) != sum(DESIGN_SLOTS.values()):
        raise ValueError("table total differs from DESIGN_SLOTS")
    write("tab_studies.tex", lines)

    # Supplementary S15: design count, one row per group, from design_composition.csv
    study_name = {
        "1": "1. Matched ablation and reference methods",
        "2": "2. Ten-run extension",
        "3": "3. External-teacher distillation",
        "4": "4. Replication",
        "5a": "5. Controls",
        "5b": "5. Controls",
    }
    label = {
        "ablation": f"{n_abl} ablation arms $\\times$ {len(DATASETS)} datasets $\\times$ {n13} seeds",
        "comparators": f"{n_ref} reference methods $\\times$ {len(DATASETS)} $\\times$ {n13}, including {comp['lbunet']} runs of the adapted LB-UNet",
        "extension": f"U, D, BDS-Lite, U-Net+GSL $\\times$ seeds {s_ext}; seeds {s13} are study-1 runs",
        "kdstudents": f"D+KD-band $\\times$ {len(DATASETS)} datasets $\\times$ {n_kd} seeds",
        "teachers": "external pretrained teachers, one per dataset",
        "replication": f"D+KD-band, D $\\times$ seeds {s_cf}",
        "controls": f"D+KD-full, D+KD-band$_{{\\mathrm{{cmp}}}}$ $\\times$ seeds {s_ctl}",
        "compactteacher": "compact teacher",
        "augmentation": f"D$_{{\\mathrm{{aug}}}}$, D+KD-band$_{{\\mathrm{{aug}}}}$ $\\times$ seeds {s_aug}",
    }
    s15 = [
        "% generated by analysis/revision/make_final_assets.py::studies_table; do not edit",
        r"\begin{tabular}{@{}llS[table-format=3.0]S[table-format=2.0]S[table-format=3.0]@{}}",
        r"\toprule",
        r"Study & Runs & {Listed} & {Already counted} & {Counted} \\",
        r"\midrule",
    ]
    prev = None
    for study, slug, _text, listed, reused, _src in comp["rows"]:
        name = study_name[study] if study_name[study] != prev else ""
        prev = study_name[study]
        s15.append(f"{name} & {label[slug]} & {listed} & {reused} & {listed - reused} \\\\")
    s15 += [
        r"\midrule",
        f"Total & & {sum(r[3] for r in comp['rows'])} & {sum(r[4] for r in comp['rows'])} & {sum(c.values())} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    write("tab_s_design_count.tex", s15)


def main() -> None:
    studies_table()
    matched_table()
    innetwork_table()
    exploratory_kd()  # keys for Supplementary S17
    data = acdc_official()
    e3_supplement()
    resource_table()
    ten_run_means()
    plain_lit_table()
    resource_main_table()
    published_table()
    put("FIN.runs.total", str(sum(DESIGN_SLOTS.values())))
    slugs = [
        "ablation",
        "comparators",
        "extension",
        "kdstudents",
        "teachers",
        "replication",
        "controls",
        "compactteacher",
        "augmentation",
    ]
    for slug, n in zip(slugs, DESIGN_SLOTS.values(), strict=True):
        put(f"FIN.slot.{slug}", str(n))
    splits = ROOT / "recovered/followup/splits"
    for ds in ("isic2018", "acdc", "synapse"):
        for part in ("train", "val", "test"):
            path = splits / ds / f"{part}.json"
            if path.exists():
                put(f"SPL.{ds}.{part}", f"{json.loads(path.read_text())['count']:,}")
    if KEYS["FIN.runs.total"] != "237":
        raise ValueError("design count differs from evidence freeze section 3")
    figure(data)
    inventory_table()
    teacher_eval()
    lines = ["% generated by analysis/revision/make_final_assets.py; do not edit"]
    lines += [f"\\expandafter\\def\\csname R@{k}\\endcsname{{{KEYS[k]}}}" for k in sorted(KEYS)]
    (GEN / "numbers_final.tex").write_text("\n".join(lines) + "\n")
    (GEN / "keys_final.json").write_text(json.dumps(KEYS, indent=0, sort_keys=True) + "\n")
    print(len(KEYS), "keys")


if __name__ == "__main__":
    main()
