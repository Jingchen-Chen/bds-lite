# ruff: noqa: E501
"""Main-text presentation layer (2026-11-12): three decimals and one result vocabulary.

The frozen generators write four decimals and the registered reading labels; the supplement keeps
both. For the main text, the response letter and the editor letter this script derives
  generated/numbers_round3.tex  - every key used by those documents, with each four-decimal number
                                  rounded to three decimals (loaded last, so it overrides);
  generated/r3_<table>.tex      - the four main-text tables with three decimals and the result
                                  words of the main-text Statistics (confirmed / detected /
                                  not detected).
No statistic is recomputed. A four-decimal string ending in 5 cannot be rounded safely from the
string, so each such case is listed in TIES with the three-decimal rounding of the exact source
value (source noted beside it); an unlisted tie stops the script.

    PYTHONPATH=src:. .venv/bin/python analysis/revision/make_main_round3.py
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "revision/manuscript/generated"
ORDER = [
    "numbers.tex",
    "numbers_extra.tex",
    "numbers_expansion.tex",
    "numbers_audit.tex",
    "numbers_kd.tex",
    "numbers_restructure.tex",
    "numbers_followup.tex",
    "numbers_final.tex",
    "numbers_review.tex",
]
USERS = [
    "revision/manuscript/main.tex",
    "revision/response.template.md",
    "revision/letter_to_editor.template.md",
    "revision/PEERJ_REQUIREMENTS_CHECKLIST.template.md",
]
NUMBER = re.compile(r"(?<![\d.])(\d+\.\d{4})(?!\d)")
# four-decimal string -> three decimals of the exact value
KEY_TIES = {
    "0.7975": "0.798",  # CF.kd.mean 0.797511 (followup contrasts.csv, S3 arm_a_mean)
    "0.0005": "0.001",  # FX.C1 low -0.0005185
    "0.0105": "0.011",  # FX.S3 IoU low 0.0105017
    "0.0265": "0.027",  # decoder pooled high 0.0265101 (expansion_interaction.csv)
    "0.0045": "0.004",  # surface without decoder, pooled 0.0044689
    "0.0125": "0.013",  # transfer with surface, pooled high 0.0125316
    "0.0215": "0.022",  # interaction 0.0215287 (posthoc_review_20261112/interaction.csv)
    "0.0075": "0.007",  # interaction, seeds 201-210, low 0.0074540
    "0.7735": "0.773",  # compact teacher 0.7734649 (results/teacher_eval/summary.json)
    "0.0405": "0.040",  # BDS-Lite - D, fresh seeds -0.0404893 (expansion_contrasts.csv)
    "0.0095": "0.009",  # BDS-Lite - U high 0.0094981
    "0.0225": "0.022",  # BDS-Lite SD 0.0224753
}
TABLE_TIES = {
    "tab_matched_main.tex": {
        "0.0165": "0.017",  # ISIC2018 U Boundary F1 SD 0.0165274 (comparison_absolute_scores.csv)
        "0.0055": "0.005",  # ISIC2018 Boundary Loss DSC SD 0.0054654
        "0.5285": "0.528",  # Synapse D+F+G DSC 0.5284894 (method_summary.csv)
        "0.5325": "0.532",  # Synapse GSL DSC 0.5324965
        "0.5555": "0.556",  # Synapse GSL Boundary F1 0.5555468
        "0.0095": "0.009",  # Synapse Attention U-Net DSC SD 0.0094896
    },
    "tab_innetwork_main.tex": {
        "0.0265": "0.027",
        "0.0215": "0.021",  # D+F+G - D, Synapse 0.0214634
        "0.0515": "0.052",  # its low -0.0515200
        "0.0175": "0.018",  # BDS-Lite - D+S, ACDC low -0.0175486
        "0.0125": "0.013",
        "0.0045": "0.004",
        "0.0095": "0.009",
        "0.0405": "0.040",
    },
    "tab_acdc_official_main.tex": {
        "0.0005": "0.001",
        "0.0035": "0.003",  # whole-image - D, DSC low 0.0034960 (followup contrasts.csv)
        "0.0165": "0.017",  # its high 0.0165217
        "0.0175": "0.018",  # compact - D, DSC high 0.0175344
    },
    "tab_published_review.tex": {},
}
INNETWORK = [
    (
        r"Contrast & {ISIC2018} & {ACDC} & {Synapse} & {Pooled} & Reading \\",
        r"Contrast & {ISIC2018} & {ACDC} & {Synapse} & {Pooled} & Result \\",
    ),
    (r"\multirow[t]{2}{=}{gain of unclear size}", r"\multirow[t]{2}{=}{detected}"),
    (r"\multirow[t]{2}{=}{undetermined}", r"\multirow[t]{2}{=}{not detected}"),
    (
        r"\item[] Reading of the pooled interval against the 0.01 SESOI: gain of unclear size, excludes zero but overlaps the SESOI; undetermined, includes zero and extends beyond $\pm$SESOI.",
        r"\item[] Result of the pooled interval: detected, the 95\% interval excludes zero; not detected, it includes zero.",
    ),
    ("& 0.260 & supported, not confirmed \\\\", "& 0.260 & detected, not confirmed \\\\"),
    ("not established \\\\", "not detected \\\\"),
    (
        r"\item[] Status: \textbf{confirmed}, fresh-seed Holm $p<0.05$ and fresh-seed interval excluding zero in the direction of the ten-run estimate; supported, not confirmed, only the ten-run interval excludes zero; not established, neither.",
        r"\item[] Result: \textbf{confirmed}, fresh-seed Holm $p<0.05$ and fresh-seed interval excluding zero in the direction of the ten-run estimate; detected, not confirmed, only the ten-run interval excludes zero; not detected, neither.",
    ),
    ("{Holm $p$} & Status \\\\", "{Holm $p$} & Result \\\\"),
    (
        r"-0.0093 & -0.0459 & -0.0713 & -0.0204 & 0.014 & \textbf{confirmed} \\",
        r"-0.0093 & -0.0459 & -0.0713 & -0.0204 & 0.014 & \textbf{confirmed}\tnote{b} \\",
    ),
    (
        r"only the ten-run interval excludes zero; not detected, neither.",
        r"only the ten-run interval excludes zero; not detected, neither."
        "\n"
        r"\item[b] Prespecified reading: a deficit of unclear size; the upper bound of the ten-run interval, $-$0.009, lies inside the preset $\pm$0.01 (Supplementary S13).",
    ),
]
OFFICIAL = [
    (r"{$p$\tnote{a}} & Reading &", r"{$p$\tnote{a}} & Result &"),
    (r"\multirow[t]{2}{=}{unresolved}", r"\multirow[t]{2}{=}{not detected}"),
    (r"\multirow[t]{2}{=}{unresolved\tnote{b}}", r"\multirow[t]{2}{=}{not detected\tnote{b}}"),
    (r"\multirow[t]{2}{=}{undetermined}", r"\multirow[t]{2}{=}{not detected}"),
    (
        r"\multicolumn{9}{@{}l}{\emph{Replication} (study 4)}",
        r"\multicolumn{9}{@{}l}{\emph{Replication} (confirmatory)}",
    ),
    (
        r"\multicolumn{9}{@{}l}{\emph{Controls} (study 5)}",
        r"\multicolumn{9}{@{}l}{\emph{Controls} (exploratory)}",
    ),
    (
        r"\multicolumn{9}{@{}l}{\emph{Augmentation comparison} (study 5)}",
        r"\multicolumn{9}{@{}l}{\emph{Augmentation comparison} (exploratory)}",
    ),
    (
        r"\item[a] Replication and augmentation: $p$ of the single primary contrast; C1--C4: Holm-adjusted $p$ over the four controls.",
        r"\item[a] Replication and augmentation: $p$ of the single primary contrast; controls: Holm-adjusted $p$ over the four contrasts.",
    ),
    (
        r"\item[b] The unadjusted interval excludes zero, but the Holm-adjusted $p$ is \R{FU.C4.boundary_f1.holm}.",
        r"\item[b] The unadjusted interval excludes zero, but the Holm-adjusted $p$ is \R{FU.C4.boundary_f1.holm}."
        "\n"
        r"\item[c] Prespecified wording: a gain of unclear size; the lower bound, \R{RAW.FIN.cf.low}, lies below the preset 0.01 (Supplementary S18).",
    ),
]


def keys() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ORDER:
        for key, value in re.findall(
            r"\\csname R@(.+?)\\endcsname\{(.*)\}$", (GEN / name).read_text(), re.M
        ):
            out[key] = value
    return out


def used() -> set[str]:
    found: set[str] = set()
    for name in USERS:
        text = (ROOT / name).read_text()
        found |= set(re.findall(r"\\R\{([^}]+)\}", text))
        found |= set(re.findall(r"\{\{(?:R:)?([A-Za-z0-9_.+-]+)\}\}", text))
    return found


def rounder(ties: dict[str, str], where: str):
    def one(match: re.Match) -> str:
        text = match.group(1)
        if text.endswith("5"):
            if text not in ties:
                raise ValueError(f"unresolved rounding tie {text} in {where}")
            return ties[text]
        return f"{float(text):.3f}"

    return one


def swap(text: str, pairs: list[tuple[str, str]], where: str) -> str:
    for old, new in pairs:
        if old not in text:
            raise ValueError(f"{where}: expected text not found: {old[:60]}")
        text = text.replace(old, new)
    return text


def tables() -> None:
    for name, ties in TABLE_TIES.items():
        text = (GEN / name).read_text()
        if name == "tab_innetwork_main.tex":
            text = swap(text, INNETWORK, name)
        if name == "tab_acdc_official_main.tex":
            if text.count(r"\multirow[t]{2}{=}{gain of unclear size}") != 3:
                raise ValueError("official table layout changed")
            text = text.replace(
                r"\multirow[t]{2}{=}{gain of unclear size}",
                r"\multirow[t]{2}{=}{\textbf{confirmed}\tnote{c}}",
                1,
            )
            text = text.replace(
                r"\multirow[t]{2}{=}{gain of unclear size}", r"\multirow[t]{2}{=}{detected}"
            )
            text = swap(text, OFFICIAL, name)
            text, n = re.subn(r"C[1-4]\\enspace ", "", text)
            text, m = re.subn(r"\\phantom\{C[1-4]\}\\enspace ", "", text)
            if (n, m) != (4, 4):
                raise ValueError("official table: control tags not found")
        text = NUMBER.sub(rounder(ties, name), text)
        text = text.replace("table-format=+1.4", "table-format=+1.3").replace(
            "table-format=1.4(3)", "table-format=1.3(3)"
        )
        (GEN / f"r3_{name}").write_text(
            "% three-decimal main-text variant written by analysis/revision/make_main_round3.py\n"
            + text
        )


def main() -> None:
    values, lines, count = (
        keys(),
        ["% generated by analysis/revision/make_main_round3.py; main text and letters only"],
        0,
    )
    for key in sorted(used()):
        if key not in values:
            continue
        new = NUMBER.sub(rounder(KEY_TIES, key), values[key])
        if new != values[key]:
            lines.append(rf"\expandafter\def\csname R@{key}\endcsname{{{new}}}")
            count += 1
    # the unrounded lower bound of the replication is quoted once in the main text
    lines.append(rf"\expandafter\def\csname R@RAW.FIN.cf.low\endcsname{{{values['FIN.cf.low']}}}")
    teacher = int(values["KDC.acdc.teacher.params"].replace(",", ""))
    lines.append(rf"\expandafter\def\csname R@R3.teacher.m\endcsname{{{teacher / 1e6:.1f}}}")
    lines.append(
        rf"\expandafter\def\csname R@R3.teacher.ratio\endcsname{{{round(float(values['RS.KDC.ratio.deployed']))}}}"
    )
    (GEN / "numbers_round3.tex").write_text("\n".join(lines) + "\n")
    tables()
    print(f"{count} keys rounded to three decimals")


if __name__ == "__main__":
    main()
