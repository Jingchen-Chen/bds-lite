# ruff: noqa: E501
"""Supplement presentation layer (2026-11-15): the arm names, study numbers and result labels of
the main text, applied to tables written by the frozen generators.

The generators listed in the SHA-256 manifests of the two evidence archives are not edited. This
script reads their output in generated/, renames labels only, and writes generated/sl_<name>.tex,
which supplement.tex inputs instead. Rules:
  * arm names follow main-text Table 3 and Table S4 (D, U+S, D+F+G, D+G+S, D+F+S, BDS-Lite,
    U-Net+GSL, D+KD-band_cmp); contrasts are written "A - B";
  * studies are numbered as in main-text Table 2 (stage 1 -> study 1 or 2, stage 2 -> study 3,
    stage 3 -> study 4, follow-up -> study 5); the analysis-plan names C1-C4 and E3 are dropped;
  * status labels follow the main-text Statistics (supported -> detected, not confirmed;
    not established -> not detected; harm -> deficit);
  * a reading against the preset smallest effect of interest (SESOI) is re-derived from the printed
    interval by the rules of Table S1 wherever a table used the older words (within SESOI,
    beyond, inconclusive), and every other printed reading is checked against its interval.
Every number of a source table must occur unchanged and in the same order in its output; the
script stops otherwise, and also when an expected label is not found.

    PYTHONPATH=src:. .venv/bin/python analysis/revision/make_supplement_labels.py
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "paper/generated"
M = "$-$"
SESOI = {"HD95": 2.0, "ASSD": 0.5}  # all overlap and boundary metrics: 0.01
READING = "Reading against $\\pm$0.01 (prespecified)"

# Arm labels of the frozen generators (method_summary.csv and family tables) -> main-text names.
ARMS = [
    ("U-Net + aux. boundary only", "D"),
    ("U-Net + SDF only", "U+S"),
    ("BDS-Lite - gate (alpha=0)", "D+F+S"),
    ("BDS-Lite - distillation", "D+G+S"),
    ("BDS-Lite - SDF", "D+F+G"),
    ("BDS-Lite (full)", "BDS-Lite"),
    ("U-Net + GSL", "U-Net+GSL"),
    ("U-Net + Boundary Loss", "U-Net+Boundary Loss"),
]
RUNLEVEL_ARMS = [
    ("U-Net + surface loss", "U+S"),
    ("U-Net + boundary decoder", "D"),
    (f"BDS-Lite {M} surface loss", "D+F+G"),
    (f"BDS-Lite {M} distillation", "D+G+S"),
    (f"BDS-Lite {M} gate", "D+F+S"),
    ("U-Net + GSL", "U-Net+GSL"),
    ("U-Net + Boundary Loss", "U-Net+Boundary Loss"),
]
STUDIES = [
    (
        r"\emph{Stage 3, secondary metrics (descriptive)}",
        r"\emph{Study 4, secondary metrics (descriptive)}",
    ),
    ("Stage 1 &", "Study 1 &"),
    ("ISIC2018 ten-run &", "Study 2 &"),
    ("Stage 2 (3 vs 3) &", "Study 3 (3 vs 3) &"),
    ("Stage 2 (3 vs 10) &", "Study 3 (3 vs 10) &"),
    ("Stage 3 &", "Study 4 &"),
    ("D+F+G+S $-$", "BDS-Lite $-$"),
    ("$-$ D+F+G+S", "$-$ BDS-Lite"),
]
CONTROLS = [
    ("Stage 3: D+KD-band $-$ D", "Replication: D+KD-band $-$ D"),
    ("C1: D+KD-band $-$ D+KD-full", "D+KD-band $-$ D+KD-full"),
    ("C2: D+KD-full $-$ D", "D+KD-full $-$ D"),
    ("C3: D+KD-band$_{\\mathrm{cmp}}$ $-$ D", "D+KD-band$_{\\mathrm{cmp}}$ $-$ D"),
    ("C4: D+KD-band $-$ D+KD-band$_{\\mathrm{cmp}}$", "D+KD-band $-$ D+KD-band$_{\\mathrm{cmp}}$"),
]
TEN_RUN = [
    ("U-Net + decoder vs U-Net", f"D {M} U"),
    ("BDS-Lite vs U-Net + decoder", f"BDS-Lite {M} D"),
    ("BDS-Lite vs U-Net", f"BDS-Lite {M} U"),
    ("U-Net + GSL vs U-Net + decoder", f"U-Net+GSL {M} D"),
    ("U-Net + GSL vs BDS-Lite", f"U-Net+GSL {M} BDS-Lite"),
    ("U-Net + GSL vs U-Net", f"U-Net+GSL {M} U"),
]
SHORT_CONTRASTS = [
    ("Decoder vs U-Net", f"D {M} U"),
    ("BDS-Lite vs U-Net", f"BDS-Lite {M} U"),
    ("BDS-Lite vs Decoder", f"BDS-Lite {M} D"),
    ("BDS-Lite vs decoder", f"BDS-Lite {M} D"),
    ("GSL vs U-Net", f"U-Net+GSL {M} U"),
    ("GSL vs BDS-Lite", f"U-Net+GSL {M} BDS-Lite"),
    ("GSL vs Decoder", f"U-Net+GSL {M} D"),
    ("GSL vs decoder", f"U-Net+GSL {M} D"),
]
STATUS = [
    ("supported, not confirmed", "detected, not confirmed"),
    ("(supported)", "(detected, not confirmed)"),
    ("not established", "not detected"),
]
WORDS = [
    ("gain, size unclear", "gain of unclear size"),
    ("harm, size unclear", "deficit of unclear size"),
    ("relevant harm", "relevant deficit"),
]
# Development arms of S15 (letters of the development study) -> component letters.
# D_0: boundary decoder trained without a boundary loss (feature matching only).
DEV = [
    (
        "B vs C: distillation given decoder and gate",
        f"D+F+G {M} D+G: distillation given decoder and gate",
    ),
    ("C vs G: decoder given gate", f"D+G {M} U+G: decoder given gate"),
    ("G vs U-Net: gate and projection", f"U+G {M} U: gate and projection"),
    (
        "D vs G: matching without boundary supervision",
        f"D$_0$+F+G {M} U+G: matching without boundary supervision",
    ),
    (
        "B vs D: boundary supervision given matching",
        f"D+F+G {M} D$_0$+F+G: boundary supervision given matching",
    ),
    ("C vs decoder: gate without distillation", f"D+G {M} D: gate without distillation"),
    ("B vs decoder: gate with distillation", f"D+F+G {M} D: gate with distillation"),
    ("Interaction (B$-$C)$-$(D$-$G)", "Interaction (D+F+G $-$ D+G) $-$ (D$_0$+F+G $-$ U+G)"),
]
APRIL = [
    ("Decoder + distillation vs decoder", f"D+F {M} D"),
    ("Decoder + gate vs decoder", f"D+G {M} D"),
    ("Full vs decoder + gate + surface", f"BDS-Lite {M} D+G+S"),
    ("Full vs decoder", f"BDS-Lite {M} D"),
    ("Decoder vs U-Net", f"D {M} U"),
]
FACTORIAL = [
    ("Distillation+gate interaction with surface", f"(BDS-Lite {M} D+S) {M} (D+F+G {M} D)"),
    ("Distillation+gate without surface", f"D+F+G {M} D"),
    ("Distillation+gate with surface", f"BDS-Lite {M} D+S"),
    ("Decoder without surface", f"D {M} U"),
    ("Decoder with surface", f"D+S {M} U+S"),
    ("Surface without decoder", f"U+S {M} U"),
    ("Surface with decoder", f"D+S {M} D"),
    ("Interaction &", f"(D+S {M} D) {M} (U+S {M} U) &"),
]
CENTRAL = [
    ("Decoder vs U-Net", f"D {M} U"),
    ("Transfer, no surface loss", f"D+F+G {M} D"),
    ("Transfer, with surface loss", f"BDS-Lite {M} D+S"),
]
POOLED_SPLIT = [("Pooled (label)", f"Pooled & {READING}")]
# The plain U-Net as an arm name; architecture names (Attention U-Net, ResNet-34 U-Net) and the
# comparators U-Net+GSL and U-Net+Boundary Loss are kept.
PLAIN = re.compile(r"(?<![\w+-])(?<!Attention )(?<!34 )U-Net(?![\w+-]| \+)")

# file -> (replacement lists, options)
RULES: dict[str, tuple[list[list[tuple[str, str]]], dict]] = {
    "rl_tab_s_duplicates.tex": ([[("SESOI judgement", READING)]], {"rejudge": True}),
    "tab_ablation_means.tex": ([ARMS[:6]], {}),
    "tab_all_arms_runlevel.tex": ([RUNLEVEL_ARMS], {}),
    "tab_comparator_means.tex": ([[ARMS[5], ARMS[6], ARMS[7]]], {}),
    "tab_lit_main_plain.tex": ([[("Decoder-only", "D"), ARMS[5], ARMS[6], ARMS[7]]], {}),
    "rl_tab_s_errorbands.tex": ([ARMS], {}),
    "tab_s_collapse_wide.tex": ([ARMS], {}),
    "tab_central.tex": ([CENTRAL, POOLED_SPLIT, WORDS], {"split_pooled": "llllll", "check": True}),
    "tab_s_ten_seed_all.tex": (
        [TEN_RUN, STATUS, WORDS, [("Effect label", READING)]],
        {"check": True},
    ),
    "tab_s_interaction.tex": ([FACTORIAL], {}),
    "tab_s_validation.tex": ([[SHORT_CONTRASTS[i] for i in (0, 1, 2, 4, 5, 6)]], {}),
    "tab_components.tex": ([POOLED_SPLIT, WORDS], {"split_pooled": "llllll", "check": True}),
    "tab_s_seed_subset.tex": ([[("U-Net + decoder", "D"), ("U-Net + GSL", "U-Net+GSL")]], {}),
    "tab_s_hist_april.tex": ([APRIL], {}),
    "tab_s_hist_october.tex": ([DEV], {}),
    "tab_s_tolerance.tex": ([[SHORT_CONTRASTS[i] for i in (0, 1, 3, 4, 5, 7)], STATUS[1:]], {}),
    "tab_s_kd_all.tex": ([[("Decoder-only", "D")]], {}),
    "tab_s_kd_costs.tex": ([[("(decoder-only reference)", "(reference arm D)")]], {}),
    "tab_kd.tex": (
        [
            [
                ("Stage &", "Study &"),
                ("2. KD exploration", "3. External-teacher distillation"),
                ("3. ACDC confirmation", "4. Replication"),
            ]
        ],
        {},
    ),
    "tab_s_confirm.tex": ([[("not established", "not detected")]], {}),
    "tab_s20.tex": ([STUDIES], {}),
    "tab_s20_coverage.tex": (
        [
            [
                s
                for s in STUDIES
                if s[0]
                not in (STUDIES[0][0], "ISIC2018 ten-run &", "Stage 2 (3 vs 10) &", "$-$ D+F+G+S")
            ]
        ],
        {},
    ),
    "tab_s_followup_all.tex": (
        [CONTROLS, WORDS, [("& Reading &", f"& {READING} &"), ("ID &", "Comparison &")]],
        {"check": True},
    ),
    "tab_s_followup_perclass.tex": ([CONTROLS], {}),
    "tab_s_reference_estimators.tex": ([CONTROLS], {}),
    "tab_s_e3_all.tex": (
        [
            [
                (
                    "E3: D+KD-band$_{\\mathrm{aug}}$ $-$ D$_{\\mathrm{aug}}$ (10 vs 10)",
                    "Distillation under augmentation: D+KD-band$_{\\mathrm{aug}}$ $-$ D$_{\\mathrm{aug}}$ (10 vs 10)",
                ),
                (
                    "E3a: D$_{\\mathrm{aug}}$ $-$ D (10 vs 17)",
                    "Augmentation of D: D$_{\\mathrm{aug}}$ $-$ D (10 vs 17)",
                ),
                (
                    "E3b: D+KD-band$_{\\mathrm{aug}}$ $-$ D+KD-band (10 vs 17)",
                    "Augmentation of D+KD-band: D+KD-band$_{\\mathrm{aug}}$ $-$ D+KD-band (10 vs 17)",
                ),
                ("& Reading &", f"& {READING} &"),
            ],
            WORDS,
        ],
        {"check": True},
    ),
    # tables relabelled only for the plain U-Net (U)
    "tab_s_original.tex": ([], {"plain": True}),
    "tab_s_run_level.tex": ([], {"plain": True}),
    "tab_s_selected_epochs.tex": ([], {"plain": True}),
    "tab_ten_seed_final.tex": ([], {"plain": True}),
    "tab_two_sets.tex": ([], {"plain": True}),
    "tab_s_development.tex": ([], {"plain": True}),
    "tab_s_review_pooling.tex": ([[("& Reading \\\\", f"& {READING} \\\\")]], {"check": True}),
}

# Text-valued keys of the frozen numbers files that the supplement prints.
KEY_STATUS = {
    "supported": "detected, not confirmed",
    "supported, not confirmed": "detected, not confirmed",
    "not established": "not detected",
    "confirmed": "confirmed",
}
KEY_MAP = {
    "T.4.decoder.unet.status": lambda v: KEY_STATUS[v],
    "T.4.full.decoder.status": lambda v: KEY_STATUS[v],
    "RS.flips.other": lambda v: v.replace("Stage 2 (3 vs 3)", "Study 3 (3 vs 3)"),
}

LABELS = re.compile(
    r"(?:Stage|Study|stage|study)[ -]?\d|\b[CE]\d[ab]?\b|\bM\d\b|\$_0\$|\$\\pm\$0\.01|\(alpha=0\)|ISIC2018 ten-run|(?m:^\d\. )"
)
NUM = re.compile(r"\d+(?:\.\d+)?")


def numbers(text: str) -> list[str]:
    return NUM.findall(LABELS.sub(" ", text))


def value(text: str) -> float:
    """A printed number such as +0.0178, $-$0.0000 or 1.22; signed zero keeps its sign."""
    text = text.strip().replace("$-$", "-").replace("−", "-")
    v = float(text)
    if v == 0.0:
        return -1e-12 if text.startswith("-") else 1e-12
    return v


def reading(low: float, high: float, sesoi: float) -> str:
    """Reading of a printed 95% interval against the SESOI (Table S1)."""
    if -sesoi < low and high < sesoi:
        return "no relevant effect"
    if low > sesoi:
        return "relevant gain"
    if high < -sesoi:
        return "relevant deficit"
    if low > 0:
        return "gain of unclear size"
    if high < 0:
        return "deficit of unclear size"
    return "undetermined"


INTERVAL = re.compile(r"\[\s*((?:\$-\$|[+-])?[\d.]+),\s*((?:\$-\$|[+-])?[\d.]+)\s*\]")
METRIC = re.compile(r"\b(HD95|ASSD|DSC|IoU|Boundary F1|BF1)\b")
KNOWN = (
    "no relevant effect",
    "relevant gain",
    "relevant deficit",
    "gain of unclear size",
    "deficit of unclear size",
    "undetermined",
)


def _csv(path: Path, **match: str) -> dict:
    with path.open() as stream:
        rows = [
            r for r in csv.DictReader(stream) if all(r[k].startswith(v) for k, v in match.items())
        ]
    if len(rows) != 1:
        raise ValueError(f"{path.name}: {len(rows)} rows match {match}")
    return rows[0]


FAMILIES = ROOT / "results/tables/families"
# Printed bounds that equal +-SESOI at the printed precision are judged on the unrounded interval
# of the archived analysis output: (table, metric, printed interval) -> (source, low, high).
UNROUNDED = {
    ("tab_central.tex", "Boundary F1", "[$-$0.010, +0.013]"): lambda: (
        "expansion_interaction.csv, distillation+gate with surface, pooled",
        _csv(
            FAMILIES / "expansion_interaction.csv",
            effect="distillation+gate with surface",
            metric="boundary_f1",
            dataset="pooled",
        ),
        "low",
        "high",
    ),
    ("tab_s_ten_seed_all.tex", "DSC", "[$-$0.010, +0.001]"): lambda: (
        "expansion_contrasts.csv, dsc, full - decoder, seeds 1-10",
        _csv(FAMILIES / "expansion_contrasts.csv", metric="dsc", first="full", second="decoder"),
        "all_low",
        "all_high",
    ),
    ("tab_s_ten_seed_all.tex", "IoU", "[$-$0.007, +0.010]"): lambda: (
        "expansion_contrasts.csv, iou, gsl - full, seeds 1-10",
        _csv(FAMILIES / "expansion_contrasts.csv", metric="iou", first="gsl", second="full"),
        "all_low",
        "all_high",
    ),
    ("tab_components.tex", "Boundary F1", "[$-$0.008, +0.010]"): lambda: (
        "expansion_components.csv, boundary_f1, distill, pooled",
        _csv(
            FAMILIES / "expansion_components.csv",
            dataset="pooled",
            metric="boundary_f1",
            component="distill",
        ),
        "low",
        "high",
    ),
    ("tab_s_followup_all.tex", "ASSD", "[$-$0.14, +0.50]"): lambda: (
        "followup/results/contrasts.csv, C4, assd",
        _csv(ROOT / "results/followup/results/contrasts.csv", id="C4", metric="assd"),
        "low",
        "high",
    ),
}


def unrounded(name: str, metric: str, printed: str, sesoi: float) -> tuple[str, str]:
    source, row, lo, hi = UNROUNDED[(name, metric, printed)]()
    low, high = float(row[lo]), float(row[hi])
    return reading(low, high, sesoi), f"{source}: [{low:+.6f}, {high:+.6f}]"


def rows_with_readings(
    text: str, rejudge: bool, last: bool, name: str = ""
) -> tuple[str, list[str], list[str]]:
    """Re-derive (rejudge) or check every reading against the interval of its row.

    The interval is the first of the row, or the last one for tables whose reading belongs to the
    pooled column. A bound printed exactly at +-SESOI is a tie at the printed precision: it is
    judged on the unrounded interval of the archived output (UNROUNDED), and the printed label is
    replaced if that judgement differs.
    """
    out, problems, ties, metric = [], [], [], None
    for line in text.splitlines():
        found = METRIC.search(line)
        if found:
            metric = found.group(1)
        hits = list(INTERVAL.finditer(line))
        if hits and line.rstrip().endswith("\\\\") and metric:
            hit = hits[-1] if last else hits[0]
            sesoi = SESOI.get(metric, 0.01)
            low, high = value(hit.group(1)), value(hit.group(2))
            expected = {reading(low, high, sesoi)}
            tie = abs(abs(low) - sesoi) < 1e-9 or abs(abs(high) - sesoi) < 1e-9
            if tie:
                if (name, metric, hit.group(0)) not in UNROUNDED:
                    raise ValueError(f"{name}: tie without an unrounded source: {line}")
                judged, source = unrounded(name, metric, hit.group(0), sesoi)
                expected = {judged}
                printed = [
                    w for w in KNOWN if re.search(rf"(?<![\w-]){re.escape(w)}(?![\w-])", line)
                ]
                if not rejudge and printed and judged not in printed:
                    line = line.replace(printed[0], judged)
                ties.append(f"{metric} {hit.group(0)} -> {source} -> {judged} (printed {printed})")
            if rejudge:
                if len(expected) > 1:
                    raise ValueError(f"tie at the printed precision, decide by hand: {line}")
                cells = line.rstrip()[:-2].rstrip().split(" & ")
                cells[-1] = expected.pop()
                line = " & ".join(cells) + " \\\\"
            else:
                labels = [
                    w for w in KNOWN if re.search(rf"(?<![\w-]){re.escape(w)}(?![\w-])", line)
                ]
                if labels and not expected & set(labels):
                    problems.append(
                        f"{metric}: {hit.group(0)} printed {labels}, Table S1 gives {expected}"
                    )
        out.append(line)
    return "\n".join(out) + ("\n" if text.endswith("\n") else ""), problems, ties


TIES: list[str] = []


def apply(name: str, lists: list[list[tuple[str, str]]], options: dict) -> list[str]:
    source = (GEN / name).read_text()
    text = source
    for pairs in lists:
        for old, new in pairs:
            if old not in text and (old, new) not in WORDS + STATUS:
                raise ValueError(f"{name}: label not found: {old!r}")
            text = text.replace(old, new)
    text, n_plain = PLAIN.subn("U", text)
    if options.get("plain") and n_plain == 0:
        raise ValueError(f"{name}: no plain U-Net arm name found")
    if "split_pooled" in options:
        spec = options["split_pooled"]
        text = text.replace(f"\\begin{{tabular}}{{{spec}}}", f"\\begin{{tabular}}{{{spec}l}}", 1)
        words = "|".join(re.escape(w) for w in KNOWN)
        text, n = re.subn(rf" \(({words})\) \\\\$", r" & \1 \\\\", text, flags=re.M)
        if n == 0:
            raise ValueError(f"{name}: no pooled reading found")
    problems: list[str] = []
    if options.get("rejudge") or options.get("check"):
        text, problems, ties = rows_with_readings(
            text, bool(options.get("rejudge")), "split_pooled" in options, name
        )
        TIES.extend(f"{name}: {t}" for t in ties)
    for word in ("within SESOI", "beyond", "inconclusive", "harm", "supported", "not established"):
        if word in text:
            raise ValueError(f"{name}: old label {word!r} remains")
    if numbers(source) != numbers(text):
        a, b = numbers(source), numbers(text)
        i = next(
            (k for k, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b))
        )
        raise ValueError(
            f"{name}: numbers changed near token {i}: {a[i : i + 5]} vs {b[i : i + 5]}"
        )
    header = f"% labels of the main text applied by analysis/revision/make_supplement_labels.py to {name}\n"
    (GEN / f"sl_{name}").write_text(header + text)
    return problems


def keys() -> None:
    values = {}
    for path in sorted(GEN.glob("numbers*.tex")):
        for k, v in re.findall(r"\\csname R@(.+?)\\endcsname\{(.*)\}$", path.read_text(), re.M):
            values.setdefault(k, v)
    lines = ["% generated by analysis/revision/make_supplement_labels.py; supplement only"]
    for key, mapping in KEY_MAP.items():
        text = mapping(values[key])
        if numbers(text) != numbers(values[key]):
            raise ValueError(f"{key}: numbers changed")
        lines.append(rf"\expandafter\def\csname R@SL.{key}\endcsname{{{text}}}")
    (GEN / "numbers_supplement_labels.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    problems = []
    for name, (lists, options) in RULES.items():
        problems += [f"{name}: {p}" for p in apply(name, lists, options)]
    keys()
    if problems:
        raise ValueError("printed readings disagree with Table S1:\n" + "\n".join(problems))
    for tie in TIES:
        print("printed tie with the SESOI, judged on the unrounded interval:", tie)
    print(f"wrote {len(RULES)} relabelled tables and numbers_supplement_labels.tex")


if __name__ == "__main__":
    main()
