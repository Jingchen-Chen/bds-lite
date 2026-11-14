# ruff: noqa: E501
"""Render revision/response.md and the PeerJ requirements checklist from their templates.

Placeholders in the template:
  {{R:key}}  a manuscript number from revision/manuscript/generated/numbers*.tex (plain text)
  {{L:key}}  line numbers of a passage in the compiled revision/manuscript/main.pdf, located by
             the anchor phrases in ANCHORS (start phrase, optional end phrase)
  {{PAGES:main}}, {{PAGES:supplement}}  page count of the compiled main.pdf or supplement.pdf
Literal numbers that the template quotes from generated tables or from the submitted version
are listed in CHECKS and must occur in the named source, otherwise the script stops.

Line numbers are those printed in the margin of main.pdf (lineno), read with pdftotext -layout.
Usage: PYTHONPATH=src:. .venv/bin/python analysis/revision/render_response.py
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "revision/manuscript/generated"
PDF = ROOT / "revision/manuscript/main.pdf"
# (template, output): the response letter and the PeerJ requirements checklist share the anchors
PAIRS = [
    (ROOT / "revision/response.template.md", ROOT / "revision/response.md"),
    (
        ROOT / "revision/PEERJ_REQUIREMENTS_CHECKLIST.template.md",
        ROOT / "revision/PEERJ_REQUIREMENTS_CHECKLIST.md",
    ),
]

# key -> (start phrase, end phrase or None); phrases as they read in the compiled text
ANCHORS: dict[str, tuple[str, str | None]] = {
    "abstract": ("Boundary errors matter whenever", "over repeated training runs."),
    "intro": ("Motivation. Segmentation is used", "studies of this kind."),
    "intro_motivation": ("Motivation. Segmentation is used", "Maier-Hein et al., 2024)."),
    "intro_compact": ("Why a compact U-Net.", "be assigned to the training signal."),
    "corrections": ("Relation to the first version.", "(Supplementary S4, S5)."),
    "res_interaction": (
        "In a post hoc test, the distillation contrast was",
        "(Supplementary S24).",
    ),
    "res_errors": ("counts missed structures and false alarms", "lower with augmentation."),
    "intro_background": ("Background. Training-time boundary signals", "Liu et al., 2024)."),
    "intro_gaps": ("Research gaps. What is known", "change one of these."),
    "intro_contrib": ("Contributions. We ask when a training-only", "studies of this kind."),
    "raj": ("Recent work on tumours follows the same direction", "memory and latency"),
    "related": ("Boundary losses. Boundary loss integrates", "each source reports)."),
    "design": (
        "The objects of comparison are training recipes",
        "including four teachers.",
    ),
    "framework": ("Contour displacement and overlap.", "what the distillation term adds."),
    "dsc_relation": ("Contour displacement and overlap.", "not compared across datasets."),
    "routes": ("Routes of a boundary signal.", "what the distillation term adds."),
    "arms": ("Every arm is a four-level", "is reported in Supplementary S7."),
    "comparators": ("Four reference methods were trained", "is reported in Supplementary S7."),
    "data": ("Three public benchmarks were used", "approval was not required."),
    "data_choice": ("Three public benchmarks were used", "a scale of difficulty."),
    "splits": ("Splits. The ISIC2018 Task", "held out for validation."),
    "preprocessing": ("Preprocessing. ACDC and Synapse volumes", "approval was not required."),
    "segloss": ("This compound loss is a common default", "shares one segmentation loss."),
    "alpha": ("bounds how much the gate can rescale", "(sensitivity: Supplementary S11)."),
    "stopgrad": ("where the stop-gradient operator", "measures qs and Fs separately."),
    "teacher": ("The teacher is a U-Net with", "capacity and pretraining."),
    "controls_method": ("Controls. Each control removes one property", "quality gate."),
    "training": ("All runs used AdamW", "as recorded with each run."),
    "metrics": ("Metrics. Predictions are the argmax", "secondary, descriptive metrics."),
    "empty": (
        "A class absent from both prediction and reference is excluded.",
        "same predictions (Tables 7 and 8).",
    ),
    "stats": ("The unit of replication is the training run;", "(Supplementary S23)."),
    "res_variation": ("Retraining one recipe with a different seed", "with three runs per arm."),
    "res_decoder": ("The decoder. The boundary decoder raised", "decoder gain is exploratory."),
    "res_innetwork": ("Feature distillation and gate. Added to D", "up to about 0.02."),
    "res_full": (
        "The complete recipe. On ISIC2018, BDS-Lite scored",
        "ten-run arm means are in Supplementary S13.",
    ),
    "res_replication": ("In the replication (", "each of the three structures."),
    "res_controls": ("The controls were run after the replication", "below the mean of D"),
    "res_aug": ("ACDC patients, D trained with D4", "(Supplementary S24)."),
    "res_published": (
        "lists published results for context",
        "averaged within patients; Supplementary S21).",
    ),
    "res_cost": (
        "gives the inference cost of the networks",
        "in training only (Supplementary S19).",
    ),
    "res_qual": ("shows three official ACDC test slices", "are in Supplementary S12."),
    "disc_answer": (
        "In the compact U-Net studied here, boundary distillation",
        "could not have separated them from chance.",
    ),
    "disc_bds": (
        "The complete recipe adds three components",
        "coefficient (−0.024, 95% CI −0.040 to −0.009; Supplementary S13).",
    ),
    "disc_teacher": ("Output distillation (Hinton", "between teacher and student."),
    "disc_mechanism": (
        "One mechanism fits these observations.",
        "depart from the reference labels.",
    ),
    "disc_aug_confound": (
        "The teacher itself was trained without augmentation",
        "between teacher and student.",
    ),
    "practice": ("These implications concern contour agreement", "a teacher of the student"),
    "practice_scope": (
        "These implications concern contour agreement",
        "did not measure a clinical outcome.",
    ),
    "lim": ("Scope. The results concern one compact", "their sizes are not compared."),
    "lim_noaug": ("The main protocol used no augmentation", "persists under augmentation."),
    "lim_scope": ("Scope. The results concern one compact", "clinical indices."),
    "lim_precision": ("Precision of the in-network results.", "establish no cause."),
    "lim_cohort": (
        "The replication and its controls. The confirmed gain",
        "their sizes are not compared.",
    ),
    "conclusions": (
        "For a compact 2D U-Net, in-network boundary distillation",
        "stronger and 3D models.",
    ),
    "future": ("Future work should compare distillation", "stronger and 3D models."),
    "ack": ("The authors used ChatGPT", None),
    "vocab": ("We classify each contrast as confirmed, detected", "(Supplementary S23)."),
    "aug_rationale": (
        "The main protocol uses no data augmentation.",
        "depend on this training regime.",
    ),
    "se_conservative": (
        "Both terms of Eq. 6 contain the residual variance",
        "(Supplementary S20).",
    ),
    "eq_dsc": ("Contour displacement and overlap.", "and do not cancel."),
    "kdfull_band": (
        "The pixels near the teacher's contours remain",
        "confining distillation to the band matters.",
    ),
}

# Literal values quoted in the template that are not manuscript keys: each must occur in its source.
V0 = "/tmp/__v0__"  # replaced by the text of the reviewed PDF
CHECKS: list[tuple[str, list[str]]] = [
    (
        "tab_s_original.tex",
        [
            "ISIC2018 & U-Net & 0.883 & 0.809 & 0.552",
            "ISIC2018 & BDS-Lite & 0.883 & 0.811 & 0.560",
            "ACDC & U-Net & 0.789 & 0.711 & 0.831",
            "ACDC & BDS-Lite & 0.785 & 0.707 & 0.827",
            "Synapse & U-Net & 0.854 & 0.836 & 0.861",
            "Synapse & BDS-Lite & 0.857 & 0.839 & 0.863",
        ],
    ),
    (
        "tab_two_sets.tex",
        [
            r"ISIC2018, original & U-Net & 0.835 $\pm$ 0.007 & 0.743 $\pm$ 0.009 & 0.362",
            r" & BDS-Lite & 0.838 $\pm$ 0.003 & 0.748 $\pm$ 0.006 & 0.392",
            r"ISIC2018, retrained & U-Net & 0.841 $\pm$ 0.003 & 0.750 $\pm$ 0.004 & 0.363",
            r" & BDS-Lite & 0.837 $\pm$ 0.005 & 0.747 $\pm$ 0.008 & 0.367",
            r"ACDC, original & U-Net & 0.764 $\pm$ 0.003 & 0.683 $\pm$ 0.003 & 0.808",
            r" & BDS-Lite & 0.764 $\pm$ 0.008 & 0.684 $\pm$ 0.008 & 0.808",
            r"ACDC, retrained & U-Net & 0.742 $\pm$ 0.008 & 0.660 $\pm$ 0.006 & 0.783",
            r" & BDS-Lite & 0.758 $\pm$ 0.013 & 0.677 $\pm$ 0.014 & 0.801",
            r"Synapse, original & U-Net & 0.516 $\pm$ 0.034 & 0.466 $\pm$ 0.030 & 0.537",
            r" & BDS-Lite & 0.517 $\pm$ 0.019 & 0.468 $\pm$ 0.017 & 0.538",
            r"Synapse, retrained & U-Net & 0.487 $\pm$ 0.028 & 0.439 $\pm$ 0.028 & 0.506",
            r" & BDS-Lite & 0.496 $\pm$ 0.012 & 0.446 $\pm$ 0.011 & 0.516",
            r"+0.031$^{*}$ [+0.026, +0.036]",
        ],
    ),
    (
        "tab_s_membership.tex",
        [
            "In these models' training set & 358",
            "In their checkpoint-selection set & 84",
            "Held out from them & 77",
        ],
    ),
    ("tab_resource_text.tex", ["U-Net & 1,927,042 & 15.970 & 1.66", "& 1,928,114 & 16.079 & 1.79"]),
    (
        V0,
        [
            "U-Net 1,927,042 1,927,042 15.970 1.93",
            "BDS-Lite 2,198,003 1,928,114 16.079 2.40",
        ],
    ),
    # R3.12: pooled additive-model coefficients of feature distillation and gate (S13);
    # main text, Discussion (M-14): ISIC2018 surface-loss coefficient, typed as a literal
    (
        "tab_components.tex",
        [
            r"Boundary F1 & Surface loss & $-$0.024 [$-$0.040, $-$0.009] &",
            r"Boundary F1 & Distillation & $-$0.007 [$-$0.027, +0.013] & +0.004 [$-$0.008, +0.017] & +0.000 [$-$0.021, +0.022] & +0.001 [$-$0.008, +0.010]",
            r"Boundary F1 & Gate & $-$0.006 [$-$0.026, +0.015] & +0.003 [$-$0.010, +0.015] & +0.013 [$-$0.012, +0.038] & +0.002 [$-$0.007, +0.012]",
        ],
    ),
    # R3.14: reference structures per area bin (Table S19 headers; S24 pair counts) and missed
    # rates below 1% of the slice (S8)
    (
        "tab_s_collapse_wide.tex",
        [
            r"<1\% ($n=370$)} & \multicolumn{2}{c}{1-2\% ($n=386$)}",
            r"<1\% ($n=1887$)} & \multicolumn{2}{c}{1-2\% ($n=675$)}",
        ],
    ),
    (
        "numbers_review.tex",
        [
            r"R@RV.err.acdc.matched.present\endcsname{1,013}",
            r"R@RV.err.synapse.matched.present\endcsname{3,370}",
        ],
    ),
    (
        "tab_s_collapse_wide.tex",
        [
            "U-Net & 0.138 & 0.636",
            "U-Net + aux. boundary only & 0.088 & 0.672",
            "U-Net & 0.171 & 0.593",
            "U-Net + aux. boundary only & 0.168 & 0.598",
        ],
    ),
]


def plain(value: str) -> str:
    value = value.replace("$-$", "−").replace(r"\,", " ").replace(r"\%", "%")
    m = re.fullmatch(r"\$([0-9.]+)\\times10\^\{(-?\d+)\}\$", value)
    if m:
        return f"{m.group(1)}e{m.group(2)}"
    if re.search(r"[\\$]", value):
        raise ValueError(f"cannot render {value!r} as plain text")
    return value


def keys() -> dict[str, str]:
    out = {}
    for path in sorted(GEN.glob("numbers*.tex")):
        for line in path.read_text().splitlines():
            m = re.match(r"\\expandafter\\def\\csname R@(.+?)\\endcsname\{(.*)\}$", line)
            if m:
                out[m.group(1)] = m.group(2)
    return out


def squeeze(text: str) -> str:
    for a, b in (("ﬁ", "fi"), ("ﬂ", "fl"), ("ﬀ", "ff"), ("’", "'"), ("“", '"'), ("”", '"')):
        text = text.replace(a, b)
    return re.sub(r"\s+", " ", text)


def numbered_text() -> tuple[str, list[int]]:
    """Body text of main.pdf with, for every character, the margin line number it is on."""
    raw = subprocess.run(
        ["pdftotext", "-layout", str(PDF), "-"], check=True, capture_output=True, text=True
    ).stdout
    text, owner, last = "", [], 0
    for line in raw.splitlines():
        m = re.match(r"^\s{0,3}(\d+)(?:\s(.*))?$", line)
        if not m or int(m.group(1)) != last + 1:
            continue  # floats, captions, footers and display equations carry no line number
        last = int(m.group(1))
        piece = squeeze(m.group(2) or "").strip()
        if not piece:
            continue
        if text.endswith("-") and piece[:1].islower():
            text, owner = text[:-1], owner[:-1]  # undo end-of-line hyphenation
        elif text:
            text, owner = text + " ", owner + [last]
        text += piece
        owner += [last] * len(piece)
    return text, owner


def locate(text: str, owner: list[int], key: str) -> str:
    start, end = ANCHORS[key]
    variants = lambda s: {squeeze(s), squeeze(s).replace("-", "")}  # noqa: E731
    i = min((text.find(v) for v in variants(start) if v in text), default=-1)
    if i < 0:
        raise ValueError(f"anchor start not found for {key}: {start!r}")
    if end is None:
        return f"line {owner[i]}"
    hits = [(text.find(v, i), v) for v in variants(end) if text.find(v, i) >= 0]
    if not hits:
        raise ValueError(f"anchor end not found for {key}: {end!r}")
    j, v = min(hits)
    a, b = owner[i], owner[j + len(v) - 1]
    return f"line {a}" if a == b else f"lines {a}–{b}"


def pages(name: str) -> int:
    """Page count of the compiled main.pdf or supplement.pdf ({{PAGES:main}}, {{PAGES:supplement}})."""
    info = subprocess.run(
        ["pdfinfo", str(PDF.with_name(f"{name}.pdf"))], check=True, capture_output=True, text=True
    ).stdout
    return int(re.search(r"^Pages:\s+(\d+)", info, re.M).group(1))


def main() -> None:
    v0 = subprocess.run(
        ["pdftotext", "-layout", str(ROOT / "revision/peerj-reviewing-142937-v0.pdf"), "-"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    v0 = re.sub(r"\s+", " ", v0.replace("ﬁ", "fi").replace("ﬂ", "fl"))
    for source, needles in CHECKS:
        hay = v0 if source == V0 else (GEN / source).read_text()
        for needle in needles:
            if needle not in hay:
                raise ValueError(f"quoted value not found in {source}: {needle!r}")
    numbers = keys()
    text, owner = numbered_text()
    for template_path, out_path in PAIRS:
        template = template_path.read_text()
        missing = [k for k in re.findall(r"\{\{R:([^}]+)\}\}", template) if k not in numbers]
        if missing:
            raise ValueError(f"unknown number keys: {sorted(set(missing))}")
        out = re.sub(r"\{\{R:([^}]+)\}\}", lambda m: plain(numbers[m.group(1)]), template)
        out = re.sub(r"\{\{L:([^}]+)\}\}", lambda m: locate(text, owner, m.group(1)), out)
        out = re.sub(r"\{\{PAGES:(main|supplement)\}\}", lambda m: str(pages(m.group(1))), out)
        if "{{" in out:
            raise ValueError(f"unresolved placeholder in {template_path.name}")
        header = (
            f"<!-- Generated from {template_path.name} by analysis/revision/render_response.py; "
            "do not edit. Line numbers refer to revision/manuscript/main.pdf. -->\n"
        )
        out_path.write_text(header + out)
        print("wrote", out_path.relative_to(ROOT), f"({len(out.split())} words)")


if __name__ == "__main__":
    main()
