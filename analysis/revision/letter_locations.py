# ruff: noqa: E501
"""Page and line locations in the compiled revised manuscript, for the response letter.

Each anchor is a phrase that starts (and optionally a phrase that ends) a passage of the main text.
The script reads revision/manuscript/main.pdf with pdftotext -layout, keeps the numbered body lines
(the PeerJ template numbers every body line continuously), and finds each phrase after removing
whitespace and hyphens, so that line breaks and hyphenation do not matter. Floats (tables and
figures) are located by their caption label and reported by page. Supplement sections are located
in supplement.pdf by heading. It fails if any anchor is missing or occurs more than once.

Writes revision/manuscript/generated/letter_locations.tex (macros \\L@<key>) and
letter_locations.csv (key, page, first line, last line, phrase).

Usage: python analysis/revision/letter_locations.py
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MS = ROOT / "revision/manuscript"
OUT = MS / "generated"

# key: (start phrase, end phrase or None)
ANCHORS = {
    "abstract": (
        "Retraining each arm several times on ISIC2018, ACDC and Synapse",
        "against controls that remove the property credited for a gain.",
    ),
    "abstract_answer": (
        "Retraining each arm several times on ISIC2018, ACDC and Synapse",
        "of the same size as the variation between retrainings.",
    ),
    "intro_contour": (
        "Segmentation is used to measure the outline",
        "recommends reporting them with overlap",
    ),
    "intro_compact": (
        "Segmentation networks have grown through attention gates",
        "isolates the effect of the training signal.",
    ),
    "intro_gap": (
        "Three gaps limit what is known about boundary distillation.",
        "can locate a gain.",
    ),
    "intro_approach": (
        "We take the network trained with a training-only boundary decoder (D) as the reference",
        "repeated under augmentation.",
    ),
    "intro_contrib": (
        "Our contributions answer three questions:",
        "band distillation did not raise Boundary F1 over the reference.",
    ),
    "related": (
        "Boundary losses. Boundary loss integrates",
        "with their contributions and limitations.",
    ),
    "design": (
        "We compared training recipes, not single trained models",
        "of the project: Supplementary S15).",
    ),
    "arms": ("Every arm is a four-level 2D U-Net", "not used as the reference for distillation."),
    "why_d": (
        "The decoder-only arm D is the reference for every distillation design",
        "measures what the distillation term adds.",
    ),
    "datasets": ("provides dermoscopic images with one lesion class", "approval was not required."),
    "liver_brain": (
        "We chose three public tasks with contrasting structure",
        "rather than on further datasets.",
    ),
    "isic_withdraw": (
        "and is evaluated on the",
        "share no image identifier with the development images",
    ),
    "focal": (
        "Losses that reweight small or hard structures",
        "every arm shares one segmentation loss.",
    ),
    "boundary_target": (
        "is the union over foreground classes of the exclusive-or of binary dilation",
        "is removed after training.",
    ),
    "alpha": (
        "bounds how much the gate can rescale a feature",
        "(sensitivity: Supplementary S11).",
    ),
    "stopgrad": ("where the stop-gradient operator", "which is trained only by"),
    "sdf": ("Surface loss (S).", "drops terms or sets"),
    "routes": ("A boundary signal can reach the deployed output by three routes", "separately."),
    "kd_method": (
        "keeps D and its losses and adds a loss from a separate teacher",
        "were not tuned (Supplementary S17).",
    ),
    "selection": ("Checkpoints were selected by the first maximum", "within patients or cases."),
    "metrics": (
        "Predictions are the argmax over classes",
        "the other metrics are secondary and descriptive.",
    ),
    "small_targets": (
        "A missed structure or false alarm scores approximately zero",
        "assigns the image diagonal to these pairs.",
    ),
    "stats": ("The unit of replication is the training run;", "(Supplementary S14)."),
    "sesoi": (
        "The smallest effect size of interest (SESOI) is 0.01",
        "has a single primary contrast.",
    ),
    "res_variation": (
        "The between-run standard deviation of Boundary F1 was",
        "for every distillation contrast below",
    ),
    "res_kd_explo": (
        "ACDC was chosen for this replication because of an exploratory study",
        "none significant after Holm correction",
    ),
    "lim_se": ("The standard error counts a run-by-unit residual", "S20, S22)."),
    "res_innetwork": (
        "Adding in-network transfer (feature distillation and gate) to D did not improve",
        "did not change detectably either",
    ),
    "res_full": ("On ISIC2018, the complete BDS-Lite recipe scored", "did not differ detectably"),
    "res_gsl": ("U-Net+GSL also scored below D on ISIC2018", "with it."),
    "res_replication": (
        "In the replication on ACDC without augmentation",
        "overlapped by",
    ),
    "res_controls": (
        "analysed after the replication and reusing its runs",
        "relative to D (Supplementary S21).",
    ),
    "res_aug": (
        "In an exploratory comparison on the same 50 ACDC patients",
        "for the unaugmented D runs",
    ),
    "cost": (
        "External-teacher distillation adds cost only in training",
        "are reported in Supplementary S19.",
    ),
    "disc_answer": (
        "The one gain that replicated on held-out patients",
        "remove the property credited for the gain.",
    ),
    "limitations": (
        "The conclusions concern one compact 2D U-Net",
        "lesion-level independence is not established",
    ),
    "lim_controls": (
        "The controls reuse the replication's reference runs",
        "do not separate these factors.",
    ),
    "lim_interaction": (
        "No interaction between augmentation and distillation was tested",
        "different numbers of runs.",
    ),
    "nnunet": (
        "We did not train nnU-Net, transformer-based or natively 3D models",
        "studied here.",
    ),
    "slices": ("Slices were processed and scored independently", "was not modelled."),
    "tumour": ("Tumour segmentation was not studied.", None),
    "units": ("Distances are in pixels of the", "were not computed."),
    "conclusions": (
        "In-network feature distillation with a gate was not shown to add to the network's own",
        "remove the property credited for a gain.",
    ),
    "data_avail": (
        "The per-case results, analysis scripts, protocols and analysis-plan records that support every reported contrast",
        "summary statistics only.",
    ),
}
SUPP = [f"S{i}" for i in range(1, 24)]
LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "’": "'", "‘": "'"}


def norm(text: str) -> str:
    for a, b in LIGATURES.items():
        text = text.replace(a, b)
    return re.sub(r"[\s\-−–—­]", "", text).lower()


def pages(pdf: Path) -> list[str]:
    n = int(
        re.search(
            r"Pages:\s+(\d+)",
            subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout,
        ).group(1)
    )
    return [
        subprocess.run(
            ["pdftotext", "-layout", "-f", str(i), "-l", str(i), str(pdf), "-"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for i in range(1, n + 1)
    ]


def body_index(texts: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Concatenated normalized body text and, per character, (page, line)."""
    chars, where = [], []
    for page, text in enumerate(texts, start=1):
        for raw in text.splitlines():
            m = re.match(r"^\s{0,8}(\d{1,4})\s+(\S.*)$", raw)
            if not m:
                continue
            piece = norm(m.group(2))
            chars.append(piece)
            where.extend([(page, int(m.group(1)))] * len(piece))
    return "".join(chars), where


def locate(body: str, where: list, phrase: str, start: int = 0) -> tuple[int, int]:
    key = norm(phrase)
    first = body.find(key, start)
    if first < 0:
        raise ValueError(f"anchor not found: {phrase!r}")
    if start == 0 and body.find(key, first + 1) >= 0:
        raise ValueError(f"anchor not unique: {phrase!r}")
    return first, first + len(key) - 1


def main() -> int:
    main_pages = pages(MS / "main.pdf")
    body, where = body_index(main_pages)
    rows, macros = [], []
    for key, (start, end) in ANCHORS.items():
        s0, s1 = locate(body, where, start)
        e1 = locate(body, where, end, s1)[1] if end else s1
        (p0, l0), (p1, l1) = where[s0], where[e1]
        if l1 - l0 > 60:
            raise ValueError(f"{key}: end anchor too far from start ({l0}-{l1})")
        text = (
            f"p.~{p0}, l.~{l0}"
            if l0 == l1
            else (f"p.~{p0}, ll.~{l0}--{l1}" if p0 == p1 else f"pp.~{p0}--{p1}, ll.~{l0}--{l1}")
        )
        rows.append((key, p0, l0, l1, start))
        macros.append(f"\\expandafter\\def\\csname L@{key}\\endcsname{{{text}}}")
        macros.append(f"\\expandafter\\def\\csname L@{key}@start\\endcsname{{p.~{p0}, l.~{l0}}}")
        macros.append(f"\\expandafter\\def\\csname L@{key}@end\\endcsname{{p.~{p1}, l.~{l1}}}")
    supp_pages = pages(MS / "supplement.pdf")
    seen_labels: set[str] = set()
    # Float names and pages follow the sources and PDFs, so a move to the supplement
    # cannot silently leave the response letter pointing to an obsolete number.
    for source, texts, prefix in (
        (MS / "main.tex", main_pages, ""),
        (MS / "supplement.tex", supp_pages, "S"),
    ):
        # Read actual LaTeX numbering: longtables in generated inputs also advance
        # the supplementary table counter, even when they have no caption.
        aux = source.with_suffix(".aux").read_text()
        for match in re.finditer(
            r"\\begin\{(table|figure)\}(?:\[[^]]*\])?.*?\\end\{\1\}",
            source.read_text(),
            re.S,
        ):
            kind = match.group(1)
            label_match = re.search(r"\\label\{([^}]+)\}", match.group())
            if label_match is None:
                continue
            key = label_match.group(1)
            if key in seen_labels:
                raise ValueError(f"label defined in both documents: {key}")
            seen_labels.add(key)
            # Number and page both come from the compiled label: \newlabel{key}{{number}{page}}.
            # The label sits in the caption, so its page is the page the float was shipped out on.
            compiled = re.search(
                r"\\newlabel\{" + re.escape(key) + r"\}\{\{([^}]+)\}\{([^}]+)\}", aux
            )
            if compiled is None:
                raise ValueError(f"compiled label missing: {key}")
            name = f"{kind.title()} {compiled.group(1)}"
            page = int(compiled.group(2))
            # Sanity check only: the PDF page must carry the float name (caption, or in-text
            # mention). Captions are not matched by text position, so a line break before a
            # cross-reference such as "Table~\ref{...}." cannot be mistaken for a caption.
            if not 1 <= page <= len(texts) or name not in texts[page - 1]:
                raise ValueError(f"{name}: page {page} from .aux does not show it in the PDF")
            rows.append((key, page, "", "", name))
            macros.append(f"\\expandafter\\def\\csname L@{key}\\endcsname{{p.~{page}}}")
            display = ("Supplementary " if prefix else "") + name.replace(" ", "~")
            macros.append(f"\\expandafter\\def\\csname F@{key}\\endcsname{{{display}}}")
    for s in SUPP:
        hits = [
            i
            for i, t in enumerate(supp_pages, start=1)
            if re.search(r"^\s*" + re.escape(s) + r"\.\s", t, re.M)
        ]
        if not hits:
            raise ValueError(f"supplement section {s} not found")
        rows.append((s, hits[0], "", "", f"Supplementary {s}"))
        macros.append(
            f"\\expandafter\\def\\csname L@{s}\\endcsname{{Supplementary {s}, p.~{hits[0]}}}"
        )
    (OUT / "letter_locations.tex").write_text(
        "% generated by analysis/revision/letter_locations.py from main.pdf and supplement.pdf; do not edit\n"
        + "\n".join(macros)
        + "\n"
    )
    with (OUT / "letter_locations.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["key", "page", "first_line", "last_line", "anchor"])
        writer.writerows(rows)
    print(f"{len(rows)} locations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
