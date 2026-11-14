"""Build the tracked-changes manuscript (original submission -> revised main.tex) with latexdiff.

Original: recovered/peerj-submission/main.tex (the submitted PeerJ source, with Figure_1/2.png).
Revised:  revision/manuscript/main.tex (generated tables and \\R{} numbers flattened by latexdiff).

latexdiff is not vendored. Download the pinned release and pass the path of latexdiff-so:
  https://github.com/ftilmann/latexdiff/releases/download/1.3.4/latexdiff-1.3.4.tar.gz
  sha256(tar.gz) aed1c39d51e5c7a8894a5e4b7190106e93968dd90edcc0dde803fcbffe01b2b4

Sectioning commands and captions are wrapped in one-argument macros before diffing so that
latexdiff treats each as one unit (otherwise deleted paragraphs can end up inside a heading or
caption argument and the diff does not compile). Markup: added text blue wavy underline,
deleted text red strike-out. Compiler output is retained even if compilation fails.

Usage: python analysis/revision/make_tracked_changes.py /path/to/latexdiff-so
Output: revision/manuscript/tracked_changes.pdf (built in a temporary directory with tectonic).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / "recovered/peerj-submission"
REVISED = ROOT / "revision/manuscript"
TECTONIC = Path.home() / ".local/bin/tectonic"
HEADINGS = (
    (r"\\section\*\{", r"\\HSec{", r"\section*"),
    (r"\\section\{", r"\\HSecN{", r"\section"),
    (r"\\subsection\*\{", r"\\HSub{", r"\subsection*"),
    (r"\\subsection\{", r"\\HSubN{", r"\subsection"),
    (r"\\paragraph\{", r"\\HPar{", r"\paragraph"),
    (r"\\caption\{", r"\\HCaption{", r"\caption"),
)


def wrap_headings(text: str) -> str:
    # Unknown author-marker commands are boxed by latexdiff/ulem, making long
    # decision notes spill off the page. Preserve their text as ordinary markup.
    text = re.sub(r"\\authordecision\{([^{}]*)\}", r"[AUTHOR DECISION: \1]", text)
    # Deleted floats/labels may be commented out by latexdiff. Keep references
    # faithful to each source version instead of rendering them as question marks.
    counts = {"table": 0, "figure": 0, "equation": 0}
    for match in re.finditer(
        r"\\begin\{(table|figure|equation)\}(?:\[[^]]*\])?.*?\\end\{\1\}",
        text,
        re.S,
    ):
        kind = match.group(1)
        counts[kind] += 1
        for label in re.findall(r"\\label\{([^}]+)\}", match.group()):
            text = text.replace(r"\ref{" + label + "}", str(counts[kind]))
            text = text.replace(r"\eqref{" + label + "}", f"({counts[kind]})")
    macros = []
    for pattern, replacement, command in HEADINGS:
        text = re.sub(pattern, replacement, text)
        name = replacement.lstrip("\\").rstrip("{")
        macros.append(f"\\newcommand{{\\{name}}}[1]{{{command}{{#1}}}}")
    return text.replace("\\begin{document}", "\n".join(macros) + "\n\\begin{document}", 1)


def main(latexdiff: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "manuscript"
        shutil.copytree(
            REVISED,
            work,
            ignore=shutil.ignore_patterns(
                "main.pdf", "supplement.pdf", "tracked_changes.pdf", "*.log"
            ),
        )
        shutil.copytree(ROOT / "revision/figures", Path(tmp) / "figures")
        # latexdiff misparses the extra width/spacing arguments of tabular*.
        # Keep the same font and columns at natural width in the marked copy;
        # the clean manuscript retains its full-width flexible table layout.
        for table in (work / "generated").glob("*.tex"):
            text = table.read_text()
            flexible = r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}"
            if flexible in text:
                table.write_text(
                    text.replace(flexible, r"\begin{tabular}{").replace(
                        r"\end{tabular*}", r"\end{tabular}"
                    )
                )
        for figure in ORIGINAL.glob("Figure_*.png"):
            shutil.copy(figure, work)
        (work / "p_old.tex").write_text(wrap_headings((ORIGINAL / "main.tex").read_text()))
        revised = (REVISED / "main.tex").read_text()
        # A recognized graphics box keeps deleted prose out of this new table's
        # cells. Using its natural width preserves the normal 10 pt font.
        for table in ("tab_central_main.tex", "tab_resource_main.tex"):
            revised = revised.replace(
                r"{\normalsize\input{generated/" + table + "}}",
                r"\resizebox{\width}{!}{\input{generated/" + table + "}}",
            )
        (work / "p_new.tex").write_text(wrap_headings(revised))
        diff = subprocess.run(
            [
                "perl",
                latexdiff,
                "--flatten",
                "--type=UNDERLINE",
                "--math-markup=coarse",
                "--disable-citation-markup",
                "--append-textcmd=authordecision",
                "p_old.tex",
                "p_new.tex",
            ],
            cwd=work,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        (work / "tracked_changes.tex").write_text(diff)
        compiled = subprocess.run(
            [str(TECTONIC), "-X", "compile", "tracked_changes.tex"],
            cwd=work,
            capture_output=True,
        )
        log = ROOT / "revision/validation/tracked_changes_build.log"
        log.write_bytes(compiled.stdout + compiled.stderr)
        if compiled.returncode:
            sys.stderr.buffer.write(compiled.stderr)
            compiled.check_returncode()
        shutil.copy(work / "tracked_changes.pdf", REVISED / "tracked_changes.pdf")
    print(REVISED / "tracked_changes.pdf")


if __name__ == "__main__":
    main(sys.argv[1])
