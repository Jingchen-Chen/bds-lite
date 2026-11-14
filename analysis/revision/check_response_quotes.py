"""Check the response letter's comment entries against the reviewer comments.

Default: reads revision/response.md, the current letter. It summarises each comment instead of
quoting it, so there is no full quotation to compare. It checks that
  - the IDs are exactly E.1-E.6, R1.1-R1.6, R2.1-R2.7, R3.1-R3.20, in this order, each a "### ID"
    heading with exactly one "**Comment (summary...).**" line;
  - every "reviewer's item ..." label names a numbered item of Reviewer 3 that exists in
    revision/reviewer.md under that heading, each item 1-15 once and in order;
  - every quotation ("..." or curly quotes) and every DOI or URL inside a Comment line occurs
    verbatim in revision/reviewer.md (after removing Markdown bold markers and normalising
    typographic quotes and dashes).
With --legacy-tex: the original check of the superseded revision/response_to_reviewers.tex, which
quotes every comment in full (verbatim, once, in order, covering the review text).
Exits 1 on any failure.

    .venv/bin/python analysis/revision/check_response_quotes.py [--legacy-tex]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = (
    [f"E.{i}" for i in range(1, 7)]
    + [f"R1.{i}" for i in range(1, 7)]
    + [f"R2.{i}" for i in range(1, 8)]
    + [f"R3.{i}" for i in range(1, 21)]
)
HEADINGS = {
    "Basic reporting",
    "Experimental design",
    "Validity of the findings",
    "Additional comments",
    "See Above",
    "Improvements:",
    "Suggestions Abstract:",
    "Introduction",
    "Materials and METHODOLOGY",
    "Results and Discussion",
    "Title: When does boundary distillation help lightweight medical image segmentation?",
}


def norm(text: str) -> str:
    for a, b in (("**", ""), ("“", '"'), ("”", '"'), ("``", '"'), ("''", '"'), ("’", "'")):
        text = text.replace(a, b)
    text = text.replace("–", "--")
    return re.sub(r"\s+", " ", text).strip()


def unescape(text: str) -> str:
    for a, b in ((r"\%", "%"), (r"\#", "#"), (r"\&", "&"), (r"\_", "_"), (r"\$", "$")):
        text = text.replace(a, b)
    return text


def braced(text: str, start: int) -> tuple[str, int]:
    depth, i = 0, start
    while True:
        if text[i] == "{" and text[i - 1] != "\\":
            depth += 1
        elif text[i] == "}" and text[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i
        i += 1


def check_legacy_tex() -> int:
    letter = (ROOT / "revision/response_to_reviewers.tex").read_text()
    source = norm((ROOT / "revision/reviewer.md").read_text())
    errors = []
    ids, quotes = [], []
    for m in re.finditer(r"\\begin\{point\}\{([^}]+)\}\s*\\comment", letter):
        ids.append(m.group(1))
        body, _ = braced(letter, m.end())
        quotes.append(norm(unescape(body)))
    if ids != EXPECTED:
        errors.append(f"IDs differ from the expected sequence: {ids}")
    pos, rest = 0, source
    for rid, quote in zip(ids, quotes, strict=True):
        at = source.find(quote, pos)
        if at < 0:
            errors.append(f"{rid}: quotation not found verbatim (in order) in reviewer.md")
            continue
        if source.count(quote) != 1:
            errors.append(f"{rid}: quotation is not unique in reviewer.md")
        pos = at + len(quote)
        rest = rest.replace(quote, " ", 1)
    start = rest.find("Editor comments")
    leftover = [
        cell.strip()
        for line in rest[start:].split("|")
        for cell in [line]
        if cell.strip()
        and not cell.strip().startswith(("Editor comments", "Reviewer ", "-"))
        and not set(cell.strip()) <= {"-", " "}
    ]
    words = " ".join(leftover)
    for heading in sorted(HEADINGS, key=len, reverse=True):
        words = words.replace(heading, " ")
    if words.strip():
        errors.append(f"review text not quoted: {words.strip()[:300]!r}")
    for error in errors:
        print("FAIL", error)
    print(f"{len(ids)} points, {len(errors)} problems")
    return 1 if errors else 0


R3_SECTIONS = ("Abstract", "Introduction", "Materials and METHODOLOGY", "Results and Discussion")
QUOTE = re.compile(r'"([^"]+)"|“([^”]+)”')
LINK = re.compile(
    r"(?:doi:\s*|https?://(?:dx\.)?doi\.org/)(10\.\d{4,}/[^\s)]+?)[.,;]?(?=[\s)]|$)|(https?://[^\s)]+)"
)
LABEL = re.compile(r"reviewer's item (?:(Abstract|Introduction) )?(\d+)")


def r3_items(source: str) -> list[tuple[str, int]]:
    """Numbered items of Reviewer 3 in source order, as (heading, number)."""
    start = source.index("Suggestions Abstract:")
    end = source.index("Over all", start)
    block = source[start:end]
    bounds = [(block.index(h), h) for h in R3_SECTIONS]
    items = []
    for m in re.finditer(r"(?<![\d./])(\d{1,2})\. (?=[A-Z])", block):
        heading = [h for at, h in bounds if at <= m.start()][-1]
        if (heading, int(m.group(1))) not in items:
            items.append((heading, int(m.group(1))))
    return items


def main() -> int:
    letter = (ROOT / "revision/response.md").read_text()
    source = norm((ROOT / "revision/reviewer.md").read_text())
    errors = []
    heads = list(re.finditer(r"^### ((?:E|R[123])\.\d+)\s*$", letter, flags=re.M))
    ids = [m.group(1) for m in heads]
    if ids != EXPECTED:
        errors.append(f"IDs differ from the expected sequence: {ids}")
    items = r3_items(source)
    labels, n_quotes, n_links = [], 0, 0
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(letter)
        comments = re.findall(
            r"^\*\*Comment \((summary[^)]*)\)\.\*\* *(.*)$", letter[m.end() : end], flags=re.M
        )
        if len(comments) != 1:
            errors.append(f"{m.group(1)}: {len(comments)} Comment lines, expected 1")
            continue
        tag, text = comments[0]
        for lab in LABEL.finditer(tag):
            heading, num = lab.group(1), int(lab.group(2))
            found = [it for it in items if it[1] == num and heading in (None, it[0])]
            if not m.group(1).startswith("R3."):
                errors.append(f"{m.group(1)}: item label outside Reviewer 3")
            if len(found) != 1:
                errors.append(
                    f"{m.group(1)}: item {heading or ''} {num} matches {len(found)} items"
                )
            else:
                labels.append(found[0])
        for q in QUOTE.finditer(text):
            n_quotes += 1
            quote = norm(q.group(1) or q.group(2))
            if quote not in source:
                errors.append(
                    f"{m.group(1)}: quotation not verbatim in reviewer.md: {quote[:80]!r}"
                )
        for lk in LINK.finditer(text):
            n_links += 1
            ref = lk.group(1) or lk.group(2)
            if ref not in source:
                errors.append(f"{m.group(1)}: {ref!r} not in reviewer.md")
    if labels != items:
        errors.append(f"item labels {labels} differ from Reviewer 3's items {items}")
    for error in errors:
        print("FAIL", error)
    print(
        f"{len(ids)} points, {len(labels)} item labels, {n_quotes} quotations, {n_links} DOI/URL; "
        f"{len(errors)} problems (response.md)"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(check_legacy_tex() if "--legacy-tex" in sys.argv[1:] else main())
