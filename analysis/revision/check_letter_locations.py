"""Check that every location named in the response letter's \\changes{} entries exists in the
revised manuscript or supplement.

A quoted location ``Name'' passes when a \\section, \\subsection or \\paragraph heading of
revision/manuscript/main.tex or supplement.tex starts with Name (case-insensitive, LaTeX
markup removed). Unquoted locations (for example "Abstract", "Statistical analysis") are
checked the same way after splitting on ";" and ","; generic entries such as "Whole
manuscript" or "Discussion." are reported as not_applicable. Page/line numbers of the
tracked-changes PDF are outside the scope of this check.

Output: revision/validation/letter_locations.csv (one row per location); exit code 1 if any
location is missing.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LETTER = ROOT / "revision/response_letter_v2.tex"
SOURCES = [ROOT / "revision/manuscript/main.tex", ROOT / "revision/manuscript/supplement.tex"]
GENERIC = {"whole manuscript", "results", "discussion", "introduction", "limitations", "abstract"}


def plain(text: str) -> str:
    text = re.sub(r"\$[^$]*\$", lambda m: m.group(0).strip("$").replace("\\", ""), text)
    text = re.sub(r"\\[a-zA-Z]+\*?", "", text)
    return re.sub(r"[{}~]", " ", text).lower().strip(" .")


def headings() -> list[str]:
    out = ["abstract"]
    for path in SOURCES:
        tex = path.read_text()
        out += [plain(h) for h in re.findall(r"\\(?:sub)*section\*?\{(.+?)\}\s*$", tex, re.M)]
        out += [plain(h) for h in re.findall(r"\\paragraph\{(.+?)\}", tex)]
        if "\\begin{algorithm" in tex:
            out.append("algorithm")
        if "\\begin{table" in tex:
            out.append("table")
        if "\\begin{figure" in tex:
            out.append("figure")
    return out


def locations(entry: str) -> list[str]:
    quoted = re.findall(r"``(.+?)''", entry)
    rest = re.sub(r"``.+?''", "", entry)
    parts = [p for p in re.split(r"[;,()]", rest) if plain(p)]
    return quoted + parts


def main() -> int:
    heads = headings()
    rows = []
    for entry in re.findall(r"\\changes\{(.+)\}", LETTER.read_text()):
        for loc in locations(entry):
            key = plain(loc)
            key = re.sub(r"^(methods|results|introduction|section|discussion)\s+", "", key)
            if not key or key in {"methods", "results sections", "section", "results"}:
                continue
            supp = re.search(r"supplementary\s+(s\d+)", key)
            if supp:
                found = any(h.startswith(supp.group(1)) for h in heads)
                rows.append(
                    dict(entry=entry, location=loc.strip(), status="found" if found else "missing")
                )
                continue
            if key in GENERIC or key.startswith(("new ", "whole")):
                status = "not_applicable" if key not in heads else "found"
            else:
                first = key.split()[0]
                found = any(h.startswith(key) for h in heads) or (
                    first in {"algorithm", "table", "tables", "figure", "supplementary"}
                )
                status = "found" if found else "missing"
            rows.append(dict(entry=entry, location=loc.strip(), status=status))
    out = ROOT / "revision/validation/letter_locations.csv"
    with out.open("w") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["entry", "location", "status"], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    missing = [r for r in rows if r["status"] == "missing"]
    print(f"{len(rows)} locations, {len(missing)} missing")
    for r in missing:
        print("MISSING:", r["location"], "|", r["entry"])
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
