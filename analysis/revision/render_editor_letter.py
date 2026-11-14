# ruff: noqa: E501
"""Render revision/letter_to_editor.md from its template with generated number keys.

Every {{KEY}} in revision/letter_to_editor.template.md is replaced by the value of \\R{KEY} in
revision/manuscript/generated/numbers*.tex, converted from LaTeX to plain text. Fails on a missing key.
Usage: python analysis/revision/render_editor_letter.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "revision/manuscript/generated"


def keys() -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(GEN.glob("numbers*.tex")):
        for key, value in re.findall(
            r"\\csname R@(.+?)\\endcsname\{(.*)\}$", path.read_text(), re.M
        ):
            out[key] = value
    return out


def plain(value: str) -> str:
    value = value.replace("$-$", "−").replace("\\,", " ")
    m = re.fullmatch(r"\$(\d\.\d)\\times10\^\{(-?\d+)\}\$", value)
    if m:
        value = f"{m.group(1)} × 10^{m.group(2)}"
    if "\\" in value or "$" in value:
        raise ValueError(f"cannot render {value!r} as plain text")
    return value


def main() -> int:
    template = (ROOT / "revision/letter_to_editor.template.md").read_text()
    template = re.sub(r"\A<!--.*?-->\n", "", template, flags=re.S)
    values = keys()
    missing = sorted({k for k in re.findall(r"\{\{(.+?)\}\}", template) if k not in values})
    if missing:
        print("missing keys:", ", ".join(missing))
        return 1
    body = re.sub(r"\{\{(.+?)\}\}", lambda m: plain(values[m.group(1)]), template)
    header = "<!-- Generated from letter_to_editor.template.md by analysis/revision/render_editor_letter.py; do not edit. -->\n"
    (ROOT / "revision/letter_to_editor.md").write_text(header + body)
    print("written revision/letter_to_editor.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
