"""Format the isolated ACDC gate addendum without changing frozen numeric keys."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    source = ROOT / "results/gate_identity_unweighted_acdc/gate_identity_effects.csv"
    with source.open() as stream:
        rows = [r for r in csv.DictReader(stream) if r["metric"] == "boundary_f1"]
    if len(rows) != 1 or rows[0]["n"] != "20":
        raise ValueError("One ACDC Boundary F1 result on 20 patients is required")
    row = rows[0]

    def number(key: str) -> str:
        value = float(row[key])
        sign = "$-$" if value < 0 else "+"
        return sign + f"{abs(value):.6f}"

    values = {
        "GIU.acdc.bf1": number("mean_delta_identity_minus_full"),
        "GIU.acdc.bf1.ci": f"[{number('ci95_low')}, {number('ci95_high')}]",
    }
    lines = ["% isolated post hoc ACDC gate addendum; frozen GID keys are unchanged"]
    lines += [
        rf"\expandafter\def\csname R@{key}\endcsname{{{value}}}" for key, value in values.items()
    ]
    (ROOT / "paper/generated/numbers_gate_addendum.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
