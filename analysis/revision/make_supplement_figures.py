# ruff: noqa: E501
"""Supplement figures with the arm names of the main text (2026-11-15).

The plotting functions live in generators that are listed in the SHA-256 manifests of the evidence
archives, so they are not edited. This script imports them, replaces only their label
dictionaries (and the legend labels of Figure S10), and calls the unchanged plotting functions:

  Figure S2  run_level.pdf           make_rewrite_assets.figure
  Figure S3  error_geometry.pdf      figure_licensed_replacements.error_geometry
  Figure S5  contours_isic2018_retrained.pdf  figure_qualitative_retrained.main
  Figure S7  structure_errors.pdf    figure_licensed_replacements.structures
  Figure S9  ten_seed.pdf            make_expansion_assets.figure
  Figure S10 transfer_mechanism.pdf  make_audit_assets.figure

make_rewrite_assets.run_level also writes results/tables/families/run_level_noise.csv, a frozen
file; here that write goes to a scratch directory (argument --scratch), and the script stops unless
the scratch copy equals the frozen file, which also shows that the plotted values are unchanged.

    PYTHONPATH=src:. .venv/bin/python analysis/revision/make_supplement_figures.py --scratch DIR
"""

from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.text import Text  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis/revision"))
sys.path.insert(0, str(ROOT))

NAMES = {
    "M1": "U+S",
    "M2": "D",
    "M3": "D+F+G",
    "M4": "D+G+S",
    "M5": "D+F+S",
    "GSL": "U-Net+GSL",
    "C1": "U-Net+Boundary Loss",
}
LEGEND_S10 = {"Decoder, not distilled": "D, D+G+S (undistilled)"}
# Labels that the frozen run-level figure sets inside its function (plain U-Net -> U)
TEXT_S2 = {
    "U-Net (original)": "U (original)",
    "U-Net (retrained)": "U (retrained)",
    "Retrained U-Net": "Retrained U",
    "Retrained U-Net mean ± run-level margin": "Retrained U mean ± run-level margin",
}


@contextmanager
def relabel_text(mapping: dict[str, str]):
    """Replace exact text strings (tick labels, legend entries) set while the block runs."""
    original = Text.set_text

    def set_text(self, s):
        return original(self, mapping.get(s, s) if isinstance(s, str) else s)

    Text.set_text = set_text
    try:
        yield
    finally:
        Text.set_text = original


def run_level(scratch: Path) -> None:
    import make_rewrite_assets as m

    frozen = m.FAM / "run_level_noise.csv"
    m.FAM = scratch
    runs = m.run_level()
    if not filecmp.cmp(scratch / "run_level_noise.csv", frozen, shallow=False):
        raise ValueError("run-level values differ from the frozen run_level_noise.csv")
    m.LABEL = {**m.LABEL, **NAMES}
    with relabel_text(TEXT_S2):
        m.figure(runs)


def licensed() -> None:
    import figure_licensed_replacements as m

    m.ARMS = {"M0": "U", "M2": "D", "M6": "BDS-Lite", "GSL": "U-Net+GSL"}
    m.error_geometry()
    m.structures()


def qualitative() -> None:
    import figure_qualitative_retrained as m

    m.TITLES = {"M0": "U", "M2": "D", "M6": "BDS-Lite", "GSL": "U-Net+GSL"}
    m.main()


def ten_seed() -> None:
    import make_expansion_assets as m

    m.ARM = {"unet": "U", "decoder": "D", "full": "BDS-Lite", "gsl": "U-Net+GSL"}
    m.figure()


def transfer() -> None:
    import make_audit_assets as m

    original = m.Line2D

    def relabelled(*args, **kwargs):
        if "label" in kwargs:
            kwargs["label"] = LEGEND_S10.get(kwargs["label"], kwargs["label"])
        return original(*args, **kwargs)

    m.Line2D = relabelled
    m.figure()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", type=Path)
    parser.add_argument(
        "--redraw",
        action="store_true",
        help="Explicitly redraw from external data and saved predictions.",
    )
    args = parser.parse_args()
    if not args.redraw:
        receipt = json.loads((ROOT / "paper/figures/published_assets.json").read_text())
        for name, expected in receipt["sha256"].items():
            path = ROOT / "paper/figures" / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Verified figure asset changed: {name}")
        print("verified published figure assets; no data loading or model execution")
        return
    if args.scratch is None:
        parser.error("--redraw requires --scratch")
    args.scratch.mkdir(parents=True, exist_ok=True)
    run_level(args.scratch)
    licensed()
    qualitative()
    ten_seed()
    transfer()
    print("wrote Figures S2, S3, S5, S7, S9 and S10")


if __name__ == "__main__":
    main()
