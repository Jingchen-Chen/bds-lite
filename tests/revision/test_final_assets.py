"""Formatting and reading rules used by analysis/revision/make_final_assets.py."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis/revision"))

from analysis.revision import make_final_assets as fa  # noqa: E402


def test_holm_reading_follows_adjusted_decision():
    assert fa.holm_reading("gain, size unclear", 0.0196) == "gain, size unclear"
    assert fa.holm_reading("gain, size unclear", 0.0945) == "unresolved"
    assert fa.holm_reading("undetermined", None) == "undetermined"


def test_running_text_interval():
    assert fa.ndash(-0.0118, 0.0044) == "$-$0.0118 to 0.0044"
    assert fa.ndash(0.34, 2.38, 2) == "0.34 to 2.38"


def test_design_count_matches_freeze():
    assert sum(fa.DESIGN_SLOTS.values()) == 237
