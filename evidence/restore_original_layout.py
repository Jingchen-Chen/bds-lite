"""Recreate the original revision/ layout as symbolic links, for rerunning frozen study scripts.

The study scripts under scripts/acdc_confirm/ and scripts/kd_followup/ are kept byte-identical to
their timestamped versions, so they still read the original paths (for example
revision/queue_acdc_confirm.yaml). Run this from the repository root before running them:
    python evidence/restore_original_layout.py
It only adds links under revision/ and preregistration/ (both git-ignored); existing files are not
touched.
"""

import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
made = 0
with (ROOT / "evidence/path_map.csv").open(newline="") as stream:
    for row in csv.DictReader(stream):
        link, source = ROOT / row["original_path"], ROOT / row["current_path"]
        if link.exists() or link.is_symlink() or not source.exists():
            continue
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(source, link.parent))
        made += 1
print(f"{made} links created")
