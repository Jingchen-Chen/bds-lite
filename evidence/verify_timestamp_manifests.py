"""Check the files listed in the two timestamp manifests against their SHA-256 digests.

The manifests were frozen (and timestamped) with the original repository layout; each listed path
is resolved through evidence/path_map.csv. Usage, from the repository root:
    python evidence/verify_timestamp_manifests.py
Split manifests under recovered/followup/splits/ are git-ignored inputs; extract them from the
primary evidence archive first, or they are reported as missing.
"""

import csv
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = (
    "protocols/acdc_confirm/TIMESTAMP_MANIFEST.txt",
    "protocols/followup/TIMESTAMP_MANIFEST.txt",
)

with (ROOT / "evidence/path_map.csv").open(newline="") as stream:
    moved = {row["original_path"]: row["current_path"] for row in csv.DictReader(stream)}

bad = missing = good = 0
for manifest in MANIFESTS:
    for digest, original in re.findall(r"([0-9a-f]{64})  (\S+)", (ROOT / manifest).read_text()):
        path = ROOT / moved.get(original, original)
        if not path.is_file():
            print(f"MISSING  {original}")
            missing += 1
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            print(f"FAILED   {original} -> {path.relative_to(ROOT)}")
            bad += 1
        else:
            good += 1
print(f"{good} OK, {bad} failed, {missing} missing")
sys.exit(1 if bad else 0)
