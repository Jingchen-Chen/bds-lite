#!/usr/bin/env bash
# Rebuild the paper from the frozen results, in dependency order, and fail loudly.
#   1. number keys, tables and figures (paper/generated/, paper/figures/), checked by verify_final_assets.py
#   2. manuscript and supplement (tectonic, via build_check.sh), compiled twice so that word counts
#      and pointers refer to the final text
#   3. a check for unresolved keys, pointers or references in both PDFs
# No frozen analysis or training is rerun.
# Usage: bash paper/build.sh   (from the repository root)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTHONPATH=src:.
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/bds-lite-matplotlib}"
$PY analysis/revision/build_paper_assets.py
for pass in 1 2; do
  bash paper/build_check.sh paper main.tex
  bash paper/build_check.sh paper supplement.tex
done
for pdf in paper/main.pdf paper/supplement.pdf; do
  if pdftotext "$pdf" - | grep -n -E '\[loc:|\[float:|\[[A-Za-z0-9_-]+\.[A-Za-z0-9_.-]+\]|\?\?'; then
    echo "FAILED: unresolved key, pointer or reference in $pdf"; exit 1
  fi
  if pdftotext "$pdf" - | tr '\n' ' ' | grep -q -E '\[(TODO|AUTHOR DECISION|AUTHOR ITEM|CLAIM-CHECK)'; then
    echo "FAILED: author placeholder in $pdf"; exit 1
  fi
done
echo "build complete"
