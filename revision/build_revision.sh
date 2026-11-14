#!/usr/bin/env bash
# Rebuild every revision deliverable in dependency order and fail loudly.
#   1. keys, tables, figure and word counts (make_distillation_assets.py, make_final_assets.py,
#      checked by verify_final_assets.py)
#   2. manuscript and supplement (tectonic, via build_check.sh)
#   3. response letter revision/response.md, rendered from response.template.md with numbers from
#      the generated keys and line numbers read from the compiled main.pdf (render_response.py)
#   4. editor letter (rendered from its template)
# The earlier LaTeX letter (response_to_reviewers.tex, with letter_locations.py) is superseded by
# response.md and is no longer built; response_to_reviewers.pdf is now rendered from response.md.
# The manuscript is compiled twice so that word counts and pointers refer to the final text.
# Usage: bash revision/build_revision.sh   (from the repository root)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTHONPATH=src:.
$PY analysis/revision/make_distillation_assets.py
$PY analysis/revision/make_final_assets.py
# review round 2026-11-12: formats revision/posthoc_review_20261112/*.csv (keys RV.*, S24 tables)
$PY analysis/revision/make_review_assets.py
# main-text presentation layer: three decimals, one result vocabulary (numbers_round3.tex, r3_*.tex)
$PY analysis/revision/make_main_round3.py
$PY analysis/revision/verify_final_assets.py
for pass in 1 2; do
  bash revision/build_check.sh revision/manuscript main.tex
  bash revision/build_check.sh revision/manuscript supplement.tex
  $PY analysis/revision/make_distillation_assets.py >/dev/null
done
# The response letter needs two local-only inputs that are not in the repository, because the
# reviewers' text must not be published: revision/peerj-reviewing-142937-v0.pdf (read by
# render_response.py) and revision/reviewer.md (read by check_response_quotes.py). Without them the
# letter steps are skipped with a message, and the committed response.md and its PDF are left as they are.
if [ -f revision/peerj-reviewing-142937-v0.pdf ]; then
  $PY analysis/revision/render_response.py
  # PDF of the response letter (replaces the stale LaTeX letter's PDF; response_to_reviewers.tex stays superseded)
  "$HOME/.local/bin/pandoc" revision/response.md -o revision/response_to_reviewers.pdf \
    --pdf-engine="$HOME/.local/bin/tectonic" -V geometry:margin=2.2cm -V fontsize=10pt -V colorlinks=true \
    -V mainfont="Liberation Serif" -V monofont="DejaVu Sans Mono"
else
  echo "SKIPPED render_response.py and the letter PDF: local input revision/peerj-reviewing-142937-v0.pdf"
  echo "  is missing (not distributed); revision/response.md, PEERJ_REQUIREMENTS_CHECKLIST.md and"
  echo "  response_to_reviewers.pdf were not rebuilt, so their line numbers may not match a changed main.pdf."
fi
if [ -f revision/reviewer.md ]; then
  $PY analysis/revision/check_response_quotes.py
else
  echo "SKIPPED check_response_quotes.py: local input revision/reviewer.md is missing (not distributed)."
fi
$PY analysis/revision/render_editor_letter.py
for pdf in revision/manuscript/main.pdf revision/manuscript/supplement.pdf; do
  if pdftotext "$pdf" - | grep -n -E '\[loc:|\[float:|\[[A-Za-z0-9_-]+\.[A-Za-z0-9_.-]+\]|\?\?'; then
    echo "FAILED: unresolved key, pointer or reference in $pdf"; exit 1
  fi
done
# Author placeholders: listed on every build; fatal only for a release build (BDS_RELEASE=1).
items=0
for pdf in revision/manuscript/main.pdf revision/manuscript/supplement.pdf; do
  found=$(pdftotext "$pdf" - | tr '\n' ' ' | sed 's/  */ /g' | grep -o -E '\[(TODO|AUTHOR DECISION|AUTHOR ITEM|CLAIM-CHECK)[^]]*\]' || true)
  if [ -n "$found" ]; then echo "author items in $pdf:"; echo "$found"; items=$((items + $(echo "$found" | wc -l))); fi
done
found=$(grep -o -E '\[(TODO|AUTHOR DECISION|AUTHOR ITEM|CLAIM-CHECK)[^]]*\]' revision/response.md || true)
if [ -n "$found" ]; then echo "author items in revision/response.md:"; echo "$found"; items=$((items + $(echo "$found" | wc -l))); fi
if [ "$items" -gt 0 ] && [ "${BDS_RELEASE:-0}" = "1" ]; then echo "FAILED: $items author placeholders remain"; exit 1; fi
echo "build complete ($items author placeholders listed above)"
