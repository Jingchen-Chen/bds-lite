#!/usr/bin/env bash
# Compile one LaTeX file and fail loudly. tectonic can crash in xdvipdfmx (create_font_alias) after
# TeX succeeds, leaving an old PDF in place, so check the exit code and the PDF timestamp.
# Usage: build_check.sh <directory> <file.tex>
set -u
cd "$1" || exit 2
pdf="${2%.tex}.pdf"
start=$(date +%s)
log_dir="${BDS_PAPER_LOG_DIR:-/tmp/bds-lite-paper-build}"
mkdir -p "$log_dir" || exit 2
log=$(mktemp "$log_dir/${2%.tex}.XXXXXX.log")
"${TECTONIC:-$HOME/.local/bin/tectonic}" -X compile "$2" >"$log" 2>&1
code=$?
if [ "$code" -ne 0 ]; then echo "FAILED $2: tectonic exit $code"; grep -E "^error" "$log" | head; exit 1; fi
if [ ! -f "$pdf" ] || [ "$(stat -c %Y "$pdf")" -lt "$start" ]; then echo "FAILED $2: PDF not rewritten"; exit 1; fi
echo "OK $2: $(pdfinfo "$pdf" | awk '/Pages/{print $2}') pages, $(grep -c '^warning' "$log") warnings"
grep -E "Citation|Reference .* undefined|multiply" "$log" | head || true
