#!/usr/bin/env bash
# gate_pdffonts.sh — Type-3 font gate over paper PDFs (PAPER2-01, D-06).
#
# Every figures/*.pdf must embed only proper (Type-1/Type-42/TrueType/CID) fonts.
# A "Type 3" font means a bitmap/figure font slipped in (matplotlib default before
# the pdf.fonttype=42 fix) — FAIL if any PDF reports one. If poppler's `pdffonts`
# is unavailable the gate SKIPs (exit 0): font embedding is an optional local check,
# not a hard build dependency.
#
# Run from anywhere (resolves its own paths):
#   bash paper/scripts/gate_pdffonts.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PAPER_DIR=$(dirname "$SCRIPT_DIR")
FIG_DIR="$PAPER_DIR/figures"

if ! command -v pdffonts >/dev/null 2>&1; then
  echo "[gate_pdffonts] SKIP: pdffonts (poppler) not installed — cannot check font embedding"
  exit 0
fi

fail=0
shopt -s nullglob
pdfs=("$FIG_DIR"/*.pdf)
if [ ${#pdfs[@]} -eq 0 ]; then
  echo "[gate_pdffonts] no figures/*.pdf found — nothing to check"
  exit 0
fi

for f in "${pdfs[@]}"; do
  # tail -n +3 strips the two-line header so prose/column names cannot false-hit.
  if pdffonts "$f" | tail -n +3 | grep -q 'Type 3'; then
    echo "[gate_pdffonts] FAIL: Type 3 font in $(basename "$f")"
    fail=1
  else
    echo "[gate_pdffonts] PASS: $(basename "$f")"
  fi
done

if [ "$fail" -ne 0 ]; then
  echo "[gate_pdffonts] FAIL: one or more PDFs embed a Type 3 font"
  exit 1
fi
echo "[gate_pdffonts] OK: no Type 3 fonts"
