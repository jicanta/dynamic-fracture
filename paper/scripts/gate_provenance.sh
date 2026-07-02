#!/usr/bin/env bash
# gate_provenance.sh — number -> source-artifact cross-check gate (PAPER2-02, T-07-05).
#
# Every quantitative claim in main.tex must be traceable to a measured first-party
# artifact — no hand-edited or remembered numbers. For each body number below, the
# gate asserts it appears BOTH in main.tex AND in its declared source artifact, and
# FAILS if a number is present in main.tex but cannot be traced back to its source
# (an untraceable / fabricated claim).
#
# Sources (declared, not transcribed as absolute paths — resolved relative to repo):
#   0.8621 0.5767 1.526e-5  -> results/comparisons/tau_headline_vs_base_regen.md
#   39.585 0.6258 0.0435    -> results/phys_metrics/PHYS_REPORT.md
#
# NOTE: This gate is RUN by plan 07-04 after the provenance table lands; it is
# Wave-0 scaffolding here. Run from anywhere (resolves its own paths):
#   bash paper/scripts/gate_provenance.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PAPER_DIR=$(dirname "$SCRIPT_DIR")
REPO_DIR=$(dirname "$PAPER_DIR")

TEX="$PAPER_DIR/main.tex"
COMPARISON="$REPO_DIR/results/comparisons/tau_headline_vs_base_regen.md"
PHYS="$REPO_DIR/results/phys_metrics/PHYS_REPORT.md"
# Phase-9 new-case (held-out / OOD) head-to-head artifact (09-08). Every new-case
# number cited in Sec.~newcase must trace back here — no hand-typed/recomputed value.
NEWCASE="$REPO_DIR/new_model/OUTPUTS/newcase_comparison.md"

# number -> source-artifact mapping (parallel arrays).
# First six: v1.0 headline + physics (byte-unchanged, gated to their own sources).
# Last three: Phase-9 new-case aggregate FractureTAU mean, ConvLSTM mean, and the
# indicative Wilcoxon p — traced to $NEWCASE so the gate can't false-pass on them.
NUMBERS=(0.8621 0.5767 1.526e-5 39.585 0.6258 0.0435 0.9252 0.5974 0.125)
SOURCES=("$COMPARISON" "$COMPARISON" "$COMPARISON" "$PHYS" "$PHYS" "$PHYS" "$NEWCASE" "$NEWCASE" "$NEWCASE")

if [ ! -f "$TEX" ]; then
  echo "[gate_provenance] FAIL: main.tex not found at $TEX"
  exit 1
fi

fail=0
for i in "${!NUMBERS[@]}"; do
  num="${NUMBERS[$i]}"
  src="${SOURCES[$i]}"
  src_name="${src#"$REPO_DIR"/}"

  in_tex=0
  grep -Fq "$num" "$TEX" && in_tex=1

  in_src=0
  if [ -f "$src" ]; then
    grep -Fq "$num" "$src" && in_src=1
  fi

  if [ "$in_tex" -eq 1 ] && [ "$in_src" -eq 1 ]; then
    echo "[gate_provenance] PASS: $num traceable to $src_name"
  elif [ "$in_tex" -eq 1 ] && [ "$in_src" -eq 0 ]; then
    echo "[gate_provenance] FAIL: $num in main.tex but NOT traceable to $src_name"
    fail=1
  elif [ "$in_tex" -eq 0 ] && [ "$in_src" -eq 1 ]; then
    echo "[gate_provenance] WARN: $num in $src_name but not yet cited in main.tex"
  else
    echo "[gate_provenance] FAIL: $num found in neither main.tex nor $src_name"
    fail=1
  fi
done

if [ "$fail" -ne 0 ]; then
  echo "[gate_provenance] FAIL: untraceable number(s) in main.tex"
  exit 1
fi
echo "[gate_provenance] OK: all cited numbers traceable to source artifacts"
