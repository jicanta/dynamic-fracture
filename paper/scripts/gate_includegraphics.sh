#!/usr/bin/env bash
# gate_includegraphics.sh — extensionless-\includegraphics + cluster-path gate (PAPER2-01/02).
#
# Two honesty/portability checks over main.tex:
#  (1) NO \includegraphics may carry a file extension. LaTeX picks the best driver
#      format (.pdf > .png) only when the extension is omitted; a hardcoded .png
#      pins the raster even after a vector .pdf is produced. FAIL on any
#      \includegraphics{...png|jpg|jpeg|pdf}.
#  (2) NO cluster/user absolute path may leak into the published source
#      (/home/, /scratch/, gilbreth, jcantare, jicanta, /apps/spack) — T-05-22/T-07-04.
#
# Run from anywhere (resolves its own paths):
#   bash paper/scripts/gate_includegraphics.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PAPER_DIR=$(dirname "$SCRIPT_DIR")
TEX="$PAPER_DIR/main.tex"

if [ ! -f "$TEX" ]; then
  echo "[gate_includegraphics] FAIL: main.tex not found at $TEX"
  exit 1
fi

fail=0

# (1) extensionless \includegraphics check.
ext_hits=$(grep -nE '\\includegraphics(\[[^]]*\])?\{[^}]*\.(png|jpe?g|pdf)\}' "$TEX" || true)
if [ -n "$ext_hits" ]; then
  echo "[gate_includegraphics] FAIL: \\includegraphics with a file extension (drop it; let LaTeX pick .pdf):"
  echo "$ext_hits"
  fail=1
else
  echo "[gate_includegraphics] PASS: no extensioned \\includegraphics"
fi

# (2) cluster/user-path leak check.
path_hits=$(grep -niE '/home/|/scratch/|gilbreth|jcantare|jicanta|/apps/spack' "$TEX" || true)
if [ -n "$path_hits" ]; then
  echo "[gate_includegraphics] FAIL: cluster/user path leaked into main.tex:"
  echo "$path_hits"
  fail=1
else
  echo "[gate_includegraphics] PASS: no cluster/user paths in main.tex"
fi

if [ "$fail" -ne 0 ]; then
  exit 1
fi
echo "[gate_includegraphics] OK"
