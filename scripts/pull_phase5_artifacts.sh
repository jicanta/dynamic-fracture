#!/bin/bash
# Pull the Phase-5 headline artifacts from Gilbreth ONCE (RESEARCH Pitfall 1).
#
# Only `tau_base` eval artifacts exist locally. Every Phase-5 number is wrong if
# computed against them: PHYS-04 (teacher-forced) needs `best.pt`; PHYS-05
# (calibration) needs the `tau_refined` per-case probs.npz/gt.npz; PHYS-06 (FLOPs)
# needs `best.pt`. This script pulls the WINNING `tau_refined` run, the Phase-4
# 16-case ConvLSTM reference `base_regen` (NOT the Phase-6 `base_repro`), and the
# 3 headline seed runs {42,43,44}. Pulling existing artifacts is freeze-legal —
# it is NOT a new experiment.
#
# Usage (prompts for BoilerKey 2FA — needs an interactive terminal):
#   bash dynamic-fracture/scripts/pull_phase5_artifacts.sh
#
# Override the cluster location without editing this file (Security T-05-04 — no
# non-parameterized scratch paths are baked in below the vars block):
#   REMOTE=user@host REMOTE_BASE=/path/to/dynamic-fracture bash scripts/pull_phase5_artifacts.sh
set -euo pipefail

# ---- parameterizable cluster vars (override via env; defaults = interfaces) ----
REMOTE="${REMOTE:-jcantare@gilbreth.rcac.purdue.edu}"
REMOTE_BASE="${REMOTE_BASE:-/home/jcantare/scratch/dynamic-fracture}"
NEW_OUT="${NEW_OUT:-$REMOTE_BASE/new_model/OUTPUTS}"
KM_OUT="${KM_OUT:-$REMOTE_BASE/kathleens-model/OUTPUTS}"
LOCAL="${LOCAL:-$HOME/Desktop/trabajo/surf-purdue-2026/dynamic-fracture}"

# Headline run names (Phase-4 04-03/04-06 conventions)
TAU_RUN="${TAU_RUN:-tau_refined}"
BASE_RUN="${BASE_RUN:-base_regen}"
SEED_RUNS=("headline_s42" "headline_s43" "headline_s44")

# Skip the heavy per-frame PNG dirs but keep *.npz, config.json, best.pt,
# per_frame_metrics.csv, training_log.csv, calibration.json (same idiom as
# scripts/pull_regen.sh).
RSYNC_OPTS=(-avz --exclude='*.png' --exclude='frames/')

# 1) tau_refined: best.pt (under checkpoints/) + per-case probs.npz/gt.npz
#    (under eval/<case>/) + config.json. THE headline model — SHA f1a0786c…
echo "=== pulling $TAU_RUN (headline FractureTAU) ==="
rsync "${RSYNC_OPTS[@]}" \
  "$REMOTE:$NEW_OUT/$TAU_RUN/" \
  "$LOCAL/new_model/OUTPUTS/$TAU_RUN/"

# 2) base_regen: Phase-4 16-case ConvLSTM reference (NOT base_repro)
echo "=== pulling $BASE_RUN (Phase-4 16-case ConvLSTM reference) ==="
rsync "${RSYNC_OPTS[@]}" \
  "$REMOTE:$KM_OUT/$BASE_RUN/" \
  "$LOCAL/kathleens-model/OUTPUTS/$BASE_RUN/"

# 3) the 3 headline seed runs (seeds 42/43/44)
for SEED_RUN in "${SEED_RUNS[@]}"; do
  echo "=== pulling $SEED_RUN (headline seed run) ==="
  rsync "${RSYNC_OPTS[@]}" \
    "$REMOTE:$NEW_OUT/$SEED_RUN/" \
    "$LOCAL/new_model/OUTPUTS/$SEED_RUN/"
done

echo
echo "Pulled headline artifacts to $LOCAL"
echo
echo "NEXT — assert provenance BEFORE computing any Phase-5 number (Pitfall 1):"
echo "  cd $LOCAL && python scripts/assert_provenance.py"
echo "  expect: PROVENANCE OK: best.pt f1a0786c…"
echo
echo "Then sanity-check the pulled artifacts:"
echo "  ls $LOCAL/new_model/OUTPUTS/$TAU_RUN/eval/*/probs.npz | head   # PHYS-05 raw probs"
echo "  ls $LOCAL/kathleens-model/OUTPUTS/$BASE_RUN/                   # 16-case CNN ref"
