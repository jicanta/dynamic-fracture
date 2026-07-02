#!/bin/bash
# Push local dynamic-fracture CODE (not data/outputs) to Gilbreth.
#
# Syncs everything the cluster needs to run BOTH pipelines + the comparison chain:
#   - repo-root modules: case_registry.py, frac_metrics.py
#   - new_model/src + new_model/scripts   (FractureTAU pipeline + sbatch)
#   - kathleens-model source/scripts/entrypoints (ConvLSTM pipeline)
#   - scripts/ (assert_provenance.py, ...) + tools/ (exo2csv)
# Does NOT touch DATASET, OUTPUTS, ALL_INPUTS, ALL_OUTPUTS, results, or checkpoints.
#
# History: previously synced only new_model/src + new_model/scripts, which left
# case_registry.py (09-02) and kathleens-model/ (09-06) stale on the cluster and
# broke --case-set on Gilbreth (09-07 finding). Now uses rsync -R (relative) to
# recreate each named code path under the repo root in ONE ssh connection.
#
# Usage: bash new_model/scripts/push_code.sh        (prompts for BoilerKey)
#        DRY_RUN=1 bash new_model/scripts/push_code.sh   (preview only)
set -euo pipefail

LOCAL_ROOT="$HOME/Desktop/trabajo/surf-purdue-2026/dynamic-fracture"
REMOTE="jcantare@gilbreth.rcac.purdue.edu"
REMOTE_ROOT="/home/jcantare/scratch/dynamic-fracture"

FLAGS="-avzR"
[ -n "${DRY_RUN:-}" ] && FLAGS="-avzRn" && echo "DRY RUN — nothing will be copied"

cd "$LOCAL_ROOT"

# -R (relative) preserves each path under REMOTE_ROOT; a single connection = one
# BoilerKey prompt. Only code paths are named, so data/outputs are never touched.
rsync $FLAGS \
    case_registry.py \
    frac_metrics.py \
    new_model/src new_model/scripts \
    kathleens-model/source kathleens-model/scripts kathleens-model/evaluate.py kathleens-model/config.py \
    scripts \
    tools \
    "$REMOTE:$REMOTE_ROOT/"

echo
echo "Code synced. Running jobs are unaffected (they loaded their code at start)."
