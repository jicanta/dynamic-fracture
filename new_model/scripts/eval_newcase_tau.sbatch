#!/bin/bash
# Phase 9 (new fracture cases): evaluate the 3 FROZEN FractureTAU headline seeds
# on the disjoint new-case roster (--case-set new), writing under the DISTINCT
# eval_newcase/ subdir so the frozen v1.0 eval/ is never clobbered. Base/MODE-1
# fracture-mask checkpoints only (D-05) — inference only, no retraining.
#
#   Submit from new_model/:  sbatch scripts/eval_newcase_tau.sbatch
#
# Produces headline_s{42,43,44}/eval_newcase/<case>/per_frame_metrics.csv (3 cases each).

#SBATCH --job-name=frac-newcase-tau
#SBATCH -A standby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-newcase-tau-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1

# Direct env python — immune to RCAC module-tree changes (per evaluate_tf.sbatch).
PYTHON=/scratch/gilbreth/jcantare/envs/fracture-sota/bin/python

cd "${SLURM_SUBMIT_DIR}"

for s in 42 43 44; do
    echo "=== FractureTAU seed ${s} (new cases, eval_newcase) ==="
    "$PYTHON" -m src.evaluate \
        --data-root DATASET \
        --out-root OUTPUTS \
        --run-name "headline_s${s}" \
        --case-set new \
        --eval-dir-name eval_newcase
done
