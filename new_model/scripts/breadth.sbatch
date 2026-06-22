#!/bin/bash
# Per-mode breadth runs of the WINNING config (HPC-01 / D-09, Plan 04-08 Task 1).
# Runs the sweep-winning FractureTAU recipe END-TO-END (full Stage-1 + Stage-2) ONCE
# for each extra-physics mode {pressure, vonmises, SED} as a Slurm array. NO per-mode
# sweep and NO per-mode ablation (D-09 bounds compute: BASE gets the full treatment;
# the other three modes get a single headline run each to populate the breadth row).
# The input channel count auto-adjusts (assembler.n_channels) from --extra, so there
# is no model change. Each mode -> OUTPUTS/headline_<mode>/eval/<case>/per_frame_metrics.csv
# plus a checkpoints/best.pt for SHA provenance (D-14).
#
# Submit from new_model/:   sbatch scripts/breadth.sbatch
# Execution is GATED to plan 08 (uses the plan-05 winner; same recipe as headline_seeds.sbatch).
#
# The loss/AR knobs below are the plan-05 sweep WINNER (trial 20) — IDENTICAL to
# headline_seeds.sbatch; only --extra and --run-name vary per array task. Each mode is
# a multi-requeue ~12h full-train job; schedule these EARLY (RESEARCH A4 / Pitfall 8).

#SBATCH --job-name=frac-tau-breadth
#SBATCH -A alta
#SBATCH --partition=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00          # full Stage-1+Stage-2 (refined.sbatch uses 12h)
#SBATCH --array=0-2              # modes {pressure, vonmises, SED}
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm-breadth-%A_%a.out

set -euo pipefail
export PYTHONUNBUFFERED=1

PYTHON=/scratch/gilbreth/jcantare/envs/fracture-sota/bin/python
cd "${SLURM_SUBMIT_DIR}"
nvidia-smi || true

# --requeue alone does NOT survive a clean TIMEOUT (Pitfall 3); self-requeue on
# SIGUSR1 (--signal=B:USR1@120) so the per-epoch last.pt + --resume auto continue.
trap 'scontrol requeue ${SLURM_JOB_ID}; exit 0' USR1

MODES=(pressure vonmises SED)
MODE="${MODES[${SLURM_ARRAY_TASK_ID}]}"
echo "=== breadth run: extra=${MODE} seed=42 (full Stage1+Stage2) on $(hostname) ==="

# WINNING config (plan-05 sweep winner, trial 20) — same recipe as headline_seeds.sbatch.
# Only --extra / --run-name differ. Full epochs + --resume auto for requeue.
"$PYTHON" -m src.train \
    --data-root DATASET \
    --out-root OUTPUTS \
    --run-name "headline_${MODE}" \
    --extra "${MODE}" \
    --seed 42 \
    --tversky-weight 1.0 --dice-weight 0.0 --bce-weight 1.0 \
    --growth-weight 2.052706929853361 --tversky-gamma 1.3333 \
    --tversky-alpha 0.3981124582220783 --tversky-beta 0.6018875417779217 --boundary-weight 0.29413090293373934 \
    --ar-pushforward 1 --feedback-noise-std 0.07643768845069009 \
    --lr-stage2 0.0002136566938028427 \
    --rollout-steps 4 --val-rollout-steps 20 \
    --epochs-stage1 60 --epochs-stage2 15 \
    --resume auto

"$PYTHON" -m src.evaluate \
    --data-root DATASET \
    --out-root OUTPUTS \
    --run-name "headline_${MODE}"

echo "=== breadth extra=${MODE} complete -> OUTPUTS/headline_${MODE}/ " \
     "(eval/<case>/per_frame_metrics.csv + checkpoints/best.pt) ==="
