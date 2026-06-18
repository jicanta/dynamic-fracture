#!/bin/bash
# Headline 3-seed fan-out of the WINNING config (HPC-02 / D-05, Plan 04-03 Task 3).
# Runs the sweep-winning FractureTAU recipe END-TO-END (full Stage-1 + Stage-2) for
# the 3 seeds {42,43,44} as a Slurm array, so seed_aggregate.py can report per-case
# late-rollout macro F1 as mean±std (the honest run-to-run error bar: bf16 AMP +
# cuDNN are not byte-deterministic, so the seed spread IS the variation we report,
# Pitfall 9). Each seed -> OUTPUTS/headline_s{seed}/eval/<case>/per_frame_metrics.csv
# plus a checkpoints/best.pt that seed_aggregate/significance sha256-hash (D-14).
#
# Submit from new_model/:   sbatch scripts/headline_seeds.sbatch
# Execution is GATED to plan 06 (after the plan-05 sweep names the winner).
#
# IMPORTANT: EDIT the loss/AR knobs below to the plan-05 sweep WINNER before
# submitting — the values here are the tau_refined recipe placeholder. Each seed is
# a multi-requeue ~12h full-train job; schedule these EARLY (RESEARCH A4 / Pitfall 8,
# never the day before the milestone).

#SBATCH --job-name=frac-tau-headline
#SBATCH -A alta
#SBATCH --partition=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00          # full Stage-1+Stage-2 (refined.sbatch uses 12h)
#SBATCH --array=0-2              # seeds {42,43,44}
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm-headline-%A_%a.out

set -euo pipefail
export PYTHONUNBUFFERED=1

PYTHON=/scratch/gilbreth/jcantare/envs/fracture-sota/bin/python
cd "${SLURM_SUBMIT_DIR}"
nvidia-smi || true

# --requeue alone does NOT survive a clean TIMEOUT (Pitfall 3); self-requeue on
# SIGUSR1 (--signal=B:USR1@120) so the per-epoch last.pt + --resume auto continue.
trap 'scontrol requeue ${SLURM_JOB_ID}; exit 0' USR1

SEEDS=(42 43 44)
SEED="${SEEDS[${SLURM_ARRAY_TASK_ID}]}"
echo "=== headline run: seed=${SEED} (full Stage1+Stage2) on $(hostname) ==="

# WINNING config (placeholder = tau_refined recipe; SUBSTITUTE the plan-05 sweep
# winner's loss/AR knobs before submitting). Full epochs + --resume auto for requeue.
"$PYTHON" -m src.train \
    --data-root DATASET \
    --out-root OUTPUTS \
    --run-name "headline_s${SEED}" \
    --extra none \
    --seed "${SEED}" \
    --tversky-weight 1.0 --dice-weight 0.0 --bce-weight 1.0 \
    --growth-weight 4.0 --tversky-gamma 1.3333 \
    --tversky-alpha 0.3 --tversky-beta 0.7 --boundary-weight 0.5 \
    --ar-pushforward 1 --feedback-noise-std 0.05 \
    --rollout-steps 4 --val-rollout-steps 20 \
    --epochs-stage1 60 --epochs-stage2 15 \
    --resume auto

"$PYTHON" -m src.evaluate \
    --data-root DATASET \
    --out-root OUTPUTS \
    --run-name "headline_s${SEED}"

echo "=== headline seed=${SEED} complete -> OUTPUTS/headline_s${SEED}/ " \
     "(eval/<case>/per_frame_metrics.csv + checkpoints/best.pt for seed_aggregate) ==="
