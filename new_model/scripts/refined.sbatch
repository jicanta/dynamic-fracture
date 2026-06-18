#!/bin/bash
# Refined full-length FractureTAU run on the WINNING directional-sweep config
# (Plan 03-06 Task 3). The Stage-2 sweep (sweep_stage2.sbatch) ranked ~10
# Stage-2-only variants on the leakage-free logged val_f1_ar; the winner was
# `ar_both` (pushforward + feedback-noise) on the approved Tversky 0.3/0.7 +
# boundary-0.5 loss recipe. This trains that config end-to-end (full Stage 1 +
# Stage 2, single-seed=42 per D-05) into OUTPUTS/tau_refined/ — the checkpoint
# Plan 03-07 evaluates for the MODEL-04 head-to-head.
#
# Submit from new_model/:   sbatch scripts/refined.sbatch
# 24h alta account; --resume auto continues a preempted/requeued job.

#SBATCH --job-name=frac-tau-refined
#SBATCH -A alta
#SBATCH --partition=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-refined-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1

PYTHON=/scratch/gilbreth/jcantare/envs/fracture-sota/bin/python
cd "${SLURM_SUBMIT_DIR}"

echo "=== refined run: WINNER ar_both (full Stage1+Stage2, seed=42) on $(hostname) ==="
nvidia-smi || true

# Approved common loss recipe (D-03) + ar_both winning knobs + full epochs.
"$PYTHON" -m src.train \
    --data-root DATASET \
    --out-root OUTPUTS \
    --run-name tau_refined \
    --extra none \
    --seed 42 \
    --tversky-weight 1.0 --dice-weight 0.0 --bce-weight 1.0 \
    --growth-weight 4.0 --tversky-gamma 1.3333 \
    --tversky-alpha 0.3 --tversky-beta 0.7 --boundary-weight 0.5 \
    --ar-pushforward 1 --feedback-noise-std 0.05 \
    --rollout-steps 4 --val-rollout-steps 20 \
    --epochs-stage1 60 --epochs-stage2 15 \
    --resume auto

echo "=== refined run complete -> OUTPUTS/tau_refined/ (best.pt selected on AR late-rollout F1) ==="
