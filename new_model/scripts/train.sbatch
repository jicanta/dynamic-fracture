#!/bin/bash
# Train FractureTAU on Gilbreth.
#
# Submit from the new_model/ directory:
#   sbatch scripts/train.sbatch
# Optional overrides:
#   RUN_NAME=tau_sed EXTRA=SED sbatch scripts/train.sbatch
#
# Runs on the lab account `alta` (24 h limit — a full train+eval fits in one
# job). The job auto-resumes from the last checkpoint, so if it ever does get
# cut short, resubmitting just continues training.

#SBATCH --job-name=frac-tau
#SBATCH -A alta
#SBATCH --partition=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-train-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1   # epoch lines reach the slurm log immediately

# Call the env's python directly — immune to RCAC module-tree changes
# (the 2026-06 migration broke `module load anaconda` mid-project).
PYTHON=/scratch/gilbreth/jcantare/envs/fracture-sota/bin/python

cd "${SLURM_SUBMIT_DIR}"

RUN_NAME="${RUN_NAME:-tau_base}"
EXTRA="${EXTRA:-none}"          # none | pressure | vonmises | SED

# ---- improved recipe (Tier 1) -------------------------------------------
# Recall-oriented Focal-Tversky loss instead of Dice (beta>alpha penalizes
# false negatives), longer scheduled-sampling AR rollout in stage 2 so the
# model is trained in the long-horizon regime it is scored on, and automatic
# val threshold calibration. Override any of these via env vars; set
# TVERSKY_WEIGHT=0 DICE_WEIGHT=1 ROLLOUT_STEPS=4 to reproduce the old baseline.
TVERSKY_WEIGHT="${TVERSKY_WEIGHT:-1.0}"
DICE_WEIGHT="${DICE_WEIGHT:-0.0}"
TVERSKY_BETA="${TVERSKY_BETA:-0.7}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-6}"   # stage-2 BPTT memory scales with this;
                                      # raise to 12-16 if you have headroom
SS_START="${SS_START:-0.5}"

# ---- optional epoch overrides (Plan-06 budget trick) --------------------
# Leave UNSET to use the Config defaults (epochs_stage1=60, epochs_stage2=15).
# To produce the SHARED frozen Stage-1 checkpoint for sweep_stage2.sbatch, run a
# Stage-1-ONLY job:   RUN_NAME=stage1_shared EPOCHS_STAGE2=0 sbatch scripts/train.sbatch
# These are passed through only when set, so default training is byte-unchanged.
EPOCH_FLAGS=()
[[ -n "${EPOCHS_STAGE1:-}" ]] && EPOCH_FLAGS+=(--epochs-stage1 "${EPOCHS_STAGE1}")
[[ -n "${EPOCHS_STAGE2:-}" ]] && EPOCH_FLAGS+=(--epochs-stage2 "${EPOCHS_STAGE2}")

echo "Training run=${RUN_NAME} extra=${EXTRA} on $(hostname)"
echo "  tversky_w=${TVERSKY_WEIGHT} dice_w=${DICE_WEIGHT} beta=${TVERSKY_BETA} rollout=${ROLLOUT_STEPS}"
[[ ${#EPOCH_FLAGS[@]} -gt 0 ]] && echo "  epoch overrides: ${EPOCH_FLAGS[*]}"
nvidia-smi || true

"$PYTHON" -m src.train \
    --data-root DATASET \
    --out-root OUTPUTS \
    --run-name "${RUN_NAME}" \
    --extra "${EXTRA}" \
    --tversky-weight "${TVERSKY_WEIGHT}" \
    --dice-weight "${DICE_WEIGHT}" \
    --tversky-beta "${TVERSKY_BETA}" \
    --rollout-steps "${ROLLOUT_STEPS}" \
    --ss-start "${SS_START}" \
    ${EPOCH_FLAGS[@]+"${EPOCH_FLAGS[@]}"} \
    --resume auto

echo "Training finished. Now evaluating (calibrated threshold auto-loaded)..."
"$PYTHON" -m src.evaluate \
    --data-root DATASET \
    --out-root OUTPUTS \
    --run-name "${RUN_NAME}"
