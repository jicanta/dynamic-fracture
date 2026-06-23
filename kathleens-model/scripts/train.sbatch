#!/bin/bash
# Train Kathleen's ConvLSTM (job-submittable port of her notebook) on Gilbreth.
#
# Submit from the kathleens-model/ directory:
#   MODE=1 RUN_NAME=base_repro sbatch scripts/train.sbatch
# Optional: EPOCHS=130 INITIAL_EPOCH=0 MODEL_PATH=... KM_DATA_ROOT=... KM_ENV_PATH=...
#
# MODE: 1=base 2=+pressure 3=+vonmises 4=+SED

#SBATCH --job-name=km-train
#SBATCH -A alta
#SBATCH --partition=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-km-train-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR}"

# Activate the packed TensorFlow-GPU env (sets KM_PYTHON + CUDA LD_LIBRARY_PATH).
source scripts/tf_env.sh

MODE="${MODE:-1}"
RUN_NAME="${RUN_NAME:-}"
EPOCHS="${EPOCHS:-130}"
INITIAL_EPOCH="${INITIAL_EPOCH:-0}"

echo "Training Kathleen-model MODE=${MODE} run=${RUN_NAME:-<default>} epochs=${EPOCHS} on $(hostname)"
nvidia-smi || true
"$KM_PYTHON" -c "import tensorflow as tf; print('TF', tf.__version__, 'GPUs', tf.config.list_physical_devices('GPU'))"

ARGS=(--mode "${MODE}" --epochs "${EPOCHS}" --initial-epoch "${INITIAL_EPOCH}")
[ -n "${RUN_NAME}" ] && ARGS+=(--run-name "${RUN_NAME}")
[ -n "${MODEL_PATH:-}" ] && ARGS+=(--model-path "${MODEL_PATH}")
[ -n "${SEED:-}" ] && ARGS+=(--seed "${SEED}")

"$KM_PYTHON" train.py "${ARGS[@]}"
