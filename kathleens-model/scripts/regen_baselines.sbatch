#!/bin/bash
# Regenerate the FROZEN ConvLSTM (MODE 1 / BASE) baselines for all 16 canonical
# cases through the shared metric seam (Plan 01-08, D-01/C-01). Inference ONLY:
# the frozen .keras weights are SHA256-verified IDENTICAL before and after the
# run (A1/A2) -- the job ABORTS if any digest changes (frozen-weight immutability,
# threat T-08-01). evaluate.py val-calibrates the threshold on the clean trainDS
# val frames (CMP-04) and writes per_frame_metrics.csv under CANONICAL keys with
# the canonical schema (frame_idx,precision,recall,f1,iou,bce).
#
# Submit from the kathleens-model/ directory:
#   sbatch scripts/regen_baselines.sbatch
# Optional env: KM_DATA_ROOT=... KM_ENV_PATH=... RUN_NAME=base_regen
#
# If the standby 4h cap binds, split into a per-case array job or use the alta
# 24h allocation (A1).

#SBATCH --job-name=km-regen
#SBATCH -A alta
#SBATCH --partition=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-km-regen-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR}"

# Activate the packed TensorFlow-GPU env (sets KM_PYTHON + CUDA LD_LIBRARY_PATH).
source scripts/tf_env.sh

RUN_NAME="${RUN_NAME:-base_regen}"
REPO_ROOT="$(cd .. && pwd)"                  # kathleens-model -> dynamic-fracture
CKPT_DIR="${REPO_ROOT}/ALL_INPUTS/MODEL/BASE/checkpoints"

# --- frozen-weight immutability: SHA256 BEFORE (A1/A2) ---
SHA_BEFORE="$(find "${CKPT_DIR}" -name '*.keras' -print0 | sort -z | xargs -0 sha256sum)"
echo "[regen] BASE checkpoint SHA256 (before):"
echo "${SHA_BEFORE}"

# --- 16 canonical case keys from the single-source registry (Plan 02) ---
CASES="$("$KM_PYTHON" -c "import sys; sys.path.insert(0, '${REPO_ROOT}'); import case_registry; print(' '.join(case_registry.TEST_CASE_FOLDERS))")"
echo "[regen] cases: ${CASES}"

nvidia-smi || true
"$KM_PYTHON" -c "import tensorflow as tf; print('TF', tf.__version__, 'GPUs', tf.config.list_physical_devices('GPU'))"

# --- BASE x 16 inference-only rollout through the shared metric (MODE 1) ---
# shellcheck disable=SC2086  # intentional word-splitting: CASES -> separate args
"$KM_PYTHON" evaluate.py --mode 1 --use-ref-ckpt --run-name "${RUN_NAME}" --cases ${CASES}

# --- frozen-weight immutability: SHA256 AFTER; abort on any change ---
SHA_AFTER="$(find "${CKPT_DIR}" -name '*.keras' -print0 | sort -z | xargs -0 sha256sum)"
echo "[regen] BASE checkpoint SHA256 (after):"
echo "${SHA_AFTER}"
if [ "${SHA_BEFORE}" != "${SHA_AFTER}" ]; then
  echo "[regen] FATAL: BASE .keras weights changed during regeneration (frozen-weight violation)!" >&2
  exit 1
fi
echo "[regen] OK: frozen BASE weights unchanged; 16 baselines regenerated under canonical keys for run '${RUN_NAME}'."
