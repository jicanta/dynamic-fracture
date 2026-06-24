#!/bin/bash
# Evaluate a checkpoint with Kathleen's autoregressive rollout (notebook cells
# 35-45) and write per-case metrics. Default loads HER reference checkpoint.
#
# Submit from kathleens-model/:
#   MODE=4 RUN_NAME=sed_ref_eval sbatch scripts/evaluate.sbatch          # her weights
#   MODE=1 RUN_NAME=base_repro_eval MODEL_PATH=OUTPUTS/base_repro/checkpoints/ep130-*.keras \
#       sbatch scripts/evaluate.sbatch                                   # your weights
# Optional: CASES="test_MS206_V400 test_MS206_V200"  KM_DATA_ROOT=...  KM_ENV_PATH=...
# Optional: EXTRA_ARGS="--no-calibrate --threshold 0.5"  # extra evaluate.py flags
#   (REQUIRED for the Phase-6 Tier-2 reproduction eval so it is apples-to-apples
#    vs her fixed-0.5 reference; evaluate.py otherwise defaults to calibration.)

#SBATCH --job-name=km-eval
#SBATCH -A alta
#SBATCH --partition=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-km-eval-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR}"

source scripts/tf_env.sh

MODE="${MODE:-1}"
declare -A MODE_NAME=( [1]=base [2]=pressure [3]=von [4]=sed )
RUN_NAME="${RUN_NAME:-${MODE_NAME[$MODE]}_eval}"

echo "Evaluating Kathleen-model MODE=${MODE} run=${RUN_NAME} on $(hostname)"
nvidia-smi || true

ARGS=(--mode "${MODE}" --run-name "${RUN_NAME}")
if [ -n "${MODEL_PATH:-}" ]; then
    ARGS+=(--model-path "${MODEL_PATH}")
else
    ARGS+=(--use-ref-ckpt)
fi
[ -n "${CASES:-}" ] && ARGS+=(--cases ${CASES})
# Append optional passthrough flags (word-split intentionally for multiple flags).
# shellcheck disable=SC2206
[ -n "${EXTRA_ARGS:-}" ] && ARGS+=(${EXTRA_ARGS})

"$KM_PYTHON" evaluate.py "${ARGS[@]}"

echo
echo "=== Verifying against Kathleen's reference per-frame metrics ==="
"$KM_PYTHON" verify.py --mode "${MODE}" --run-dir "OUTPUTS/${RUN_NAME}" || true
