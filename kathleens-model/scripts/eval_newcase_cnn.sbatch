#!/bin/bash
# Phase 9 (new fracture cases): evaluate the FROZEN ConvLSTM reference (Kathleen's
# base/MODE-1 .keras) on the disjoint new-case roster (--case-set new) under the
# identical AR-rollout protocol. Inference only, frozen reference checkpoint.
#
#   Submit from the repo root (dynamic-fracture/):
#       sbatch kathleens-model/scripts/eval_newcase_cnn.sbatch
#
# Produces base_regen_newcase/<case>/per_frame_metrics.csv (3 cases).

#SBATCH --job-name=frac-newcase-cnn
#SBATCH -A alta
#SBATCH --partition=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-newcase-cnn-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR}"

# tf_env.sh sets KM_PYTHON + CUDA/cuDNN LD_LIBRARY_PATH for the packed gpu_tf env.
source kathleens-model/scripts/tf_env.sh

"$KM_PYTHON" kathleens-model/evaluate.py \
    --mode 1 \
    --use-ref-ckpt \
    --case-set new \
    --run-name base_regen_newcase
