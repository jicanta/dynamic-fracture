#!/bin/bash
# PHYS-04 teacher-forced inference pass (freeze-legal per D-14: inference only, no
# weight updates). Runs the frozen best.pt over all (or selected) test cases with
# --teacher-forced, writing a PARALLEL per_frame_metrics_tf.csv per case alongside
# the existing AR per_frame_metrics.csv. The AR CSVs / probs.npz are NOT touched.
#
#   RUN_NAME=tau_refined sbatch scripts/evaluate_tf.sbatch
#   RUN_NAME=tau_refined CASES="test_MS206_V400 test_inclusions_1_2" sbatch scripts/evaluate_tf.sbatch
#
# After it finishes, rsync the TF CSVs back to the laptop:
#   rsync -avz --include='*/' --include='per_frame_metrics_tf.csv' --exclude='*' \
#     jcantare@gilbreth.rcac.purdue.edu:/home/jcantare/scratch/dynamic-fracture/new_model/OUTPUTS/tau_refined/eval/ \
#     new_model/OUTPUTS/tau_refined/eval/

#SBATCH --job-name=frac-tau-eval-tf
#SBATCH -A standby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-eval-tf-%j.out

set -euo pipefail

module purge
module load anaconda
source activate fracture-sota 2>/dev/null || conda activate fracture-sota

cd "${SLURM_SUBMIT_DIR}"

RUN_NAME="${RUN_NAME:-tau_refined}"
CASES="${CASES:-}"

# --teacher-forced is the PHYS-04 data-source flag (added in 05-05): feeds GT
# channel-0 each step and writes per_frame_metrics_tf.csv (never per_frame_metrics.csv).
# shellcheck disable=SC2086
python -m src.evaluate \
    --data-root DATASET \
    --out-root OUTPUTS \
    --run-name "${RUN_NAME}" \
    --teacher-forced \
    ${CASES:+--cases ${CASES}}
