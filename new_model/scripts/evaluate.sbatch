#!/bin/bash
# Evaluate an already-trained run on all (or selected) test cases.
#
#   sbatch scripts/evaluate.sbatch
#   RUN_NAME=tau_sed sbatch scripts/evaluate.sbatch
#   CASES="test_MS206_V400 test_inclusions_1_2" sbatch scripts/evaluate.sbatch

#SBATCH --job-name=frac-tau-eval
#SBATCH -A standby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-eval-%j.out

set -euo pipefail

module purge
module load anaconda
source activate fracture-sota 2>/dev/null || conda activate fracture-sota

cd "${SLURM_SUBMIT_DIR}"

RUN_NAME="${RUN_NAME:-tau_base}"
CASES="${CASES:-}"

# shellcheck disable=SC2086
python -m src.evaluate \
    --data-root DATASET \
    --out-root OUTPUTS \
    --run-name "${RUN_NAME}" \
    ${CASES:+--cases ${CASES}}
