#!/bin/bash
# Train FractureTAU on Gilbreth.
#
# Submit from the new_model/ directory:
#   sbatch scripts/train.sbatch
# Optional overrides:
#   RUN_NAME=tau_sed EXTRA=SED sbatch scripts/train.sbatch
#
# Pick your account with `slist` (replace `standby` below with your lab
# account for jobs longer than 4 h). The job auto-resumes from the last
# checkpoint, so resubmitting after a standby timeout just continues training.

#SBATCH --job-name=frac-tau
#SBATCH -A standby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-train-%j.out

set -euo pipefail

module purge
module load anaconda
# Use the env created in the README setup step:
source activate fracture-sota 2>/dev/null || conda activate fracture-sota

cd "${SLURM_SUBMIT_DIR}"

RUN_NAME="${RUN_NAME:-tau_base}"
EXTRA="${EXTRA:-none}"          # none | pressure | vonmises | SED

echo "Training run=${RUN_NAME} extra=${EXTRA} on $(hostname)"
nvidia-smi || true

python -m src.train \
    --data-root DATASET \
    --out-root OUTPUTS \
    --run-name "${RUN_NAME}" \
    --extra "${EXTRA}" \
    --resume auto

echo "Training finished. Now evaluating..."
python -m src.evaluate \
    --data-root DATASET \
    --out-root OUTPUTS \
    --run-name "${RUN_NAME}"
