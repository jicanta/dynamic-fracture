#!/bin/bash
# Reproduce Kathleen's SED (mode 4) results at HER settings (threshold 0.5, no
# calibration) for the 6 SED cases, so verify_against_kathleen.py can confirm our
# pipeline matches her numbers. Inference-only on the frozen .keras (SHA256
# checked pre/post). SED mirror of repro_baselines.sbatch (which is the BASE/mode-1
# fidelity check) -- this run is the SED Tier-1 fidelity check.
#
# Submit from kathleens-model/:  sbatch scripts/repro_baselines_sed.sbatch

#SBATCH --job-name=km-repro-sed
#SBATCH -A alta
#SBATCH --partition=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-km-repro-sed-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"
source scripts/tf_env.sh

REPO_ROOT="$(cd .. && pwd)"
CKPT_DIR="${REPO_ROOT}/ALL_INPUTS/MODEL/SED/checkpoints"
SHA_BEFORE="$(find "${CKPT_DIR}" -name '*.keras' -print0 | sort -z | xargs -0 sha256sum)"

# The 6 SED cases gated by verify.py at threshold 0.5 (Tier-1 parity).
CASES="test_horizontal_layers_4 test_inclusions_1_2 test_MS206_V100 test_MS206_V200 test_MS206_V400 test_MS206_V1000"

nvidia-smi || true
# shellcheck disable=SC2086
"$KM_PYTHON" evaluate.py --mode 4 --use-ref-ckpt --no-calibrate --threshold 0.5 \
    --run-name sed_repro --cases ${CASES}

SHA_AFTER="$(find "${CKPT_DIR}" -name '*.keras' -print0 | sort -z | xargs -0 sha256sum)"
if [ "${SHA_BEFORE}" != "${SHA_AFTER}" ]; then
  echo "[repro] FATAL: SED .keras weights changed during reproduction!" >&2
  exit 1
fi
echo "[repro] OK: frozen weights unchanged; sed_repro (threshold 0.5) written for the 6 SED cases."
