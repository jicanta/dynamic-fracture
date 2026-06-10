#!/bin/bash
# Submit N chained training jobs so training continues overnight on the
# standby queue (4 h limit each). Each job resumes from the last checkpoint
# (--resume auto), so the chain just keeps going; once training is complete,
# remaining jobs re-run evaluation (idempotent) and exit quickly.
#
# Usage (from new_model/):
#   bash scripts/chain_train.sh 6                     # 6 x 4h = up to 24h
#   RUN_NAME=tau_sed EXTRA=SED bash scripts/chain_train.sh 6

set -euo pipefail

N="${1:?usage: bash scripts/chain_train.sh <num_jobs>}"
export RUN_NAME="${RUN_NAME:-tau_base}"
export EXTRA="${EXTRA:-none}"

prev=""
for i in $(seq 1 "$N"); do
    if [ -z "$prev" ]; then
        jid=$(sbatch --parsable --export=ALL scripts/train.sbatch)
    else
        # afterany: run even if the previous job hit the time limit
        jid=$(sbatch --parsable --export=ALL --dependency=afterany:"$prev" scripts/train.sbatch)
    fi
    echo "submitted job $i/$N: $jid (RUN_NAME=$RUN_NAME EXTRA=$EXTRA)"
    prev="$jid"
done

echo
echo "Monitor with:  squeue -u \$USER"
echo "Cancel all:    scancel -u \$USER --name=frac-tau"
