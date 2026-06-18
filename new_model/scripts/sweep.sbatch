#!/bin/bash
# Optuna + ASHA Stage-2 sweep for FractureTAU (HPC-01 / HPC-05, D-01..D-04, D-10).
#
# Each array task is an IDENTICAL Optuna worker that attaches to ONE shared study
# (JournalStorage + JournalFileBackend on Gilbreth scratch) via load_if_exists and
# pulls trials until its 4h cap. ASHA (SuccessiveHalvingPruner) kills weak trials
# early so dozens of configs fit the cap.
#
# THE BUDGET TRICK (D-02): every trial is a Stage-2-only fine-tune from a SHARED
# frozen Stage-1 checkpoint (--init-from <stage1> --epochs-stage1 0). Train Stage-1
# ONCE, freeze it, point this array at it. Do NOT retrain Stage-1 per task.
#
# STEP 0 (ONCE, separately): produce the shared frozen Stage-1 checkpoint, e.g.
#     RUN_NAME=stage1_shared EPOCHS_STAGE2=0 sbatch scripts/train.sbatch
# then create the study once on the login node:
#     /scratch/gilbreth/jcantare/envs/fracture-sota/bin/python -m src.sweep --create \
#         --study-name tau_stage2_v1 \
#         --storage /scratch/gilbreth/jcantare/optuna/tau_stage2_v1.log
# then launch this array:
#     STAGE1_CKPT=OUTPUTS/stage1_shared/last.pt sbatch scripts/sweep.sbatch
#
# BUDGET GATE (A2/A3 — confirm BEFORE submitting the full array): check that
#   epochs_stage2 * epoch_time_s + eval_time < 4h
# against a REAL train_log.csv `epoch_time_s` from a Phase-2/3 run. With ASHA
# (min_resource=3, reduction_factor=3) most trials die at rung 0 (~3 epochs), so
# a single 4h worker clears many trials; gate the WORST case (a full ~15-epoch
# trial) against the cap. Run the Plan-05 2-trial %2-throttled flock smoke test
# first to confirm JournalFileBackend locking on scratch (RESEARCH A1).
#
# LEAKAGE RULE (D-04 / T-04-01): trials are scored ONLY on the trainer's logged
# val_f1_ar (the temporal-tail VAL split, the same scalar best.pt selects on).
# The 16 BASE cases are held-out TEST and never drive sampling/pruning. src/sweep.py
# ports the rank_variants leakage guard so any stray TEST-case eval dir raises.
#
# Submission is Plan 05, not this plan. This file just encodes the worker array.

#SBATCH --job-name=frac-tau-sweep
#SBATCH -A alta
# Partition must match the alta allocation (same as train.sbatch). The original
# 'standby' free-tier partition is not visible to this account on Gilbreth;
# a100-80gb already enforces the same 4h --time cap the budget trick relies on.
#SBATCH --partition=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-19%4          # throttle to 4 concurrent JournalFileBackend writers (RESEARCH Pitfall 1/A1)
#SBATCH --requeue
#SBATCH --signal=B:USR1@120     # SIGUSR1 to the batch shell 120s before walltime
#SBATCH --output=slurm-sweep-%A_%a.out

set -euo pipefail
export PYTHONUNBUFFERED=1       # epoch lines reach the slurm log immediately

# Call the env's python directly — immune to RCAC module-tree changes.
PYTHON=/scratch/gilbreth/jcantare/envs/fracture-sota/bin/python

cd "${SLURM_SUBMIT_DIR}"
nvidia-smi || true

# ---- required inputs (fail loud; no silent test-set fallback) -------------
: "${STAGE1_CKPT:?set STAGE1_CKPT to the SHARED frozen Stage-1 checkpoint (e.g. OUTPUTS/stage1_shared/last.pt) — every trial warm-starts it with --epochs-stage1 0}"

# ---- shared study config (all array tasks attach to the SAME study) -------
STUDY_NAME="${STUDY_NAME:-tau_stage2_v1}"
STORAGE="${STORAGE:-/scratch/gilbreth/jcantare/optuna/tau_stage2_v1.log}"
N_TRIALS="${N_TRIALS:-8}"       # trials this worker pulls before its 4h cap

# ---- SIGUSR1 self-requeue (RESEARCH Pitfall 3) ----------------------------
# Plain --requeue covers preemption/node-failure but NOT a clean TIMEOUT, so a
# timed-out worker would never resubmit. Trap the pre-walltime SIGUSR1 and
# self-requeue; per-epoch last.pt + `--resume auto` means an in-flight trial
# continues its OWN progress on the requeued task (resolve_seed_source: resume
# wins over init_from).
trap 'echo "[sweep] caught USR1 — requeueing job ${SLURM_JOB_ID}"; scontrol requeue "${SLURM_JOB_ID}"; exit 0' USR1

echo "=== sweep worker [array ${SLURM_ARRAY_TASK_ID}] on $(hostname) ==="
echo "  shared Stage-1 ckpt : ${STAGE1_CKPT}"
echo "  study / storage     : ${STUDY_NAME} @ ${STORAGE}"
echo "  trials this worker  : ${N_TRIALS}"
echo "  selection signal    : logged val_f1_ar (temporal-tail VAL, D-04)"

# ---- one Optuna worker attaching to the shared JournalFileBackend study ----
# Identical across all array tasks; ASHA + the shared study coordinate which
# trials each worker runs. The objective shells out to src.train per trial with
# the budget-trick flags and scores on val_f1_ar (never the 16 BASE TEST cases).
"$PYTHON" -m src.sweep --worker \
    --study-name "${STUDY_NAME}" \
    --storage "${STORAGE}" \
    --stage1-ckpt "${STAGE1_CKPT}" \
    --n-trials "${N_TRIALS}"

echo "=== sweep worker [array ${SLURM_ARRAY_TASK_ID}] complete ==="
# After the array drains, inspect the study / pick the winner (Plan 05/06):
#   "$PYTHON" -c "import optuna,sys; s=optuna.load_study(study_name='${STUDY_NAME}', \
#       storage=__import__('optuna').storages.JournalStorage( \
#       __import__('optuna').storages.journal.JournalFileBackend('${STORAGE}'))); \
#       print(s.best_trial.number, s.best_value, s.best_params)"
