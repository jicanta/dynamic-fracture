#!/bin/bash
# Directional Stage-2 sweep for Phase 3 (D-02/D-03 APPROVED grid).
#
# THE BUDGET TRICK (D-02, from 03-RESEARCH "Ablation Protocol & Compute Budget"):
# train Stage 1 (teacher-forced) ONCE, freeze it, and run every directional
# variant as a Stage-2-only fine-tune from that SHARED init. Each variant is then
# ~15 Stage-2 epochs (NOT a full Stage-1+Stage-2 train), so it fits the 4h
# `standby` cap on one A100-80GB and the variants stay apples-to-apples.
#
# STEP 0 (ONCE, separately — DO NOT do this per array task):
#   produce the shared frozen Stage-1 checkpoint. Reuse train.sbatch with
#   Stage 2 disabled, e.g.:
#     RUN_NAME=stage1_shared EPOCHS_STAGE2=0 sbatch scripts/train.sbatch
#   (or any Phase-2 run's end-of-Stage-1 checkpoint). Then point this array at it:
#     STAGE1_CKPT=OUTPUTS/stage1_shared/last.pt \
#     VAL_CASES="val_run_a val_run_b ..." sbatch scripts/sweep_stage2.sbatch
#
# BUDGET GATE (A2 — confirm BEFORE submitting): check that
#   epochs_stage2 * epoch_time_s + eval_time < 4h
# against a REAL `train_log.csv` `epoch_time_s` from a Phase-2 run. If it does not
# fit, drop EPOCHS_STAGE2 (e.g. 12) or shrink VAL_CASES before submitting.
#
# LEAKAGE RULE (D-06 / T-03-13): VAL_CASES MUST be the VALIDATION split only.
# The 16 BASE cases are the held-out TEST set — ranking on them inflates the win.
# This script refuses to run if VAL_CASES is unset, so it can never silently fall
# back to the test set. rank_variants.py applies the same guard at ranking time.
#
# Submission is Plan 06, not this plan. This file just encodes the approved grid.

#SBATCH --job-name=frac-tau-sweep2
#SBATCH -A alta
#SBATCH --partition=standby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-9
#SBATCH --output=slurm-sweep2-%A_%a.out

set -euo pipefail
export PYTHONUNBUFFERED=1   # epoch lines reach the slurm log immediately

# Call the env's python directly — immune to RCAC module-tree changes.
PYTHON=/scratch/gilbreth/jcantare/envs/fracture-sota/bin/python

cd "${SLURM_SUBMIT_DIR}"

# ---- required inputs (fail loud; no silent test-set fallback) -------------
: "${STAGE1_CKPT:?set STAGE1_CKPT to the SHARED frozen Stage-1 checkpoint (e.g. OUTPUTS/stage1_shared/last.pt) — the budget trick resumes from it with --epochs-stage1 0}"
: "${VAL_CASES:?set VAL_CASES to the VALIDATION-split cases ONLY — NEVER the 16 BASE TEST cases (leakage, D-06 / T-03-13)}"

EXTRA="${EXTRA:-none}"                 # none | pressure | vonmises | SED
EPOCHS_STAGE2="${EPOCHS_STAGE2:-15}"   # Stage-2-only fine-tune length (A2 budget gate)

# ---- APPROVED common loss recipe (D-03, signed off "approved as proposed") ----
# tversky_weight=1.0, dice_weight=0.0, bce_weight=1.0, growth_weight~3-5 (=4),
# tversky_gamma=4/3 (~1.3333); AR K fixed at 4; Nyquist selection horizon=20.
COMMON="--tversky-weight 1.0 --dice-weight 0.0 --bce-weight 1.0 \
--growth-weight 4.0 --tversky-gamma 1.3333 \
--rollout-steps 4 --val-rollout-steps 20 \
--epochs-stage2 ${EPOCHS_STAGE2}"

# ---- APPROVED ~10-variant coarse grid (single-seed, directional) ---------
# Tversky (alpha=FP, beta=FN/recall): P0 0.5/0.5, P1 0.4/0.6, P2 0.3/0.7,
#   P3 0.25/0.75 (diagnostic only — FP-lock stress probe, not meant to win).
# Boundary: torch-only morphological band, boundary_weight=0.5, boundary_ramp
#   default True (linear 0->1 across Stage 1, held 1.0 in Stage 2); one control
#   with boundary_weight=0 isolates the term.
# AR (MODEL-02): K=4; {pushforward, feedback-noise, both} + SS-only control;
#   feedback_noise_std=0.05 (soft-prob Gaussian, pre-threshold, no-healing-safe).
# Head: monotone_delta on the best loss+AR combo.
# NOTE: "best alpha/beta / best loss+AR" for the control + head variants is a
# placeholder (P2 0.3/0.7, pushforward+noise) confirmed/re-pointed in Plan 06
# AFTER the loss runs rank — a single static array cannot resolve "best" a priori.
VARIANTS=(
  "loss_P0|--tversky-alpha 0.5 --tversky-beta 0.5 --boundary-weight 0.5"
  "loss_P1|--tversky-alpha 0.4 --tversky-beta 0.6 --boundary-weight 0.5"
  "loss_P2|--tversky-alpha 0.3 --tversky-beta 0.7 --boundary-weight 0.5"
  "loss_P3_diag|--tversky-alpha 0.25 --tversky-beta 0.75 --boundary-weight 0.5"
  "loss_ctrl_noboundary|--tversky-alpha 0.3 --tversky-beta 0.7 --boundary-weight 0.0"
  "ar_pushforward|--tversky-alpha 0.3 --tversky-beta 0.7 --boundary-weight 0.5 --ar-pushforward 1"
  "ar_feedbacknoise|--tversky-alpha 0.3 --tversky-beta 0.7 --boundary-weight 0.5 --feedback-noise-std 0.05"
  "ar_both|--tversky-alpha 0.3 --tversky-beta 0.7 --boundary-weight 0.5 --ar-pushforward 1 --feedback-noise-std 0.05"
  "ar_ctrl_ss_only|--tversky-alpha 0.3 --tversky-beta 0.7 --boundary-weight 0.5 --ar-pushforward 0 --feedback-noise-std 0.0"
  "head_monotone|--tversky-alpha 0.3 --tversky-beta 0.7 --boundary-weight 0.5 --ar-pushforward 1 --feedback-noise-std 0.05 --head-type monotone_delta"
)

ENTRY="${VARIANTS[${SLURM_ARRAY_TASK_ID}]}"
VARIANT="${ENTRY%%|*}"
FLAGS="${ENTRY#*|}"

echo "=== sweep variant [${SLURM_ARRAY_TASK_ID}] ${VARIANT} on $(hostname) ==="
echo "  shared Stage-1 ckpt : ${STAGE1_CKPT}"
echo "  Stage-2 epochs      : ${EPOCHS_STAGE2}  (K=4, val_rollout_steps=20)"
echo "  variant flags       : ${FLAGS}"
echo "  VAL cases (no leak) : ${VAL_CASES}"
nvidia-smi || true

# ---- 1) Stage-2-only fine-tune from the SHARED frozen Stage-1 (budget trick) --
# --resume the shared checkpoint and --epochs-stage1 0 so ONLY Stage 2 runs;
# do NOT retrain Stage 1 per variant.
"$PYTHON" -m src.train \
    --data-root DATASET \
    --out-root OUTPUTS \
    --run-name "sweep/${VARIANT}" \
    --extra "${EXTRA}" \
    --resume "${STAGE1_CKPT}" \
    --epochs-stage1 0 \
    ${COMMON} \
    ${FLAGS}

# ---- 2) Evaluate on the VAL split into val_eval/ (val-only; D-06) ----------
# rank_variants.py reads OUTPUTS/sweep/<variant>/val_eval/<case>/per_frame_metrics.csv
echo "Stage-2 fine-tune done. Evaluating variant ${VARIANT} on the VAL split..."
"$PYTHON" -m src.evaluate \
    --data-root DATASET \
    --out-root OUTPUTS \
    --run-name "sweep/${VARIANT}" \
    --eval-dir-name val_eval \
    --cases ${VAL_CASES}

echo "=== variant ${VARIANT} complete -> OUTPUTS/sweep/${VARIANT}/val_eval/ ==="
# After all array tasks finish, rank on the val-only seam (Plan 06):
#   python scripts/rank_variants.py --sweep-root OUTPUTS/sweep
