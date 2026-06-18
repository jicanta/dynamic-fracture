#!/bin/bash
# Leave-one-out ablation for Phase 4 (HPC-04 / D-07). Isolates the components of
# the winning `tau_refined` recipe by toggling ONE knob per row off the shared
# baseline, scored by the late-rollout macro F1 drop on the 16 BASE cases.
#
# THE BUDGET TRICK (D-02, same as sweep_stage2.sbatch): the TAU Stage-1 is
# trained ONCE, frozen, and rows 2-4 run as Stage-2-only fine-tunes from that
# SHARED init (~15 Stage-2 epochs each) so they fit the 4h cap and stay
# apples-to-apples. SINGLE seed=42 per D-08 (only the headline gets 3 seeds).
#
# THE 5 D-07 ROWS (RESEARCH Pattern 6):
#   row 1  AR fine-tune OFF      = evaluate the frozen Stage-1 ITSELF (no training)   [budget-trick: it IS the init]
#   row 2  boundary loss OFF     = --boundary-weight 0.0                              [Stage-2-only fine-tune]
#   row 3  AR-stabilization OFF  = --ar-pushforward 0 --feedback-noise-std 0.0        [Stage-2-only] (= ar_ctrl_ss_only)
#   row 4  head monotone-Δ       = --head-type monotone_delta                         [Stage-2-only, warm-start valid]
#   row 5  plain translator      = --translator-type plain                            [NOT budget-trick: own full Stage-1+Stage-2]
#
# Rows 2-4 are this array (--array=0-2). Rows 1 and 5 are documented at the
# bottom: row 1 is a bare evaluate of the frozen Stage-1 (no train); row 5 has
# DIFFERENT params (plain translator) and CANNOT --init-from a TAU Stage-1
# (shape mismatch, RESEARCH Pitfall 5 / T-04-11) — it needs its OWN full train.
#
# STEP 0 (ONCE, separately): produce/point at the SHARED frozen TAU Stage-1 —
# the SAME checkpoint the sweep used (provenance: its best.pt SHA). Then:
#   STAGE1_CKPT=OUTPUTS/stage1_shared/last.pt sbatch scripts/ablation.sbatch
#
# Submission is Plan 07, not this plan. This file encodes the row->delta map.

#SBATCH --job-name=frac-tau-ablation
#SBATCH -A alta
# Partition must match the alta allocation (same as sweep_stage2.sbatch). The
# 'standby' free-tier partition is not visible to this account on Gilbreth;
# a100-80gb already enforces the same 4h --time cap the budget trick relies on.
#SBATCH --partition=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm-abl-%A_%a.out

set -euo pipefail
export PYTHONUNBUFFERED=1   # epoch lines reach the slurm log immediately

# ---- self-requeue on TIMEOUT (HPC-05) -------------------------------------
# Plain --requeue covers node FAILURE; a clean TIME-LIMIT hit needs an explicit
# `scontrol requeue` on the SIGUSR1 Slurm sends 120s before the wall clock. The
# requeued task resumes its OWN last.pt (resolve_seed_source: resume beats
# init_from), so no Stage-2 epochs are lost.
_requeue() {
    echo "[ablation] caught SIGUSR1 (approaching time limit) -> requeueing ${SLURM_JOB_ID}"
    scontrol requeue "${SLURM_JOB_ID}" || true
    exit 0
}
trap _requeue USR1

# Call the env's python directly — immune to RCAC module-tree changes.
PYTHON=/scratch/gilbreth/jcantare/envs/fracture-sota/bin/python

cd "${SLURM_SUBMIT_DIR}"

# ---- required inputs (fail loud; no silent default init) ------------------
# The SHARED frozen TAU Stage-1 the budget-trick rows 2-4 warm-start from. No
# default -> a missing STAGE1_CKPT aborts before any run drifts onto a stray
# init (T-04-12). MUST be the SAME Stage-1 the sweep used (apples-to-apples).
: "${STAGE1_CKPT:?set STAGE1_CKPT to the SHARED frozen Stage-1 checkpoint (e.g. OUTPUTS/stage1_shared/last.pt) — rows 2-4 warm-start from it with --epochs-stage1 0}"

EPOCHS_STAGE2="${EPOCHS_STAGE2:-15}"   # Stage-2-only fine-tune length (budget gate)

# ---- COMMON: the winning tau_refined recipe (refined.sbatch:40-45) --------
# Every ablation row holds this fixed and toggles exactly ONE knob (leave-one-out).
COMMON="--tversky-weight 1.0 --dice-weight 0.0 --bce-weight 1.0 \
--growth-weight 4.0 --tversky-gamma 1.3333 \
--tversky-alpha 0.3 --tversky-beta 0.7 --boundary-weight 0.5 \
--ar-pushforward 1 --feedback-noise-std 0.05 \
--rollout-steps 4 --val-rollout-steps 20"

# ---- D-07 budget-trick rows (2-4): Stage-2-only fine-tunes ----------------
# Each entry: "run_name|--flags-that-OVERRIDE-the-baseline". The trailing flag
# wins, so e.g. row 2's --boundary-weight 0.0 overrides COMMON's 0.5.
VARIANTS=(
  "abl_noboundary|--boundary-weight 0.0"                          # row 2: boundary loss OFF
  "abl_no_arstab|--ar-pushforward 0 --feedback-noise-std 0.0"     # row 3: AR-stabilization OFF (= ar_ctrl_ss_only)
  "abl_head_monotone|--head-type monotone_delta"                  # row 4: monotone-Δ head
)

ENTRY="${VARIANTS[${SLURM_ARRAY_TASK_ID}]}"
VARIANT="${ENTRY%%|*}"
FLAGS="${ENTRY#*|}"

echo "=== ablation row [${SLURM_ARRAY_TASK_ID}] ${VARIANT} on $(hostname) ==="
echo "  shared Stage-1 ckpt : ${STAGE1_CKPT}"
echo "  Stage-2 epochs      : ${EPOCHS_STAGE2}  (K=4, val_rollout_steps=20, seed=42)"
echo "  leave-one-out delta : ${FLAGS}"
nvidia-smi || true

# ---- 1) Stage-2-only fine-tune from the SHARED frozen Stage-1 (budget trick) --
# --init-from warm-starts the shared TAU Stage-1 WEIGHTS with a FRESH epoch
# schedule (so --epochs-stage1 0 => only the Stage-2 epochs run); --resume auto
# makes a preempted/requeued task continue its OWN last.pt instead of restarting.
"$PYTHON" -m src.train \
    --data-root DATASET \
    --out-root OUTPUTS \
    --run-name "ablation/${VARIANT}" \
    --extra none \
    --seed 42 \
    --init-from "${STAGE1_CKPT}" \
    --resume auto \
    --epochs-stage1 0 \
    ${COMMON} ${FLAGS} \
    --epochs-stage2 "${EPOCHS_STAGE2}"

# ---- 2) Evaluate the fine-tuned row on the 16 BASE cases -------------------
# Produces OUTPUTS/ablation/<variant>/eval/.../per_frame_metrics.csv + the
# row's best.pt (whose SHA is the per-row provenance, captured in plan 07).
"$PYTHON" -m src.evaluate \
    --data-root DATASET \
    --out-root OUTPUTS \
    --run-name "ablation/${VARIANT}"

echo "=== ablation row ${VARIANT} complete -> OUTPUTS/ablation/${VARIANT}/ (eval/.../per_frame_metrics.csv + best.pt SHA) ==="

# ===========================================================================
# ROW 1 (AR fine-tune OFF): NO training — evaluate the frozen Stage-1 ITSELF.
#   The "ablation" of the AR fine-tune is simply the Stage-1 checkpoint before
#   any Stage 2, so there is nothing to train. Evaluate it directly:
#     "$PYTHON" -m src.evaluate --data-root DATASET --out-root OUTPUTS \
#         --run-name <frozen_stage1_run>
#   Its best.pt SHA MUST equal the STAGE1_CKPT SHA above (same init = provenance).
#
# ROW 5 (plain translator): NOT budget-trick compatible. The plain translator's
#   block params differ in shape from the TAU Stage-1, so --init-from would
#   shape-mismatch (RESEARCH Pitfall 5 / T-04-11). Submit a SEPARATE full
#   Stage-1+Stage-2 job (refined.sbatch-style, --time=12:00:00, NO --init-from)
#   with --translator-type plain, e.g.:
#     "$PYTHON" -m src.train --data-root DATASET --out-root OUTPUTS \
#         --run-name ablation/abl_plain_translator --extra none --seed 42 \
#         --translator-type plain \
#         ${COMMON} --epochs-stage1 60 --epochs-stage2 15 --resume auto
#     "$PYTHON" -m src.evaluate --run-name ablation/abl_plain_translator
#   See plan 07; gate on the row-5 compute decision (a full train, not 15 epochs).
# ===========================================================================
