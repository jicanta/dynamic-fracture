# src/sweep.py
"""Optuna + ASHA Stage-2 hyperparameter sweep for FractureTAU (HPC-01, D-01..D-04).

Every trial is a *Stage-2-only* fine-tune from a SHARED frozen Stage-1 checkpoint
(the Phase-3 "budget trick", D-02): ``--init-from <stage1> --epochs-stage1 0``.
Trials are scored ONLY on the trainer's logged ``val_f1_ar`` -- the AR-rollout F1
on the temporal-tail VALIDATION split, the same scalar ``best.pt`` selects on.
The 16 BASE cases are held-out TEST and MUST NEVER drive sampling/pruning/ranking
(D-04); ``_looks_like_test_set`` (ported verbatim from ``scripts/rank_variants.py``)
refuses any eval dir that looks like them.

Storage is an offline ``JournalStorage(JournalFileBackend(...))`` on Gilbreth
scratch -- the only fully-offline, concurrent-writer-capable backend (SQLite is
not safe on a network FS; RDB needs a server). Many identical Slurm-array workers
attach to one shared study via ``load_if_exists=True``; ASHA
(``SuccessiveHalvingPruner``) kills weak trials early so dozens fit the 4h cap.

Run (one worker against the shared study; the sbatch fans this out as an array):
    python -m src.sweep --worker --study-name tau_stage2_v1 \\
        --storage /scratch/gilbreth/jcantare/optuna/tau_stage2_v1.log \\
        --stage1-ckpt OUTPUTS/stage1_shared/last.pt --n-trials 8
    # create the study once (login node):
    python -m src.sweep --create --study-name tau_stage2_v1 \\
        --storage /scratch/gilbreth/jcantare/optuna/tau_stage2_v1.log
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence

from .config import Config

# ---- repo-root sys.path shim (S1): src -> new_model -> dynamic-fracture ----
# Makes the framework-free case_registry (leakage guard) and the optional
# dashboard aggregate seam importable.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from case_registry import TEST_CASE_FOLDERS  # noqa: E402

# The ONE val-only selection seam (do NOT fork the metric, D-05). Import is
# deliberately try/except-tolerant: the default train-log scoring does NOT need
# the dashboard package, and on the cluster only new_model/ is synced (no
# results/), so a hard top-level import would break the worker.
try:
    from results.dashboard.aggregate import (  # noqa: E402
        late_rollout_selection_metric,
    )
except ModuleNotFoundError:
    late_rollout_selection_metric = None  # type: ignore

# Optuna is the sweep orchestrator (D-01). Imported tolerantly so the leakage
# guard + knob-plumbing surface (the unit-tested API) import WITHOUT optuna
# installed locally; only the study/worker entrypoints require it (Gilbreth env).
try:
    import optuna  # noqa: E402
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
except ModuleNotFoundError:
    optuna = None  # type: ignore
    JournalStorage = None  # type: ignore
    JournalFileBackend = None  # type: ignore

# >= half of the 16 held-out cases overlapping -> it is the TEST set (D-04).
_LEAKAGE_OVERLAP_THRESHOLD = 8
DEFAULT_LOG_SUBPATH = "logs/train_log.csv"
MAX_STAGE2_EPOCHS = 15  # ASHA max resource R (bounded in the objective, not the pruner)

# ---- tau_refined common recipe (D-03 FIXED terms; sampled knobs override) ----
# Constructed literally (NOT Config.load) so the objective is independent of any
# OUTPUTS/tau_refined/config.json being present on the running machine.
_TAU_REFINED_RECIPE: Dict[str, object] = {
    "extra": "none",
    # loss (fixed): recall-oriented focal-Tversky + BCE, no Dice.
    "tversky_weight": 1.0,
    "dice_weight": 0.0,
    "bce_weight": 1.0,
    "tversky_beta": 0.7,            # sampled in the objective (alpha derived)
    "tversky_gamma": 1.3333,
    "boundary_weight": 0.5,        # sampled
    "boundary_ramp": True,
    "growth_weight": 4.0,          # sampled
    # AR stabilization (fixed winners): pushforward ON, soft feedback noise.
    "ar_pushforward": 1,
    "feedback_noise_std": 0.05,    # sampled
    "rollout_steps": 4,            # sampled
    "val_rollout_steps": 20,       # Nyquist selection horizon (fixed)
    # optimization
    "lr_stage2": 1e-4,             # sampled (log-uniform)
    # architecture (fixed winners, NOT re-searched): sigmoid head, TAU translator.
    "head_type": "sigmoid",
    "translator_type": "tau",
    "seed": 42,                    # search uses seed 42; D-05 seeds come later
}


# ---- leakage guard (D-04, BINDING) — ported from rank_variants.py:132-141 ----
def _looks_like_test_set(case_names: Sequence[str]) -> bool:
    """True if the case names look like the 16 BASE held-out TEST cases (D-04).

    Matches against both the short keys and folder-name values of
    ``case_registry.TEST_CASE_FOLDERS`` so neither naming convention slips a
    TEST-set eval dir past the guard. Scoring the sweep on these would be
    selection-time leakage (T-04-01).
    """
    base = set(TEST_CASE_FOLDERS.keys()) | set(TEST_CASE_FOLDERS.values())
    return len(set(case_names) & base) >= _LEAKAGE_OVERLAP_THRESHOLD


def _guard_no_test_leakage(run_dir: Path) -> None:
    """Refuse if any eval-style subdir under ``run_dir`` holds the TEST cases.

    Defensive: the objective scores on the trainer's logged ``val_f1_ar`` and
    never reads an eval dir, but if a future edit points it at one this aborts
    loudly rather than leaking the held-out TEST set into selection (D-04).
    """
    for eval_dir in run_dir.glob("*eval*"):
        if not eval_dir.is_dir():
            continue
        case_names = [c.name for c in eval_dir.iterdir() if c.is_dir()]
        if _looks_like_test_set(case_names):
            raise ValueError(
                f"[sweep] refusing TEST-set leakage: {eval_dir} looks like the 16 "
                f"BASE held-out TEST cases — the sweep may score ONLY the "
                f"temporal-tail val_f1_ar (D-04 / T-04-01)."
            )


# ---- val-only selection signal: the trainer's logged val_f1_ar -------------
def _val_f1_ar_series(train_log: str | Path) -> List[float]:
    """All logged ``val_f1_ar`` values (per epoch) from a run's train_log.csv.

    ``val_f1_ar`` is the AR-rollout F1 on the temporal-tail VALIDATION split —
    the same scalar ``best.pt`` selects on (train.py:430-434). Leakage-free: it
    is computed on the trainer's val split, never the 16 BASE TEST cases.
    """
    path = Path(train_log)
    with open(path, newline="") as f:
        return [
            float(r["val_f1_ar"])
            for r in csv.DictReader(f)
            if r.get("val_f1_ar") not in (None, "")
        ]


# ---- Config construction — reuse the dataclass, do NOT fork knobs ----------
def build_cfg_from_tau_refined(**overrides) -> Config:
    """Build a ``Config`` = tau_refined recipe ⊕ sampled overrides (D-02/D-03).

    Applies overrides onto the fixed common recipe, derives
    ``tversky_alpha = round(1 - tversky_beta, 4)`` (the FP weight tracks the
    sampled recall knob), and FORCES ``epochs_stage1 = 0`` so every trial is a
    Stage-2-only fine-tune (the budget trick). Unknown keys are ignored (only
    real ``Config`` fields are set), mirroring ``Config.load``.
    """
    params: Dict[str, object] = dict(_TAU_REFINED_RECIPE)
    params.update(overrides)
    beta = float(params.get("tversky_beta", 0.7))
    params["tversky_alpha"] = round(1.0 - beta, 4)  # FP weight tracks recall knob
    params["epochs_stage1"] = 0                     # budget trick: Stage-2 only
    cfg = Config()
    for key, value in params.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


# ---- one Stage-2-only fine-tune step (shell out to the existing trainer) ----
def _train_to_epoch(cfg: Config, *, epochs_stage2: int, stage1_ckpt: str) -> None:
    """Run the Stage-2-only fine-tune up to ``epochs_stage2`` epochs.

    Shells out to ``src.train`` with the budget-trick flags (S4): ``--init-from``
    warm-starts the shared frozen Stage-1 weights with a FRESH schedule, and
    ``--resume auto`` makes each successive call continue the trial's OWN
    ``last.pt`` (so calling with epochs 1,2,...,K trains one more epoch each
    time, enabling per-epoch ASHA reporting). ``resolve_seed_source`` enforces
    that resume wins over init_from after the first epoch (config.py:181-187).
    """
    argv = [
        sys.executable, "-m", "src.train",
        "--data-root", cfg.data_root,
        "--out-root", cfg.out_root,
        "--run-name", cfg.run_name,
        "--extra", cfg.extra,
        "--init-from", stage1_ckpt,
        "--resume", "auto",
        "--epochs-stage1", "0",
        "--epochs-stage2", str(epochs_stage2),
        "--tversky-weight", str(cfg.tversky_weight),
        "--dice-weight", str(cfg.dice_weight),
        "--bce-weight", str(cfg.bce_weight),
        "--tversky-alpha", str(cfg.tversky_alpha),
        "--tversky-beta", str(cfg.tversky_beta),
        "--tversky-gamma", str(cfg.tversky_gamma),
        "--growth-weight", str(cfg.growth_weight),
        "--boundary-weight", str(cfg.boundary_weight),
        "--rollout-steps", str(cfg.rollout_steps),
        "--val-rollout-steps", str(cfg.val_rollout_steps),
        "--feedback-noise-std", str(cfg.feedback_noise_std),
        "--ar-pushforward", str(cfg.ar_pushforward),
        "--lr-stage2", str(cfg.lr_stage2),
        "--head-type", cfg.head_type,
        "--translator-type", cfg.translator_type,
        "--seed", str(cfg.seed),
    ]
    print(f"[sweep] train -> {cfg.run_name} (epochs_stage2={epochs_stage2})")
    subprocess.run(argv, check=True)


# ---- the Optuna objective: sample 6 knobs, score val_f1_ar, ASHA-prune ------
# Worker config (study-independent) is stashed at module scope by ``main`` so the
# bare ``objective`` callable matches the optuna.study.optimize(objective) API.
_STAGE1_CKPT: str = "none"
_MAX_EPOCHS: int = MAX_STAGE2_EPOCHS


def objective(trial) -> float:
    """One trial = one Stage-2-only fine-tune scored on val_f1_ar (D-04).

    Samples the 6 D-03 knobs, runs the fine-tune epoch-by-epoch (so ASHA can
    prune after rung 0 = 3 epochs), reports the per-epoch ``val_f1_ar`` (the
    temporal-tail VAL signal — NEVER the 16 BASE TEST cases), and returns the
    best. Raises ``optuna.TrialPruned`` when ASHA kills the trial.
    """
    if optuna is None:
        raise SystemExit("[sweep] optuna is not installed — `pip install \"optuna>=4.0\"`")

    # the 6 D-03 sweep knobs (ranges signed off in the plan).
    beta = trial.suggest_float("tversky_beta", 0.60, 0.80)
    cfg = build_cfg_from_tau_refined(
        feedback_noise_std=trial.suggest_float("feedback_noise_std", 0.0, 0.10),
        tversky_beta=beta,
        boundary_weight=trial.suggest_float("boundary_weight", 0.0, 1.0),
        growth_weight=trial.suggest_float("growth_weight", 2.0, 6.0),
        lr_stage2=trial.suggest_float("lr_stage2", 3e-5, 3e-4, log=True),
        rollout_steps=trial.suggest_categorical("rollout_steps", [3, 4, 5, 6]),
        run_name=f"sweep/trial_{trial.number}",
    )

    run_dir = Path(cfg.out_root) / cfg.run_name
    best = float("-inf")
    for epoch in range(1, _MAX_EPOCHS + 1):
        _train_to_epoch(cfg, epochs_stage2=epoch, stage1_ckpt=_STAGE1_CKPT)
        # D-04: read ONLY the trainer's logged val_f1_ar; never src.evaluate on
        # the 16 BASE TEST cases. Guard any stray eval dir defensively.
        _guard_no_test_leakage(run_dir)
        series = _val_f1_ar_series(run_dir / DEFAULT_LOG_SUBPATH)
        if not series:
            raise optuna.TrialPruned()  # crashed before its first val epoch
        val_f1_ar = series[-1]
        best = max(best, val_f1_ar)
        trial.report(val_f1_ar, step=epoch)
        if trial.should_prune():
            print(f"[sweep] trial {trial.number} pruned at epoch {epoch} "
                  f"(val_f1_ar={val_f1_ar:.4f})")
            raise optuna.TrialPruned()
    print(f"[sweep] trial {trial.number} done: best val_f1_ar={best:.4f}")
    return best


# ---- study factory: offline JournalFileBackend + TPE + ASHA ----------------
def make_study(study_name: str, storage_path: str):
    """Create/attach the shared offline study (D-01, RESEARCH Pattern 2).

    ``JournalStorage(JournalFileBackend(...))`` is the only fully-offline,
    concurrent-writer backend (SQLite corrupts on network FS; RDB needs a
    server). ``load_if_exists=True`` lets every Slurm-array worker attach to the
    SAME study. ASHA prunes weak trials (rungs ~3/9/15) to fit the 4h cap.
    """
    if optuna is None:
        raise SystemExit("[sweep] optuna is not installed — `pip install \"optuna>=4.0\"`")
    storage = JournalStorage(JournalFileBackend(storage_path))
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",                       # maximize val_f1_ar
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.SuccessiveHalvingPruner(
            min_resource=3, reduction_factor=3, min_early_stopping_rate=0),
        load_if_exists=True,                         # workers share one study
    )


# ---- CLI -------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> None:
    global _STAGE1_CKPT, _MAX_EPOCHS
    ap = argparse.ArgumentParser(
        description="Optuna + ASHA Stage-2 sweep for FractureTAU (HPC-01, D-01..D-04)."
    )
    ap.add_argument("--worker", action="store_true",
                    help="attach to the shared study and optimize --n-trials trials")
    ap.add_argument("--create", action="store_true",
                    help="just create/attach the study (run once on the login node)")
    ap.add_argument("--study-name", default="tau_stage2_v1")
    ap.add_argument("--storage", required=True,
                    help="JournalFileBackend log path on shared scratch")
    ap.add_argument("--stage1-ckpt", default="none",
                    help="SHARED frozen Stage-1 checkpoint (--init-from) — required for --worker")
    ap.add_argument("--n-trials", type=int, default=8,
                    help="trials this worker pulls before its 4h cap")
    ap.add_argument("--max-epochs", type=int, default=MAX_STAGE2_EPOCHS,
                    help="ASHA max resource R (Stage-2 epochs per trial)")
    args = ap.parse_args(argv)

    if optuna is None:
        raise SystemExit("[sweep] optuna is not installed — `pip install \"optuna>=4.0\"`")
    if not (args.worker or args.create):
        raise SystemExit("[sweep] choose a mode: --worker or --create")

    if args.create:
        make_study(args.study_name, args.storage)
        print(f"[sweep] study '{args.study_name}' ready at {args.storage}")
        return

    # --worker
    if args.stage1_ckpt in ("none", "") or not Path(args.stage1_ckpt).is_file():
        raise SystemExit(
            f"[sweep] --stage1-ckpt must point at the SHARED frozen Stage-1 "
            f"checkpoint (got '{args.stage1_ckpt}'); the budget trick warm-starts it.")
    _STAGE1_CKPT = args.stage1_ckpt
    _MAX_EPOCHS = args.max_epochs
    study = make_study(args.study_name, args.storage)
    print(f"[sweep] worker attached to '{args.study_name}', pulling {args.n_trials} trials")
    study.optimize(objective, n_trials=args.n_trials)


if __name__ == "__main__":
    main()
