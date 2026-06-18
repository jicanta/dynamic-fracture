# tests/test_sweep_objective.py
"""HPC-01 / D-04: the Optuna sweep objective is leakage-safe and plumbs the
sampled knobs onto the existing ``Config`` fields.

Pins the locked contract for ``new_model/src/sweep.py``:
  * D-04 (leakage rule) -- the objective MUST refuse to score on the 16 BASE
    held-out TEST cases. ``sweep._looks_like_test_set`` (ported verbatim from
    ``rank_variants.py``) flags any case list that overlaps the registry by >=8
    (the exact analog of ``test_rank_variants.py:190-198``).
  * knob plumbing -- ``sweep.build_cfg_from_tau_refined`` maps the 6 D-03 sweep
    knobs onto the real ``Config`` dataclass: ``tversky_beta`` direct,
    ``tversky_alpha = round(1-beta, 4)``, ``rollout_steps`` direct, and the
    budget-trick / fixed-winner holds (``epochs_stage1 == 0``, sigmoid head, TAU
    translator).

Framework rule: CPU-only. ``sweep.py`` imports optuna with a try/except-tolerant
guard (the S1 idiom used for the aggregate seam), so the leakage-guard and
knob-plumbing surface import and run WITHOUT optuna installed locally; only the
study/worker entrypoints need optuna (exercised on Gilbreth).

Run:  cd dynamic-fracture && python -m pytest tests/test_sweep_objective.py -x -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ---- repo-root sys.path shim (dynamic-fracture/) so new_model.src.* imports ----
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from new_model.src import sweep  # noqa: E402  (module under test)
from case_registry import TEST_CASE_FOLDERS  # noqa: E402


def test_refuses_test_set_leakage():
    # A case list that IS the 16 BASE TEST cases -> selection-time leakage (D-04).
    assert sweep._looks_like_test_set(list(TEST_CASE_FOLDERS.keys())) is True
    # also the folder-name spelling of the same cases must trip the guard.
    assert sweep._looks_like_test_set(list(TEST_CASE_FOLDERS.values())) is True


def test_allows_val_only_case_list():
    # A genuine (non-TEST) validation case list must NOT trip the guard.
    val_cases = ["val_run_a", "val_run_b", "val_run_c"]
    assert sweep._looks_like_test_set(val_cases) is False


def test_knob_plumbing():
    cfg = sweep.build_cfg_from_tau_refined(
        tversky_beta=0.72,
        feedback_noise_std=0.03,
        boundary_weight=0.4,
        growth_weight=5.0,
        lr_stage2=1e-4,
        rollout_steps=5,
    )
    # Tversky recall knob maps direct; alpha is the derived FP weight (1 - beta).
    assert cfg.tversky_beta == 0.72
    assert cfg.tversky_alpha == round(1.0 - 0.72, 4) == 0.28
    # budget trick: Stage-1 is FORCED off (every trial is a Stage-2-only fine-tune).
    assert cfg.epochs_stage1 == 0
    # the remaining sampled knobs land on their fields.
    assert cfg.rollout_steps == 5
    assert cfg.feedback_noise_std == 0.03
    assert cfg.boundary_weight == 0.4
    assert cfg.growth_weight == 5.0
    assert cfg.lr_stage2 == 1e-4
    # fixed winners held out of the search (Pitfall 10): sigmoid head, TAU translator.
    assert cfg.head_type == "sigmoid"
    assert cfg.translator_type == "tau"
    # the common recipe is pinned (D-03 fixed terms).
    assert cfg.tversky_weight == 1.0
    assert cfg.dice_weight == 0.0
    assert cfg.bce_weight == 1.0
    assert cfg.val_rollout_steps == 20


def test_exports_present():
    # the worker/study entrypoints the sbatch drives must exist as named exports.
    for name in ("objective", "build_cfg_from_tau_refined", "make_study",
                 "_looks_like_test_set"):
        assert hasattr(sweep, name)
