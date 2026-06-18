# tests/test_resume_lr.py
"""HPC-05 / D-10: requeue/resume LR-determinism + resume-precedence.

Pins two locked invariants that make a `--requeue`d Stage-2 fine-tune resume
correctly under the 4h `a100-80gb` cap:

  * D-10 (LR determinism) -- ``utils.cosine_warmup_lr(step, ...)`` is a PURE
    function of ``step`` (no persisted scheduler object). On resume,
    ``train.py:360-361`` reconstructs the global step at epoch granularity:
        global_step_s2 = max(len(dl_s2), 1) * max(start_epoch - epochs_stage1, 0)
    and re-runs the in-progress epoch from its start. Because ``len(dl_s2)`` is
    deterministic (drop_last=True, fixed dataset/batch) the per-step LR sequence
    from ``start_epoch`` onward is BIT-IDENTICAL to an uninterrupted run. We
    assert that with exact ``==`` (not ``approx``) -- it is a pure function, so
    bit-identity is the honest claim (NOT weight bit-identity, which AMP/cuDNN
    break, Pitfall 9).

  * resume precedence -- ``config.resolve_seed_source`` makes ``resume`` win over
    ``init_from`` so a requeued sweep/ablation task continues its OWN ``last.pt``
    instead of restarting from the shared frozen Stage-1 init.

Framework rule: CPU-only, torch-optional (a minimal stub lets ``utils.py`` import
without a real torch install; ``cosine_warmup_lr`` is pure math + stdlib).

Run:  cd dynamic-fracture && python -m pytest tests/test_resume_lr.py -x -q
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

# ---- ensure utils.py / config.py import (with or without real torch) ----
# utils.py imports torch at module top and uses @torch.no_grad() at class-def
# time; inject a minimal stub when real torch is absent (idiom mirrored from
# tests/test_determinism.py:19-40).
try:
    import torch as _real_torch  # noqa: F401
except Exception:
    if "torch" not in sys.modules:
        _stub = types.ModuleType("torch")

        def _no_grad():
            def _deco(fn):
                return fn
            return _deco

        _stub.no_grad = _no_grad
        sys.modules["torch"] = _stub

SRC = Path(__file__).resolve().parents[1] / "new_model" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import config  # noqa: E402  (resolve_seed_source under test)
import utils   # noqa: E402  (cosine_warmup_lr under test)


# ---- D-10: pure-function LR identity across an uninterrupted vs resumed run ----
def test_lr_pure_function_identity():
    # A concrete Stage-2-only schedule (epochs_stage1=0 => global_step counts
    # Stage-2 steps directly). len(dl_s2) is fixed and deterministic.
    len_dl_s2 = 7
    epochs_stage1 = 0
    total_epochs = 15
    warmup_epochs = 3
    base_lr = 1e-4

    total_steps = len_dl_s2 * total_epochs
    warmup_steps = len_dl_s2 * warmup_epochs

    # uninterrupted run: the full per-step LR sequence.
    uninterrupted = [
        utils.cosine_warmup_lr(step, total_steps, warmup_steps, base_lr)
        for step in range(total_steps)
    ]

    # resumed-at-epoch-k run: train.py reconstructs the starting global step at
    # epoch granularity, then re-runs from there to the end.
    for start_epoch in range(0, total_epochs):
        global_step_s2 = max(len_dl_s2, 1) * max(start_epoch - epochs_stage1, 0)
        resumed = [
            utils.cosine_warmup_lr(step, total_steps, warmup_steps, base_lr)
            for step in range(global_step_s2, total_steps)
        ]
        # bit-identical to the corresponding tail of the uninterrupted run.
        assert resumed == uninterrupted[global_step_s2:]
        # spot-check the boundary step itself is exactly equal (pure function).
        assert (utils.cosine_warmup_lr(global_step_s2, total_steps, warmup_steps, base_lr)
                == uninterrupted[global_step_s2])


def test_lr_no_persisted_scheduler_state():
    """Guardrail (Anti-Pattern): resume must NOT persist a scheduler object.

    The whole point of D-10 is that LR is recomputed from the epoch. If a future
    edit serializes a torch scheduler into the checkpoint, this source-grep trips.
    """
    train_src = (SRC / "train.py").read_text()
    assert "scheduler.state_dict()" not in train_src
    assert "lr_scheduler.state_dict()" not in train_src


# ---- resume precedence: resume wins over init_from ----
def test_resume_precedence():
    # requeued task with its OWN last.pt -> continue it (resume), NOT the shared
    # frozen Stage-1 init.
    assert config.resolve_seed_source(
        resume="auto", init_from="x.pt", last_ckpt_exists=True) == "resume"
    # first launch (no last.pt yet) -> warm-start from init_from (the budget
    # trick actually runs its Stage-2 epochs).
    assert config.resolve_seed_source(
        resume="auto", init_from="x.pt", last_ckpt_exists=False) == "init"
    # explicit --resume /path always wins.
    assert config.resolve_seed_source(
        resume="/some/last.pt", init_from="x.pt", last_ckpt_exists=False) == "resume"
    # nothing to seed from -> fresh.
    assert config.resolve_seed_source(
        resume="none", init_from="none", last_ckpt_exists=False) == "fresh"
