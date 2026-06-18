# tests/test_ablation_config.py
"""HPC-04 / D-07: the 5-row leave-one-out ablation row -> Config/build_model map.

The ablation (ablation.sbatch) isolates the components of the winning
``tau_refined`` recipe by toggling ONE knob per row off the shared baseline:

    row 1  AR fine-tune OFF      = no Stage 2 (epochs_stage2 == 0; the frozen Stage-1 itself)
    row 2  boundary loss OFF     = boundary_weight == 0.0
    row 3  AR-stabilization OFF  = ar_pushforward == 0 and feedback_noise_std == 0.0
    row 4  head monotone-Δ       = head_type == "monotone_delta"
    row 5  plain translator      = translator_type == "plain"

Rows 1-4 are budget-trick compatible (Stage-2-only fine-tunes that warm-start
from the SAME frozen TAU Stage-1). Row 5 changes a STRUCTURAL parameter (plain
vs TAU translator blocks), so its weights are shape-incompatible with a TAU
``--init-from`` — it MUST be a separate full train (plan 07). This test pins
that incompatibility structurally: a ``translator_type="plain"`` build produces
a model whose translator is a ``PlainTranslator`` distinct from the default TAU
build (so a silent warm-start no-op, threat T-04-11, cannot pass unnoticed).

Rows 1-4 are pure ``Config`` deltas (no torch). Row 5 asserts the actual
``build_model`` structure, so it guards on real torch using the same idiom as
``test_monotone_head.py`` (torch is Gilbreth-only on the workstation; a stub
``torch`` may be present in ``sys.modules`` from ``test_determinism.py``, so we
import a real submodule to detect it).

Run:  cd dynamic-fracture && python -m pytest tests/test_ablation_config.py -q
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

# ---- real-torch guard (mirrors test_monotone_head.py) ----
try:
    import torch as _real_torch  # noqa: F401
    import torch.nn as _real_torch_nn  # noqa: F401
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# ---- sys.path shim to reach new_model/src ----
SRC = Path(__file__).resolve().parents[1] / "new_model" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import Config  # noqa: E402  (no torch import in config.py)


def baseline_cfg() -> Config:
    """The winning ``tau_refined`` recipe (refined.sbatch:34-46) as a Config.

    Built LITERALLY (not Config.load) so the ablation row->delta map is pinned
    independently of any OUTPUTS/tau_refined/config.json existing — same idiom
    as build_cfg_from_tau_refined in src/sweep.py (plan 04-02).
    """
    return Config(
        run_name="tau_refined",
        extra="none",
        seed=42,
        tversky_weight=1.0,
        dice_weight=0.0,
        bce_weight=1.0,
        growth_weight=4.0,
        tversky_gamma=1.3333,
        tversky_alpha=0.3,
        tversky_beta=0.7,
        boundary_weight=0.5,
        ar_pushforward=1,
        feedback_noise_std=0.05,
        rollout_steps=4,
        val_rollout_steps=20,
        epochs_stage1=60,
        epochs_stage2=15,
        head_type="sigmoid",
        translator_type="tau",
    )


# ---- row -> delta map (D-07). Rows 1-4 are pure Config overrides. ----
# Row 1 is expressed as "no Stage 2" rather than a loss/model knob.
ROW_DELTAS = {
    1: {"epochs_stage2": 0},                                   # AR fine-tune OFF
    2: {"boundary_weight": 0.0},                               # boundary loss OFF
    3: {"ar_pushforward": 0, "feedback_noise_std": 0.0},       # AR-stabilization OFF
    4: {"head_type": "monotone_delta"},                        # monotone-Δ head
    5: {"translator_type": "plain"},                           # plain translator
}


def apply_row(row: int) -> Config:
    """Apply a single D-07 row's delta to the tau_refined baseline."""
    return replace(baseline_cfg(), **ROW_DELTAS[row])


# ===================================================================
# Rows 1-4: pure Config deltas — exactly one knob differs from baseline.
# ===================================================================
def test_row1_ar_finetune_off_is_no_stage2():
    base = baseline_cfg()
    cfg = apply_row(1)
    # Row 1 is "evaluate the frozen Stage-1 itself" => zero Stage-2 epochs.
    assert cfg.epochs_stage2 == 0
    assert base.epochs_stage2 == 15            # baseline does fine-tune
    # NOT a loss/model delta: every other knob equals the baseline.
    assert cfg.boundary_weight == base.boundary_weight
    assert cfg.ar_pushforward == base.ar_pushforward
    assert cfg.feedback_noise_std == base.feedback_noise_std
    assert cfg.head_type == base.head_type
    assert cfg.translator_type == base.translator_type


def test_row2_boundary_loss_off():
    base = baseline_cfg()
    cfg = apply_row(2)
    assert cfg.boundary_weight == 0.0
    assert base.boundary_weight == 0.5         # baseline has boundary ON
    # Tversky recipe and everything else unchanged (apples-to-apples).
    assert cfg.tversky_alpha == 0.3 and cfg.tversky_beta == 0.7
    assert cfg.ar_pushforward == base.ar_pushforward
    assert cfg.head_type == base.head_type
    assert cfg.translator_type == base.translator_type


def test_row3_ar_stabilization_off():
    base = baseline_cfg()
    cfg = apply_row(3)
    # = ar_ctrl_ss_only: pushforward + feedback-noise both off, SS only.
    assert cfg.ar_pushforward == 0
    assert cfg.feedback_noise_std == 0.0
    assert base.ar_pushforward == 1 and base.feedback_noise_std == 0.05
    assert cfg.boundary_weight == base.boundary_weight
    assert cfg.head_type == base.head_type
    assert cfg.translator_type == base.translator_type


def test_row4_head_monotone_delta():
    base = baseline_cfg()
    cfg = apply_row(4)
    assert cfg.head_type == "monotone_delta"
    assert base.head_type == "sigmoid"
    assert cfg.boundary_weight == base.boundary_weight
    assert cfg.ar_pushforward == base.ar_pushforward
    assert cfg.translator_type == base.translator_type


def test_row5_translator_type_plain():
    base = baseline_cfg()
    cfg = apply_row(5)
    assert cfg.translator_type == "plain"
    assert base.translator_type == "tau"
    # Pure leave-one-out: nothing else moves.
    assert cfg.boundary_weight == base.boundary_weight
    assert cfg.ar_pushforward == base.ar_pushforward
    assert cfg.head_type == base.head_type
    assert cfg.epochs_stage2 == base.epochs_stage2


# ===================================================================
# Row 5 structural check: plain build differs from TAU build (T-04-11).
# Requires real torch (model.py imports torch at module top).
# ===================================================================
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="real torch not installed")
def test_row5_builds_plain_translator_distinct_from_tau():
    from model import build_model

    tau_model = build_model(baseline_cfg(), c_in=5)
    plain_model = build_model(apply_row(5), c_in=5)

    # The plain build's translator is a PlainTranslator; the TAU build's is not.
    assert type(plain_model.translator).__name__ == "PlainTranslator"
    assert type(tau_model.translator).__name__ == "Translator"
    assert type(plain_model.translator).__name__ != type(tau_model.translator).__name__
    assert plain_model.translator_type == "plain"
    assert tau_model.translator_type == "tau"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="real torch not installed")
def test_row5_warmstart_incompatible_param_shapes():
    """Row 5 cannot --init-from a TAU Stage-1: the translator block params have
    DIFFERENT shapes, so a warm-start would shape-mismatch (no silent no-op).
    Compare the translator sub-state-dicts of the two builds."""
    from model import build_model

    tau_t = build_model(baseline_cfg(), c_in=5).translator.state_dict()
    plain_t = build_model(apply_row(5), c_in=5).translator.state_dict()

    # The in/out 1x1 convs match by design (so encoder/decoder are unchanged),
    # but the BLOCK parameter sets differ -> the full translator state_dicts are
    # NOT load-compatible. Assert the block key sets diverge.
    tau_block_keys = {k for k in tau_t if k.startswith("blocks.")}
    plain_block_keys = {k for k in plain_t if k.startswith("blocks.")}
    assert tau_block_keys != plain_block_keys
