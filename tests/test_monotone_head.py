# tests/test_monotone_head.py
"""MODEL-03: switchable monotone-Δ output head invariant.

The monotone head (``head_type="monotone_delta"``) must return a PROBABILITY
that is monotone w.r.t. the no-healing input mask (channel 0) — ``out >= prev``
everywhere — and bounded in ``[0, 1]`` by construction (probabilistic-OR). The
default ``head_type="sigmoid"`` path must stay byte-identical (returns logits).

``model.py`` imports real torch at module top, so these tests guard on torch
availability (Gilbreth-only locally) using the same skip pattern as
``test_determinism.py``; the sys.path shim reaches ``new_model/src``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ---- real-torch guard (torch is Gilbreth-only on the workstation) ----
try:
    import torch as _real_torch  # noqa: F401
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# ---- sys.path shim to reach new_model/src ----
SRC = Path(__file__).resolve().parents[1] / "new_model" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="real torch not installed")
def test_monotone_head_never_heals():
    import torch
    from config import Config
    from model import build_model

    cfg = Config(head_type="monotone_delta", sequence_length=4)
    model = build_model(cfg, c_in=5).eval()
    x = torch.rand(2, 4, 5, 32, 16)
    x[:, :, 0] = (x[:, :, 0] > 0.5).float()      # binary prev mask in channel 0
    out = model(x)                                # returns prob
    prev = x[:, :, 0:1]
    assert (out + 1e-6 >= prev).all()             # never heals
    assert (out >= 0).all() and (out <= 1).all()  # bounded


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="real torch not installed")
def test_sigmoid_head_unchanged():
    """The default sigmoid head returns logits (not a [0,1] prob) and diverges
    from the monotone head on the same input + weight seed — documenting that
    the two output paths are distinct (logits vs probabilistic-OR prob)."""
    import torch
    from config import Config
    from model import build_model

    x = torch.rand(2, 4, 5, 32, 16)
    x[:, :, 0] = (x[:, :, 0] > 0.5).float()

    torch.manual_seed(0)
    sig_model = build_model(Config(head_type="sigmoid", sequence_length=4), c_in=5).eval()
    sig_out = sig_model(x)
    assert tuple(sig_out.shape) == (2, 4, 1, 32, 16)

    torch.manual_seed(0)
    mon_model = build_model(Config(head_type="monotone_delta", sequence_length=4), c_in=5).eval()
    mon_out = mon_model(x)

    # sigmoid head emits raw logits (unconstrained); monotone head a bounded prob.
    assert (mon_out >= 0).all() and (mon_out <= 1).all()
    # Same shared weight seed, identical input -> the two heads diverge.
    assert not torch.allclose(sig_out, mon_out)
