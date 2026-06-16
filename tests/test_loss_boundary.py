# tests/test_loss_boundary.py
"""MODEL-01 / MODEL-03: SegLoss boundary term + ramp + prob-path (no double-sigmoid).

Boundary term (MODEL-01): a Config-gated, ramp-scaled crack-front penalty added to
``SegLoss``; default off (``boundary_weight=0.0``) so legacy behavior is byte-identical.
The default boundary target is a torch-only morphological band (RESEARCH Open Question 3 —
avoids a SciPy dependency mid-phase).

Prob-path branch (MODEL-03): when the monotone-delta head returns a probability,
``SegLoss`` must NOT apply a second sigmoid (RESEARCH Pitfall 4 / threat T-03-03).

torch is Gilbreth-only; the availability guard and ``sys.path`` shim are copied from
``tests/test_determinism.py`` (lines 20-24, 36-40, 53).

Run:  cd dynamic-fracture && python -m pytest tests/test_loss_boundary.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ---- torch-availability guard (test_determinism.py:20-24) ----
# NB: test_determinism.py injects a bare ``torch`` STUB into sys.modules when real
# torch is absent, so a plain ``import torch`` can spuriously succeed in the full
# suite. Probe a real submodule (the stub is not a package) to detect real torch.
try:
    import torch as _real_torch  # noqa: F401
    import torch.nn.functional as _F_probe  # noqa: F401 — stub lacks submodules
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# ---- sys.path shim to reach new_model/src (test_determinism.py:36-40) ----
SRC = Path(__file__).resolve().parents[1] / "new_model" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Skip the whole module cleanly when real torch is absent (test_determinism.py:53).
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="real torch not installed")

if TORCH_AVAILABLE:
    import torch
    import torch.nn.functional as F

    from losses import SegLoss  # noqa: E402


def _small_logits_target(shape=(2, 1, 33, 17)):
    """Deterministic small logits + binary target (seed pinned in-test)."""
    torch.manual_seed(42)
    logits = torch.randn(*shape)
    target = (torch.rand(*shape) > 0.5).float()
    return logits, target


def _front_band(shape=(2, 1, 33, 17), rows=slice(10, 14)):
    band = torch.zeros(*shape)
    band[..., rows, :] = 1.0
    return band


def test_boundary_term_runs_and_shapes():
    """Boundary-only SegLoss returns a finite 0-dim scalar."""
    logits, target = _small_logits_target()
    band = _front_band()
    loss_fn = SegLoss(bce_weight=0.0, dice_weight=0.0, tversky_weight=0.0,
                      boundary_weight=0.5)
    loss = loss_fn(logits, target, dist_map=band, boundary_scale=1.0)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert float(loss) > 0.0


def test_boundary_scale_monotone():
    """The ramp scales the boundary contribution; scale=0 zeroes the term."""
    logits = torch.zeros(2, 1, 33, 17)        # p = sigmoid(0) = 0.5 everywhere
    target = torch.zeros_like(logits)         # all background near the band
    band = _front_band()
    loss_fn = SegLoss(bce_weight=0.0, dice_weight=0.0, tversky_weight=0.0,
                      boundary_weight=0.5)
    loss0 = loss_fn(logits, target, dist_map=band, boundary_scale=0.0)
    loss1 = loss_fn(logits, target, dist_map=band, boundary_scale=1.0)
    assert float(loss0) < float(loss1)        # boundary penalty grows with scale

    # boundary_scale=0 truly zeroes the term: identical to boundary_weight=0.
    off = SegLoss(bce_weight=0.0, dice_weight=0.0, tversky_weight=0.0,
                  boundary_weight=0.0)
    loss_off = off(logits, target, dist_map=band, boundary_scale=1.0)
    assert float(loss0) == pytest.approx(float(loss_off), abs=1e-7)


def test_no_double_sigmoid_prob_path(golden_masks):
    """Monotone-delta head: BCE is computed on the probability, not sigmoid(prob)."""
    _, _, prob = golden_masks
    p = torch.from_numpy(prob).float().unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)
    target = (p >= 0.5).float()
    loss_fn = SegLoss(bce_weight=1.0, dice_weight=0.0, tversky_weight=0.0,
                      head_type="monotone_delta")
    loss = loss_fn(p, target)

    expected = F.binary_cross_entropy(p, target)            # single-sigmoid (correct)
    assert float(loss) == pytest.approx(float(expected), abs=1e-5)

    # And explicitly NOT the double-sigmoid value (the bug this pins shut).
    double = F.binary_cross_entropy(torch.sigmoid(p), target)
    assert abs(float(loss) - float(double)) > 1e-4
