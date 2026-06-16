# tests/test_rollout_pushforward.py
"""MODEL-02: pushforward gradient-isolation in ``rollout_losses`` (Pitfall 3 / T-03-09).

The pushforward trick (Brandstetter et al.) detaches the fed-back state and
backprops the LAST unroll step only, so memory no longer scales with the rollout
depth K. The hard invariant this module pins shut:

  * ``ar_pushforward=True``  -> the fed-back state is DETACHED (no gradient flows
    back through it), while the returned loss still carries grad from the final
    model() call (so the last step is scored).
  * ``ar_pushforward=False`` -> full BPTT: the fed-back state requires grad, the
    current (pre-MODEL-02) behavior, unchanged.

``train.py`` imports real torch at module top and pulls in the model, so these
tests guard on torch availability (Gilbreth-only locally) using the same skip
pattern as ``test_determinism.py``. ``rollout_losses`` lives in the ``src``
package (relative imports), so it is imported as ``src.train`` with ``new_model``
on ``sys.path``; ``frac_metrics`` (repo root) is already on ``sys.path`` via the
conftest shim.

Run:  cd dynamic-fracture && python -m pytest tests/test_rollout_pushforward.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ---- torch-availability guard (test_determinism.py injects a bare stub) ----
# Probe a real submodule so the stub torch is correctly detected as "not real".
try:
    import torch as _real_torch  # noqa: F401
    import torch.nn as _real_torch_nn  # noqa: F401
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# ---- sys.path shim to reach the new_model package (src.train) ----
NEW_MODEL = Path(__file__).resolve().parents[1] / "new_model"
if str(NEW_MODEL) not in sys.path:
    sys.path.insert(0, str(NEW_MODEL))

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="real torch not installed")

if TORCH_AVAILABLE:
    import torch

    from src.config import Config
    from src.losses import SegLoss
    from src.model import build_model
    from src.train import rollout_losses


T = 4
N_STEPS = 3
C = 2          # synth channel count (0 = mask, 1 = physics)


def _tiny_model():
    """A tiny FractureTAU on the synth grid geometry (fast on CPU)."""
    cfg = Config(sequence_length=T, hid_s=8, hid_t=16, n_temporal=1)
    torch.manual_seed(0)
    return build_model(cfg, c_in=C).train()


def _batch_from_synth(synth_runs):
    """(B, L, C, H, W) window with L = T + N_STEPS, built from two synth runs."""
    L = T + N_STEPS
    xs = []
    for run in synth_runs[:2]:
        feats = torch.from_numpy(run["features"][:L]).float()   # (L, C, H, W)
        xs.append(feats)
    return torch.stack(xs, dim=0)                                # (2, L, C, H, W)


def _criterion():
    return SegLoss(bce_weight=1.0, dice_weight=0.0, focal_weight=0.0,
                   tversky_weight=0.0, growth_weight=0.0)


def test_pushforward_detaches_fed_back_state(synth_runs):
    """ar_pushforward=True: the fed-back state carries NO gradient, yet the loss
    is still differentiable (scored on the final step only)."""
    model = _tiny_model()
    batch = _batch_from_synth(synth_runs)
    loss, last_pred, _ = rollout_losses(
        model, batch, T, N_STEPS, _criterion(),
        feedback_hard=True, no_healing=True, ar_pushforward=True)

    # The fed-back state is detached -> no gradient flows back through it.
    assert last_pred.requires_grad is False
    # The loss still has grad from the final model() call (last step is scored).
    assert loss.requires_grad is True
    # And it actually backprops cleanly (final-step-only graph is valid).
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_pushforward_off_keeps_graph(synth_runs):
    """ar_pushforward=False (default): full BPTT -> the fed-back state requires
    grad, the pre-MODEL-02 behavior, unchanged."""
    model = _tiny_model()
    batch = _batch_from_synth(synth_runs)
    loss, last_pred, _ = rollout_losses(
        model, batch, T, N_STEPS, _criterion(),
        feedback_hard=True, no_healing=True, ar_pushforward=False)

    assert last_pred.requires_grad is True     # full BPTT keeps the graph
    assert loss.requires_grad is True
