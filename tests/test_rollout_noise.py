# tests/test_rollout_noise.py
"""MODEL-02: no-healing-safe feedback-noise injection in ``rollout_losses`` (T-03-10).

Feedback-noise perturbs the fed-back mask to harden the model against its own
rollout drift, but it must respect the hard AR-feedback no-healing rule: noise may
ADD damage (0->1) but never REMOVE it (1->0). The invariants this module pins:

  * with ``feedback_noise_std`` (and/or ``feedback_noise_p``) and ``no_healing``,
    the accumulated fed-back state never drops below the seed mask (no 1->0) and
    stays bounded in [0, 1];
  * ``feedback_noise_std=0.0`` is a no-op: it reproduces the no-noise rollout
    byte-for-byte (no RNG consumed, identical output).

torch is Gilbreth-only; the availability guard + ``sys.path`` shim mirror
``test_determinism.py`` / ``test_rollout_pushforward.py``.

Run:  cd dynamic-fracture && python -m pytest tests/test_rollout_noise.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ---- torch-availability guard (probe a real submodule; stub is not a package) ----
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
N_STEPS = 4
C = 2          # synth channel count (0 = mask, 1 = physics)


def _tiny_model():
    cfg = Config(sequence_length=T, hid_s=8, hid_t=16, n_temporal=1)
    torch.manual_seed(0)
    return build_model(cfg, c_in=C).eval()


def _batch_from_synth(synth_runs):
    L = T + N_STEPS
    xs = [torch.from_numpy(run["features"][:L]).float() for run in synth_runs[:2]]
    return torch.stack(xs, dim=0)               # (2, L, C, H, W)


def _criterion():
    return SegLoss(bce_weight=1.0, dice_weight=0.0, focal_weight=0.0,
                   tversky_weight=0.0, growth_weight=0.0)


def test_feedback_noise_never_heals(synth_runs):
    """Noise + no_healing: the fed-back state never drops below the seed mask
    (no 1->0) and stays bounded in [0, 1]."""
    model = _tiny_model()
    batch = _batch_from_synth(synth_runs)
    seed_mask = batch[:, T - 1, 0]              # the no-healing prev seed

    torch.manual_seed(7)
    _, last_pred, _ = rollout_losses(
        model, batch, T, N_STEPS, _criterion(),
        feedback_hard=True, no_healing=True,
        feedback_noise_std=0.1, feedback_noise_p=0.05)

    # bounded
    assert float(last_pred.min()) >= 0.0
    assert float(last_pred.max()) <= 1.0
    # never heals: every pixel set in the seed is still set after the rollout
    assert (last_pred + 1e-6 >= seed_mask).all()


def test_zero_noise_is_noop(synth_runs):
    """feedback_noise_std=0.0 reproduces the no-noise rollout exactly (no RNG
    consumed, identical fed-back state + loss)."""
    model = _tiny_model()
    batch = _batch_from_synth(synth_runs)

    torch.manual_seed(123)
    loss_a, pred_a, _ = rollout_losses(
        model, batch, T, N_STEPS, _criterion(),
        feedback_hard=True, no_healing=True, feedback_noise_std=0.0)

    torch.manual_seed(123)
    loss_b, pred_b, _ = rollout_losses(
        model, batch, T, N_STEPS, _criterion(),
        feedback_hard=True, no_healing=True)

    assert torch.allclose(pred_a, pred_b)
    assert torch.allclose(loss_a, loss_b)
