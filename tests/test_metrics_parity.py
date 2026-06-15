"""CMP-01 / D-12 -- golden-input metric parity for the shared metrics seam.

Both pipelines (new PyTorch ``evaluate.py`` and old TF/Keras
``predict_full_simulation_metrics.py``) emit their per-frame numbers via the SAME
function, ``frac_metrics.per_frame_metrics``. So "do the two pipelines agree?"
reduces locally to "does one function return identical numbers for identical
inputs?" -- which we prove byte-for-byte (``==``, not ``approx``) on golden
arrays. This is the anti-divergence guarantee for threat T-03-01 (duplicated
metric code) without needing either heavyweight framework installed.

NOTE: The full both-pipeline CSV parity (feeding REAL model outputs through each
pipeline's emission loop and diffing ``per_frame_metrics.csv``) is a
Gilbreth-only, manual-only check (see 01-VALIDATION.md). Locally, byte-identity
is guaranteed structurally by routing both pipelines through this single shared
function, which is what these tests pin.

Run:  cd dynamic-fracture && python -m pytest tests/test_metrics_parity.py -q
"""

from __future__ import annotations

import numpy as np

# imported via the conftest.py repo-root sys.path shim
import frac_metrics


# ---- CMP-01: byte-identical outputs for identical inputs ----
def test_parity_identity(golden_masks):
    """Feeding the SAME golden arrays through per_frame_metrics twice (standing
    in for the two pipelines' wrappers, which now call the identical function)
    yields byte-identical f1/iou/bce."""
    gt_bin, pred_bin, prob = golden_masks

    a = frac_metrics.per_frame_metrics(gt_bin, pred_bin, prob)
    b = frac_metrics.per_frame_metrics(gt_bin, pred_bin, prob)

    # byte-identical, not approx -- determinism (no RNG in the metric fn)
    assert a["f1"] == b["f1"]
    assert a["iou"] == b["iou"]
    assert a["bce"] == b["bce"]

    # the golden case is a real partial overlap -> non-degenerate metrics
    assert 0.0 < a["f1"] < 1.0
    assert 0.0 < a["iou"] < 1.0
    assert a["tp"] > 0 and a["fp"] > 0 and a["fn"] > 0


# ---- identity-model path: perfect prediction -> f1==1, iou==1 ----
def test_parity_identity_model(golden_masks):
    """An 'identity model' (pred_bin == gt_bin, prob == gt clamped) scores a
    perfect f1/iou and a small finite BCE."""
    gt_bin, _pred_bin, _prob = golden_masks

    pred_bin = gt_bin.copy()
    # prob equal to the gt mask as float; the metric fn clamps to [EPS, 1-EPS]
    prob = gt_bin.astype(np.float64)

    d = frac_metrics.per_frame_metrics(gt_bin, pred_bin, prob)

    assert d["f1"] == 1.0
    assert d["iou"] == 1.0
    assert np.isfinite(d["bce"])
    assert d["bce"] >= 0.0
    # clamp keeps it finite & small even though prob hit the {0,1} extremes
    assert d["bce"] < 1e-3


# ---- closed-form known values on a tiny hand-built case ----
def test_known_values():
    """2x2 case with hand-computed tp/fp/fn so f1/iou match closed form exactly.

    gt   = [[1,1],[0,0]]   pred = [[1,0],[1,0]]
      tp = 1 (top-left), fp = 1 (bottom-left), fn = 1 (top-right), tn = 1
      precision = 1/2, recall = 1/2, f1 = 1/2, iou = 1/(1+1+1) = 1/3
    """
    gt_bin = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    pred_bin = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    prob = pred_bin.astype(np.float64)

    d = frac_metrics.per_frame_metrics(gt_bin, pred_bin, prob)

    assert d["tp"] == 1.0
    assert d["fp"] == 1.0
    assert d["fn"] == 1.0
    assert d["tn"] == 1.0
    assert d["precision"] == 0.5
    assert d["recall"] == 0.5
    assert d["f1"] == 0.5
    assert d["iou"] == 1.0 / 3.0


# ---- degenerate 0/0 -> 0.0 convention (threat T-03-03) ----
def test_degenerate_zero_convention(golden_degenerate):
    """All-zero gt/pred -> precision/recall/f1/iou all resolve to 0.0 (not NaN)."""
    gt_bin, pred_bin, prob = golden_degenerate

    d = frac_metrics.per_frame_metrics(gt_bin, pred_bin, prob)

    assert d["f1"] == 0.0
    assert d["iou"] == 0.0
    assert d["precision"] == 0.0
    assert d["recall"] == 0.0
    assert np.isfinite(d["bce"])
