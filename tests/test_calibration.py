# tests/test_calibration.py
"""CMP-04: old-side threshold calibration mirrors the new-side objective/grid.

calibrate() takes arrays (no TF), so the objective and grid are unit-testable
locally; the Gilbreth job supplies real frozen-model val probabilities.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "kathleens-model"
for _p in (str(REPO_ROOT), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from source.calibrate import calibrate, default_grid  # noqa: E402
import frac_metrics  # noqa: E402


def test_grid_is_identical():
    # Byte-identical to the new side's np.linspace(0.2, 0.8, 25).
    assert np.array_equal(default_grid(), np.linspace(0.2, 0.8, 25))
    assert len(default_grid()) == 25


def test_calibrate_picks_known_max():
    # gt foreground = first two cells. Probabilities chosen so the perfect-
    # separation window (0.49, 0.50] contains exactly ONE grid point: 0.5.
    gt = np.array([1, 1, 0, 0], dtype=np.uint8)
    prob = np.array([0.80, 0.50, 0.49, 0.20], dtype=np.float64)
    res = calibrate([prob], [gt])
    assert abs(res["threshold"] - 0.5) < 1e-6
    assert abs(res["val_f1"] - 1.0) < 1e-9


def test_calibrate_deterministic():
    rng = np.random.default_rng(42)
    probs = [rng.random((6, 6)) for _ in range(3)]
    gts = [(rng.random((6, 6)) > 0.5).astype(np.uint8) for _ in range(3)]
    r1 = calibrate(probs, gts)
    r2 = calibrate(probs, gts)
    assert r1 == r2


def test_objective_matches_shared_f1():
    # The reported val_f1 must equal the shared per_frame_metrics micro-F1
    # recomputed at the chosen threshold over the same frames.
    rng = np.random.default_rng(7)
    probs = [rng.random((8, 8)) for _ in range(4)]
    gts = [(rng.random((8, 8)) > 0.5).astype(np.uint8) for _ in range(4)]
    res = calibrate(probs, gts)
    thr = res["threshold"]
    tp = fp = fn = 0.0
    for p, g in zip(probs, gts):
        m = frac_metrics.per_frame_metrics(g, (p >= thr).astype(np.uint8), p)
        tp += m["tp"]; fp += m["fp"]; fn += m["fn"]
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    assert abs(res["val_f1"] - f1) < 1e-12
