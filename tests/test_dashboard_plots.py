# tests/test_dashboard_plots.py
"""METR-02 / METR-06 / METR-07: dashboard figure contracts.

Pins the LOCKED conventions the Plan-05 implementation must satisfy:
  * METR-02/D-11 -- variable-length F1 curves resample to a common 100-pt grid;
                    a median+IQR band figure is produced.
  * METR-07/D-12 -- threshold sweep on synthetic probs reproduces the
                    0.5-threshold F1 and returns ``(thr_grid, macro_f1)`` arrays
                    (reuses the ``golden_masks`` ``prob>=0.5 == pred`` invariant).
  * METR-06/D-10 -- ``fpfn_rgb``: red where pred&~gt, blue where gt&~pred,
                    gray where both, black background.

Wave-0 state: the ``plots`` bodies raise ``NotImplementedError`` -> RED (runtime
failure, NOT a collection/import error). Plan 05 turns them GREEN.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ---- sys.path shim: repo root + results + results/dashboard ----
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
DASHBOARD = RESULTS / "dashboard"
for _p in (str(REPO_ROOT), str(RESULTS), str(DASHBOARD)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dashboard.plots import (  # noqa: E402
    COLOR_BG,
    COLOR_FN,
    COLOR_FP,
    COLOR_TP,
    fpfn_rgb,
    stability_band,
    threshold_curve,
)


def test_stability_band_resample(tmp_path):
    # Variable-length per-case curves must resample onto a common 100-pt grid.
    curves = [np.linspace(0.0, 1.0, k) for k in (12, 25, 50)]
    out = tmp_path / "stability.png"
    stability_band(curves, out, n_grid=100, label="median F1 (3 cases)")
    assert out.exists()


def test_threshold_curve(golden_masks, tmp_path):
    gt, pred, prob = golden_masks
    probs = prob[None]        # (1, H, W) single-frame stack
    gts = gt[None]
    out = tmp_path / "threshold.png"
    thr_grid = np.linspace(0.05, 0.95, 19)
    grid, macro_f1 = threshold_curve(probs, gts, out, thr_grid=thr_grid)
    grid = np.asarray(grid, dtype=float)
    macro_f1 = np.asarray(macro_f1, dtype=float)
    assert grid.shape == macro_f1.shape
    # At the threshold nearest 0.5, the sweep must reproduce pred_bin's F1.
    tp = int(((gt == 1) & (pred == 1)).sum())
    fp = int(((gt == 0) & (pred == 1)).sum())
    fn = int(((gt == 1) & (pred == 0)).sum())
    expected = 2 * tp / (2 * tp + fp + fn)
    i = int(np.argmin(np.abs(grid - 0.5)))
    assert macro_f1[i] == pytest.approx(expected)
    assert out.exists()


def test_fpfn_coloring():
    gt = np.zeros((4, 4), dtype=np.uint8)
    pred = np.zeros((4, 4), dtype=np.uint8)
    gt[0, 0] = 1; pred[0, 0] = 1     # both -> gray (TP)
    pred[1, 1] = 1                   # pred only -> red (FP)
    gt[2, 2] = 1                     # gt only -> blue (FN)
    rgb = fpfn_rgb(gt, pred)
    assert rgb.shape == (4, 4, 3)
    assert rgb.dtype == np.uint8
    assert tuple(rgb[0, 0]) == COLOR_TP
    assert tuple(rgb[1, 1]) == COLOR_FP
    assert tuple(rgb[2, 2]) == COLOR_FN
    assert tuple(rgb[3, 3]) == COLOR_BG     # background stays black
