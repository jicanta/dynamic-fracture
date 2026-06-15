# source/calibrate.py
"""Old-side val-only threshold calibration (CMP-04).

Gives the frozen ConvLSTM a decision threshold chosen with the IDENTICAL
objective and grid as the new FractureTAU pipeline, instead of the hardcoded
0.5 -- otherwise the head-to-head is apples-to-oranges (threat T-08-03).

Mirrors new_model/src/train.py:142-190 (calibrate_threshold) and its grid at
train.py:398 (np.linspace(cfg.thr_min, cfg.thr_max, cfg.thr_steps) with config
defaults thr_min=0.2, thr_max=0.8, thr_steps=25 -> config.py:113-115). The F1 is
the SHARED frac_metrics.per_frame_metrics, so old and new optimise the very same
number. This function takes ARRAYS (no TF), so it is locally unit-testable; the
Gilbreth job feeds it the frozen model's probabilities over the SAME clean
frame-boundary val frames the new pipeline defines (Plan 04, D-07).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Repo-root shim so the numpy-only metric seam is importable (no framework coupling).
REPO_ROOT = Path(__file__).resolve().parents[2]   # source -> kathleens-model -> dynamic-fracture
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from frac_metrics import per_frame_metrics  # noqa: E402


def default_grid() -> np.ndarray:
    """The BYTE-IDENTICAL grid the new side uses.

    new_model/src/train.py:398 builds np.linspace(cfg.thr_min, cfg.thr_max,
    cfg.thr_steps); config.py:113-115 defaults thr_min=0.2, thr_max=0.8,
    thr_steps=25. These literals are hardcoded here on purpose so the two
    pipelines sweep exactly the same thresholds.
    """
    return np.linspace(0.2, 0.8, 25)


def calibrate(val_probs, val_gt_bin, grid=None) -> dict:
    """Pick the threshold maximizing foreground micro-F1 over the val frames.

    Args:
        val_probs:  iterable of per-frame RAW probability maps (array-like).
        val_gt_bin: matching per-frame GT binary maps (array-like, {0,1}).
        grid:       threshold candidates; defaults to np.linspace(0.2, 0.8, 25).

    Returns:
        {"threshold": best_thr, "val_f1": best_f1}.

    Objective (mirrors train.py:180-190): for each threshold, binarize the probs,
    accumulate GLOBAL tp/fp/fn across ALL val frames, compute
    f1 = 2*prec*rec/(prec+rec) (0.0 when a denominator is 0), and take the
    argmax (first/lowest threshold wins ties, same as the new side). F1 comes
    from the shared per_frame_metrics so it is byte-identical to eval.
    """
    if grid is None:
        grid = default_grid()
    probs = [np.asarray(p) for p in val_probs]
    gts = [np.asarray(g) for g in val_gt_bin]
    if len(probs) != len(gts):
        raise ValueError(
            f"val_probs/val_gt_bin length mismatch: {len(probs)} vs {len(gts)}"
        )

    best_thr, best_f1 = float(grid[0]), -1.0
    for thr in grid:
        tp = fp = fn = 0.0
        for p, g in zip(probs, gts):
            pred_bin = (p >= thr).astype(np.uint8)
            m = per_frame_metrics(g, pred_bin, p)   # prob arg unused for the F1 counts
            tp += m["tp"]; fp += m["fp"]; fn += m["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return {"threshold": best_thr, "val_f1": best_f1}
