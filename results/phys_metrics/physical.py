"""Physical fracture metrics: crack length-over-time + time-to-fracture-onset.

This module turns a binary mask STACK (one frame per simulation step) into the
two physically-interpretable, framework-free curves the Phase-5 paper reports:

  * crack length-over-time -- skeleton-pixel count per frame, optionally scaled to
    a physical length by ``pixel_pitch``;
  * time-to-fracture-onset -- the first frame index where the crack appears.

It is built ON TOP of ``phys_metrics.postproc.skeleton_and_components`` (PHYS-01):
there is exactly one skeletonizer in the codebase, reused here by import (S4),
NOT re-rolled.

Implements:
  * PHYS-03 -- ``crack_length_curve`` + ``time_to_onset``.
  * D-02    -- the chosen physical descriptors are length-over-time + onset.
               Tip-position error is EXPLICITLY NOT selected (D-02); do not add it
               here.

Framework rule (S1): numpy / scikit-image / stdlib ONLY. NO torch, NO
tensorflow.

Binarization / pixel-pitch note (A3 / Pitfall 3): callers pass an ALREADY
-binarized ``(n_frames, H, W)`` mask stack, binarized upstream at the
val-calibrated threshold via
``results/dashboard/plots.py::read_calibrated_threshold`` — never re-derived
here. ``pixel_pitch`` converts skeleton-pixel counts to physical length; callers
should source it from ``grid.json`` when available, else leave the default 1.0
and report length in pixels.

Run:
    cd dynamic-fracture && python -c "import sys; sys.path.insert(0,'results'); \
import numpy as np; from phys_metrics.physical import crack_length_curve, time_to_onset; \
s=np.zeros((3,8,8),np.uint8); s[2,4,1:6]=1; \
print(crack_length_curve(s), time_to_onset(s))"
"""

from __future__ import annotations

import numpy as np

from phys_metrics.postproc import skeleton_and_components

# ---- numpy + reused PHYS-01 post-processor only; NO torch / tensorflow (S1) ----


def crack_length_curve(
    mask_stack: np.ndarray,
    *,
    pixel_pitch: float = 1.0,
    connectivity: int = 2,
) -> np.ndarray:
    """Per-frame crack length = skeleton-pixel count x pixel_pitch (PHYS-03, D-02).

    Parameters
    ----------
    mask_stack : np.ndarray
        ``(n_frames, H, W)`` bool/uint8 binary mask stack, already binarized
        upstream at the val-calibrated threshold (Pitfall 3).
    pixel_pitch : float, keyword-only
        Physical length of one skeleton pixel. Default ``1.0`` => length reported
        in pixels. Source from ``grid.json`` when available (A3).
    connectivity : int, keyword-only
        Forwarded to ``skeleton_and_components`` (2 => 8-connectivity).

    Returns
    -------
    np.ndarray
        ``(n_frames,)`` float array of crack lengths, one per frame. For a
        no-healing (monotone) mask stack this is non-decreasing; empty frames
        contribute 0.0.

    Notes
    -----
    Reuses ``phys_metrics.postproc.skeleton_and_components`` (S4) — the single
    skeletonizer — rather than importing ``skeletonize`` again.
    """
    arr = np.asarray(mask_stack)
    if arr.ndim != 3:
        raise ValueError(
            f"mask_stack must be 3-D (n_frames, H, W); got ndim={arr.ndim}, "
            f"shape={arr.shape}"
        )
    lengths = [
        float(skeleton_and_components(arr[t], connectivity=connectivity)[0].sum())
        * float(pixel_pitch)
        for t in range(arr.shape[0])
    ]
    return np.asarray(lengths, dtype=float)


def time_to_onset(mask_stack: np.ndarray, *, min_pixels: int = 1) -> int:
    """First frame index with >= ``min_pixels`` positive pixels, else -1 (PHYS-03).

    Parameters
    ----------
    mask_stack : np.ndarray
        ``(n_frames, H, W)`` bool/uint8 binary mask stack.
    min_pixels : int, keyword-only
        Minimum number of positive pixels for a frame to count as fracture
        onset. Default ``1``.

    Returns
    -------
    int
        The smallest frame index whose positive-pixel count is
        ``>= min_pixels``; ``-1`` if no frame reaches the threshold.
    """
    arr = np.asarray(mask_stack)
    if arr.ndim != 3:
        raise ValueError(
            f"mask_stack must be 3-D (n_frames, H, W); got ndim={arr.ndim}, "
            f"shape={arr.shape}"
        )
    counts = arr.astype(bool).reshape(arr.shape[0], -1).sum(axis=1)
    onset = np.nonzero(counts >= min_pixels)[0]
    return int(onset[0]) if onset.size else -1
