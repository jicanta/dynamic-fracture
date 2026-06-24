"""Crack-front boundary metrics for the physical-metrics paper: HD95 + boundary-F1.

This module computes BOTH crack-front boundary-localization metrics that decision
D-03 requires, from a pair of already-binarized masks:

  * ``hd95``        -- the 95th-percentile (symmetric) Hausdorff surface distance.
                       Reviewer-familiar (the de-facto medical-segmentation
                       boundary metric); cross-checked against
                       ``medpy.metric.binary.hd95`` in the golden test.
  * ``boundary_f1`` -- the BF score (Csurka et al. 2013): boundary
                       precision/recall within ``tol`` pixels, harmonic-meaned.
                       Physically interpretable as "fraction of the predicted
                       crack front that lands on the true front (and vice-versa)".

Both are built on a single one-pixel boundary extraction (``_boundary``) and on
scipy Euclidean distance transforms — there is NO hand-rolled Hausdorff /
pairwise-distance loop (RESEARCH §Don't Hand-Roll): the EDT gives every pixel's
distance to the nearest boundary pixel in O(HW).

Implements:
  * PHYS-02 -- HD95 + boundary-F1 on the crack front.
  * D-03    -- BOTH boundary metrics reported (HD95 reviewer-familiar,
               boundary-F1 physically interpretable).

Empty-mask convention (S5, documented inline at each site):
  * ``hd95`` -> ``np.nan`` when EITHER mask has no boundary (surface distance is
    undefined with no surface to measure from / to).
  * ``boundary_f1`` -> ``1.0`` when BOTH masks are empty (vacuously perfect,
    mirroring the D-03 0/0 -> 1.0 aggregation convention) and ``0.0`` when
    exactly one is empty (a present front vs. an absent one is a total miss).

Framework rule (S1): numpy / scipy / stdlib ONLY. NO torch, NO tensorflow —
these metric modules live on the numpy side of the hard cross-framework boundary
and a grep gate enforces it.

Binarization note (Pitfall 3): callers pass ALREADY-binarized masks. The
val-calibrated threshold must be sourced upstream via
``results/dashboard/plots.py::read_calibrated_threshold`` — never re-derived here.

Run:
    cd dynamic-fracture && python -c "import sys; sys.path.insert(0,'results'); \
import numpy as np; from phys_metrics.boundary import hd95, boundary_f1; \
g=np.zeros((20,20),np.uint8); g[5:15,5:15]=1; p=np.zeros((20,20),np.uint8); \
p[7:17,7:17]=1; print(hd95(p,g), boundary_f1(p,g))"
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


# ---- one-pixel boundary extraction ----
def _boundary(mask: np.ndarray) -> np.ndarray:
    """Return the 1-px inner boundary of ``mask`` (foreground minus its erosion).

    ``m & ~binary_erosion(m)`` keeps exactly the foreground pixels that have at
    least one background neighbour under scipy's default cross structuring
    element (connectivity 1) — the same border convention ``medpy`` uses, so the
    HD95 oracle cross-check matches.
    """
    m = mask.astype(bool)
    return m & ~binary_erosion(m)


# ---- PHYS-02 / D-03: 95th-percentile Hausdorff surface distance ----
def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """95th-percentile (symmetric) Hausdorff distance between mask boundaries.

    ``pred`` / ``gt``: ``(H, W)`` bool/uint8 binarized masks. Returns the larger
    of the two directed 95th-percentile boundary distances. Empty-mask
    convention (S5): if EITHER boundary is empty, the surface distance is
    undefined -> ``np.nan``.
    """
    pb, gb = _boundary(pred), _boundary(gt)
    if not pb.any() or not gb.any():
        return np.nan                       # undefined surface distance (S5)
    dt_g = distance_transform_edt(~gb)      # dist from every px to GT boundary
    dt_p = distance_transform_edt(~pb)      # dist from every px to pred boundary
    d_pg = dt_g[pb]                         # pred-boundary px -> nearest GT px
    d_gp = dt_p[gb]                         # GT-boundary px -> nearest pred px
    return max(np.percentile(d_pg, 95), np.percentile(d_gp, 95))


# ---- PHYS-02 / D-03: BF score (boundary precision/recall within tol) ----
def boundary_f1(pred: np.ndarray, gt: np.ndarray, tol: int = 2) -> float:
    """BF score: boundary precision/recall within ``tol`` pixels (Csurka 2013).

    ``pred`` / ``gt``: ``(H, W)`` bool/uint8 masks. Precision = fraction of
    predicted-boundary pixels within ``tol`` of a GT-boundary pixel; recall is
    the symmetric quantity; returns the harmonic mean ``2PR/(P+R)``.

    Empty-mask convention (S5): both-empty -> ``1.0`` (vacuously perfect,
    D-03-style 0/0 -> 1.0); exactly-one-empty -> ``0.0``.
    """
    pb, gb = _boundary(pred), _boundary(gt)
    if not pb.any() and not gb.any():
        return 1.0                          # both empty -> vacuously perfect (S5)
    if not pb.any() or not gb.any():
        return 0.0                          # one empty -> total miss (S5)
    dt_g = distance_transform_edt(~gb)
    dt_p = distance_transform_edt(~pb)
    prec = float((dt_g[pb] <= tol).mean())  # pred boundary px near a GT boundary
    rec = float((dt_p[gb] <= tol).mean())   # GT boundary px near a pred boundary
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
