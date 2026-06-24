"""Mask post-processing for the physical-metrics paper: skeleton + components.

This module is the GEOMETRIC FOUNDATION of the Phase-5 physical metrics. It
exposes a single post-processor, ``skeleton_and_components``, that thins a binary
fracture mask to a 1-pixel-wide skeleton and labels its connected components.
The crack-length and onset metrics (``phys_metrics.physical``, PHYS-03) and the
paper figures both build on this one function — there is exactly ONE place that
skeletonizes, so a change to the morphology convention propagates everywhere.

Implements:
  * PHYS-01 -- skeletonize (scikit-image medial-axis thinning) + 8-connected
    connected-component labelling (``connectivity=2`` default => 8-conn).

Framework rule (S1): numpy / scikit-image / stdlib ONLY. NO torch, NO
tensorflow — these metric modules live on the numpy side of the hard
cross-framework boundary and a grep gate enforces it.

Binarization note (Pitfall 3): callers pass an ALREADY-binarized mask. The
val-calibrated threshold must be sourced upstream via
``results/dashboard/plots.py::read_calibrated_threshold`` — never re-derived
here, so the geometry stays physics-driven, not threshold-driven.

Run:
    cd dynamic-fracture && python -c "import sys; sys.path.insert(0,'results'); \
import numpy as np; from phys_metrics.postproc import skeleton_and_components; \
m=np.zeros((8,8),np.uint8); m[3,1:6]=1; \
print(skeleton_and_components(m)[0].sum())"
"""

from __future__ import annotations

import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize

# ---- numpy/skimage only; intentionally NO torch / tensorflow import (S1) ----


def skeleton_and_components(mask: np.ndarray, *, connectivity: int = 2):
    """Skeletonize a binary mask and label its connected components (PHYS-01).

    Parameters
    ----------
    mask : np.ndarray
        ``(H, W)`` bool/uint8 binary mask. Already binarized by the caller at
        the val-calibrated threshold (Pitfall 3) — this function does NOT
        threshold.
    connectivity : int, keyword-only
        Connected-component connectivity passed to ``skimage.measure.label``.
        ``2`` (default) => 8-connectivity in 2-D; ``1`` => 4-connectivity.

    Returns
    -------
    (skeleton, labels, props) : Tuple[np.ndarray, np.ndarray, list]
        ``skeleton`` -- bool ``(H, W)`` 1-px-wide medial-axis skeleton.
        ``labels``   -- int ``(H, W)`` component-label image (0 = background).
        ``props``    -- list of ``skimage.measure.regionprops`` RegionProperties,
                        one per labelled component.

    An empty mask yields an all-False skeleton (0 px), an all-zero label image
    (``labels.max() == 0``) and an empty ``props`` list — no exception.

    Notes
    -----
    Thinning/labelling are delegated to scikit-image (RESEARCH "Don't
    Hand-Roll"): no bespoke flood-fill or thinning loop lives here.
    """
    m = np.asarray(mask).astype(bool)
    skel = skeletonize(m)
    labels = label(m, connectivity=connectivity)
    props = regionprops(labels)
    return skel, labels, props
