# tests/test_phys_postproc.py
"""PHYS-01: skeletonization + connected-components golden test (RED, Wave 0).

Pins the contract Wave-2 ``results/phys_metrics/postproc.py`` must satisfy:
``skeleton_and_components(mask, *, connectivity=2)`` thins a mask to a 1-px
skeleton and labels its connected components. On a KNOWN mask (a 1-px horizontal
line of length 7 plus one isolated pixel) the skeleton is unchanged (8 px) and
there are exactly 2 components under 8-connectivity (connectivity=2).

Wave-0 state: ``phys_metrics`` does not exist yet, so this file fails at
COLLECTION (ImportError) — the intended RED. Wave 2 turns it GREEN.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest  # noqa: F401  (kept for symmetry with the other phys tests)

# ---- sys.path shim: repo root + results + results/phys_metrics ----
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
for _p in (str(REPO_ROOT), str(RESULTS), str(RESULTS / "phys_metrics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from phys_metrics.postproc import skeleton_and_components  # noqa: E402


def _known_mask() -> np.ndarray:
    """A 1-px line (len 7) + one isolated pixel -> skeleton invariant, 2 comps."""
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[2, 1:8] = 1          # straight 1-px line, length 7 (skeleton-stable)
    mask[8, 9] = 1            # isolated pixel -> a second component
    return mask


def test_skeleton_pixel_count_and_components():
    mask = _known_mask()
    skel, labels, props = skeleton_and_components(mask, connectivity=2)

    # 1-px structures are skeleton-stable: 7 (line) + 1 (pixel) = 8 px.
    assert int(np.asarray(skel).astype(bool).sum()) == 8
    # Two disjoint structures -> exactly 2 connected components (8-connectivity).
    assert int(np.asarray(labels).max()) == 2
    assert len(props) == 2
