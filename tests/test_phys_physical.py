# tests/test_phys_physical.py
"""PHYS-03: crack-length monotonicity + onset-frame golden test (RED, Wave 0).

Pins the contract Wave-2 ``results/phys_metrics/physical.py`` must satisfy on
the ``growing_crack_stack`` fixture (frames 0-1 empty, then a connected crack
that grows each frame, onset == 2):
  * ``crack_length_curve(mask_stack, *, pixel_pitch=1.0, connectivity=2)`` is
    monotonically NON-DECREASING (a no-healing crack never shrinks), one value
    per frame, zero on the empty pre-onset frames.
  * ``time_to_onset(mask_stack, *, min_pixels=1)`` returns the first frame index
    with >= ``min_pixels`` positive pixels (== 2 for this fixture).

Wave-0 state: ``phys_metrics`` does not exist yet -> COLLECTION ImportError
(the intended RED). Wave 2 turns it GREEN.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest  # noqa: F401

# ---- sys.path shim: repo root + results + results/phys_metrics ----
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
for _p in (str(REPO_ROOT), str(RESULTS), str(RESULTS / "phys_metrics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from phys_metrics.physical import crack_length_curve, time_to_onset  # noqa: E402


def test_crack_length_is_monotone_non_decreasing(growing_crack_stack):
    curve = np.asarray(crack_length_curve(growing_crack_stack), dtype=float)
    assert curve.shape[0] == growing_crack_stack.shape[0]   # one value per frame
    assert curve[0] == 0.0 and curve[1] == 0.0              # empty before onset
    assert np.all(np.diff(curve) >= 0.0)                    # never shrinks


def test_time_to_onset_returns_first_positive_frame(growing_crack_stack):
    assert time_to_onset(growing_crack_stack, min_pixels=1) == 2
