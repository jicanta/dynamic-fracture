# tests/test_error_accum.py
"""PHYS-04: teacher-forced vs autoregressive error-gap golden test (RED, Wave 0).

Pins the contract Wave-2 ``results/phys_metrics/error_accum.py`` must satisfy:
``tf_vs_ar_gap(ar_rows, tf_rows)`` returns the per-frame degradation of the AR
rollout relative to teacher forcing, ``gap[i] = f1_tf[i] - f1_ar[i]`` under the
D-03 per-frame F1 convention. Identical inputs -> an all-zero gap; a known
per-frame offset -> the expected gap curve.

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

from phys_metrics.error_accum import tf_vs_ar_gap  # noqa: E402


def test_identical_rows_give_zero_gap():
    rows = [
        {"tp": 10, "fp": 0, "fn": 0},     # f1 = 1.0
        {"tp": 5, "fp": 5, "fn": 5},      # f1 = 0.5
    ]
    gap = np.asarray(tf_vs_ar_gap([dict(r) for r in rows], [dict(r) for r in rows]), dtype=float)
    assert np.allclose(gap, 0.0)


def test_known_offset_gives_expected_gap_curve():
    # Teacher forcing perfect on both frames (f1 = 1.0, 1.0); AR degrades on
    # frame 1 only (tp=5,fp=5,fn=5 -> f1 = 0.5). gap = f1_tf - f1_ar = [0.0, 0.5].
    tf_rows = [{"tp": 10, "fp": 0, "fn": 0}, {"tp": 10, "fp": 0, "fn": 0}]
    ar_rows = [{"tp": 10, "fp": 0, "fn": 0}, {"tp": 5, "fp": 5, "fn": 5}]
    gap = np.asarray(tf_vs_ar_gap(ar_rows, tf_rows), dtype=float)
    assert np.allclose(gap, [0.0, 0.5])
