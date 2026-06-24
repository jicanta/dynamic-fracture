# tests/test_calibration_ece.py
"""PHYS-05: reliability / expected-calibration-error golden test (RED, Wave 0).

Pins the contract Wave-2 ``results/phys_metrics/calibration.py`` must satisfy:
``reliability_ece(probs, gts, n_bins=15)`` -> ``(ece, bins)`` where, on the
``perfectly_calibrated_probs`` fixture (per-bin mean confidence == empirical
positive fraction), ``ece`` is 0 to machine precision and the per-bin
populations sum to N (no point dropped or double-counted).

NOTE: distinct from ``tests/test_calibration.py`` (decision-THRESHOLD calibration
on the val set); this is probability calibration / reliability.

Wave-0 state: ``phys_metrics`` does not exist yet -> COLLECTION ImportError
(the intended RED). Wave 2 turns it GREEN.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ---- sys.path shim: repo root + results + results/phys_metrics ----
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
for _p in (str(REPO_ROOT), str(RESULTS), str(RESULTS / "phys_metrics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from phys_metrics.calibration import reliability_ece  # noqa: E402


def test_ece_zero_on_perfectly_calibrated(perfectly_calibrated_probs):
    probs, gts = perfectly_calibrated_probs
    ece, _bins = reliability_ece(probs, gts, n_bins=15)
    assert ece == pytest.approx(0.0, abs=1e-6)


def test_reliability_bins_population_sums_to_n(perfectly_calibrated_probs):
    probs, gts = perfectly_calibrated_probs
    _ece, bins = reliability_ece(probs, gts, n_bins=15)
    populations = np.asarray(bins, dtype=float)         # per-bin populations
    assert int(round(populations.sum())) == probs.size
