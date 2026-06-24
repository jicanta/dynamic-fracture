# tests/test_phys_boundary.py
"""PHYS-02: HD95 + boundary-F1 golden/oracle test (RED, Wave 0).

Pins the contract Wave-2 ``results/phys_metrics/boundary.py`` must satisfy:
  * ``hd95(pred, gt)`` -> 0.0 for identical masks; matches the ``medpy`` oracle
    (dev-dep) within tolerance on a shifted shape; returns ``nan`` when either
    mask is empty (undefined surface distance).
  * ``boundary_f1(pred, gt, tol=2)`` -> 1.0 for identical masks; both-empty ->
    1.0 (vacuously perfect); exactly-one-empty -> 0.0.

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

from phys_metrics.boundary import boundary_f1, hd95  # noqa: E402


def _square(top: int, left: int, side: int = 10, n: int = 20) -> np.ndarray:
    m = np.zeros((n, n), dtype=np.uint8)
    m[top:top + side, left:left + side] = 1
    return m


def test_hd95_identical_is_zero():
    gt = _square(5, 5)
    assert hd95(gt.copy(), gt.copy()) == pytest.approx(0.0, abs=1e-9)


def test_boundary_f1_identical_is_one():
    gt = _square(5, 5)
    assert boundary_f1(gt.copy(), gt.copy(), tol=2) == pytest.approx(1.0)


def test_empty_mask_conventions():
    nonempty = _square(5, 5)
    empty = np.zeros_like(nonempty)
    # hd95 undefined when a mask is empty -> nan.
    assert np.isnan(hd95(empty, nonempty))
    assert np.isnan(hd95(empty, empty))
    # boundary_f1: both empty -> 1.0 (vacuously perfect); one empty -> 0.0.
    assert boundary_f1(empty, empty, tol=2) == pytest.approx(1.0)
    assert boundary_f1(empty, nonempty, tol=2) == pytest.approx(0.0)
    assert boundary_f1(nonempty, empty, tol=2) == pytest.approx(0.0)


def test_hd95_matches_medpy_oracle():
    # Cross-check against the de-facto medical-seg metric library (dev dep).
    medpy_binary = pytest.importorskip("medpy.metric.binary")
    gt = _square(5, 5)
    pred = _square(7, 7)                       # shifted by (2, 2)
    ours = hd95(pred, gt)
    oracle = float(medpy_binary.hd95(pred.astype(bool), gt.astype(bool)))
    assert ours == pytest.approx(oracle, abs=1e-6)
