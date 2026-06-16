# tests/test_probs_io.py
"""METR-07: raw-probability / GT fp16 persistence round-trip contract (Plan 03).

Pins the LOCKED behavior the Plan-03 ``probs_io`` bodies must satisfy:
  * ``save_case_probs_gt`` then ``load_case_probs_gt`` round-trips a
    ``(n_frames, H, W)`` probability stack + a binary GT stack.
  * probs are stored as fp16 (lossy) and loaded back as fp32 equal to the input
    within fp16 tolerance; GT is stored/loaded as uint8 exactly.
  * the loader reads with ``allow_pickle=False`` (threat T-02D-07, Tampering)
    and raises ``FileNotFoundError`` (fail loud) when the case dir has no npz.

These tests are RED while ``probs_io`` raises ``NotImplementedError`` (a runtime
failure, NOT a collection/import error). Plan 03 turns them GREEN.

Run:  cd dynamic-fracture && python -m pytest tests/test_probs_io.py -x
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

from dashboard.probs_io import (  # noqa: E402
    load_case_probs_gt,
    save_case_probs_gt,
)


def _synth_stacks(n: int = 5, h: int = 8, w: int = 6):
    """Deterministic (probs, gts): random probs in [0,1], binary GT (seed=42)."""
    rng = np.random.default_rng(42)
    probs = rng.random((n, h, w)).astype(np.float32)
    gts = (rng.random((n, h, w)) > 0.5).astype(np.uint8)
    return probs, gts


def test_probs_gt_roundtrip(tmp_path):
    probs, gts = _synth_stacks()
    case_dir = tmp_path / "test_case"

    save_case_probs_gt(case_dir, probs, gts)

    # both per-case files written
    assert (case_dir / "probs.npz").exists()
    assert (case_dir / "gt.npz").exists()

    loaded_probs, loaded_gts = load_case_probs_gt(case_dir)

    # shapes preserved
    assert loaded_probs.shape == probs.shape
    assert loaded_gts.shape == gts.shape

    # probs upcast to fp32 for arithmetic; equal within fp16 tolerance
    assert loaded_probs.dtype == np.float32
    np.testing.assert_allclose(
        loaded_probs, probs.astype(np.float16).astype(np.float32), rtol=0, atol=0
    )
    np.testing.assert_allclose(loaded_probs, probs, rtol=1e-2, atol=1e-3)

    # GT round-trips exactly as uint8
    assert loaded_gts.dtype == np.uint8
    np.testing.assert_array_equal(loaded_gts, gts)


def test_save_accepts_array_like_lists(tmp_path):
    # the rollout appends per-step frames to python lists; save must stack them.
    probs, gts = _synth_stacks(n=4, h=5, w=3)
    case_dir = tmp_path / "list_case"

    save_case_probs_gt(case_dir, list(probs), [g for g in gts])

    loaded_probs, loaded_gts = load_case_probs_gt(case_dir)
    assert loaded_probs.shape == (4, 5, 3)
    assert loaded_gts.shape == (4, 5, 3)


def test_load_missing_raises_filenotfound(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        load_case_probs_gt(missing)
