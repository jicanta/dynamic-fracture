"""Shared pytest fixtures for the data-integrity / comparison-harness phase.

This conftest is the Wave-0 prerequisite for the whole phase. It provides:

  * a ``sys.path`` shim so the framework-free shared modules that live at the
    repo root (``dynamic-fracture/``) -- e.g. ``case_registry`` and
    ``frac_metrics``, created in Wave 2 -- import cleanly from tests;
  * a ``synth_runs`` fixture: a tiny in-memory, multi-run dataset shaped like
    the real grid data (channel 0 = binary fracture mask), with KNOWN and
    DISTINCT per-run frame counts so val-split disjointness can be asserted
    deterministically -- without touching Gilbreth scratch;
  * ``golden_masks`` / ``golden_degenerate`` fixtures: deterministic
    ``gt_bin`` / ``pred_bin`` / ``prob`` numpy arrays for the metrics-parity
    test (a real partial-overlap case and the degenerate all-zero 0/0 -> 0.0
    case).

All fixtures are SEEDED (numpy seed 42 / ``np.random.default_rng(42)``) so the
downstream parity and disjointness evidence is reproducible across runs
(threat T-01-01: a non-seeded fixture would make parity tests flaky and yield a
false integrity signal). The array internals mirror
``new_model/scripts/make_synth_data.py`` (tiny grid, binary mask in channel 0,
extra physics channel) but are built in-memory rather than shelled out.

Run:  cd dynamic-fracture && python -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest

# ---- repo-root sys.path shim ----
# repo root = dynamic-fracture/ (parent of tests/); makes case_registry /
# frac_metrics importable once those modules land in Wave 2.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---- synthetic grid geometry ----
# Small stand-in for the real 321x161 grid; large enough to exercise
# scatter/crop logic, small enough to keep fixtures sub-millisecond.
SYNTH_H: int = 33          # grid height (rows)
SYNTH_W: int = 17          # grid width (cols)
SYNTH_C: int = 2           # channels: 0 = binary fracture mask, 1 = physics field
SEED: int = 42

# Three runs with DISTINCT frame counts so frame-boundary val splits are
# exercised (the split logic must not assume equal-length runs).
SYNTH_RUN_SPECS: List[Tuple[str, int]] = [
    ("run_V100", 20),
    ("run_V150", 26),
    ("run_V200", 31),
]


# ---- synth multi-run dataset fixture ----
@pytest.fixture
def synth_runs() -> List[Dict[str, object]]:
    """A tiny multi-run dataset shaped like the real preprocessed grids.

    Returns a list of per-run dicts::

        {"name": str, "n_frames": int, "features": np.ndarray}

    where ``features`` has shape ``(n_frames, C, H, W)`` (float32). Channel 0 is
    a binary fracture mask in {0, 1} that grows monotonically over frames (a
    no-healing crack, mirroring the real data's monotone fracture growth);
    channel 1 is a deterministic float physics field. Frame counts are KNOWN
    and DISTINCT (20, 26, 31) so val-split disjointness is assertable.
    """
    rng = np.random.default_rng(SEED)
    runs: List[Dict[str, object]] = []
    for name, n_frames in SYNTH_RUN_SPECS:
        features = np.zeros((n_frames, SYNTH_C, SYNTH_H, SYNTH_W), dtype=np.float32)
        # channel 0: monotonically growing binary mask (no-healing crack)
        for t in range(n_frames):
            grown_rows = int(round((t + 1) / n_frames * SYNTH_H))
            features[t, 0, :grown_rows, :] = 1.0
        # channel 1: deterministic physics-like float field in [0, 1)
        features[:, 1, :, :] = rng.random((n_frames, SYNTH_H, SYNTH_W)).astype(np.float32)
        runs.append({"name": name, "n_frames": n_frames, "features": features})
    return runs


# ---- golden mask fixtures (metrics parity) ----
@pytest.fixture
def golden_masks() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic ``(gt_bin, pred_bin, prob)`` for the metrics-parity test.

    Each array is shape ``(H, W)``:
      * ``gt_bin`` / ``pred_bin`` -- uint8 in {0, 1}, two PARTIALLY OVERLAPPING
        rectangles so tp/fp/fn are all nonzero (exercises a real F1, not a
        degenerate 0/0).
      * ``prob`` -- float in [0, 1] built so that
        ``(prob >= 0.5).astype(uint8) == pred_bin`` exactly, i.e. binarizing the
        probability map at 0.5 reproduces ``pred_bin``.
    """
    rng = np.random.default_rng(SEED)

    gt_bin = np.zeros((SYNTH_H, SYNTH_W), dtype=np.uint8)
    pred_bin = np.zeros((SYNTH_H, SYNTH_W), dtype=np.uint8)
    # gt rectangle and a shifted pred rectangle -> overlap (tp), gt-only (fn),
    # pred-only (fp) all guaranteed nonzero.
    gt_bin[5:20, 3:12] = 1
    pred_bin[8:23, 5:14] = 1

    # prob: < 0.5 where pred_bin == 0, >= 0.5 where pred_bin == 1.
    prob = np.where(
        pred_bin == 1,
        rng.uniform(0.5, 1.0, size=(SYNTH_H, SYNTH_W)),
        rng.uniform(0.0, 0.5, size=(SYNTH_H, SYNTH_W)),
    ).astype(np.float64)
    # invariant the parity test relies on
    assert np.array_equal((prob >= 0.5).astype(np.uint8), pred_bin)

    return gt_bin, pred_bin, prob


@pytest.fixture
def golden_degenerate() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All-zero ``(gt_bin, pred_bin, prob)`` to pin the 0/0 -> 0.0 convention.

    With no positives anywhere, tp = fp = fn = 0; precision/recall/F1 must
    resolve to 0.0 (not NaN). ``prob`` is all-zero so binarization at 0.5 also
    yields the all-zero ``pred_bin``.
    """
    gt_bin = np.zeros((SYNTH_H, SYNTH_W), dtype=np.uint8)
    pred_bin = np.zeros((SYNTH_H, SYNTH_W), dtype=np.uint8)
    prob = np.zeros((SYNTH_H, SYNTH_W), dtype=np.float64)
    return gt_bin, pred_bin, prob
