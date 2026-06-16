"""Dashboard figure contracts: stability band, threshold curve, FP/FN panels.

Three trustworthy figures for the multi-metric dashboard, all rendered headless
(``matplotlib.use("Agg")`` lazily inside each function body, mirroring
``new_model/src/utils.py:103-145``: ``figsize=(11,6)``, ``grid(alpha=0.3)``,
``mkdir(parents=True)`` before save, ``dpi=200``, ``plt.close(fig)``).

Decision IDs implemented here:
  * D-11 -- stability median+IQR band over the 16 cases (resample variable-length
            rollouts onto a common 100-pt grid via ``np.interp``).
  * D-12 -- threshold-sensitivity curve (vectorized sweep over saved probs,
            no-healing via ``np.maximum.accumulate``, mark the Phase-1
            calibrated threshold read from ``calibration.json``).
  * D-10 -- FP/FN dark-background RGB panel + key-frame selection.
  * D-14 -- panels feed the one-command REPORT.md.

ORIENTATION TRAP (Pitfall 2): ``evaluate.py`` ``save_mask_png`` writes masks
``np.flipud``-ed; GT read from ``features.npy`` is NOT flipped. Build BOTH arrays
in ONE consistent orientation before differencing or the FP/FN coloring is
nonsense.

Framework rule (D-13): numpy / matplotlib / pandas / Pillow / stdlib ONLY.
NO torch, NO tensorflow.

This is a Wave-0 CONTRACT module: every public function raises
``NotImplementedError``. Plan 05 fills in the bodies and turns the RED tests
(``tests/test_dashboard_plots.py``) GREEN.

Run:
    cd dynamic-fracture && python -c "import sys; sys.path.insert(0,'results'); \
from dashboard.plots import fpfn_rgb; print(fpfn_rgb)"
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

# ---- repo-root sys.path shim ----
# dashboard -> results -> dynamic-fracture.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---- FP/FN palette (D-10/D-14) ----
# TP gray, FP red, FN blue, on a black background.
COLOR_TP: Tuple[int, int, int] = (160, 160, 160)
COLOR_FP: Tuple[int, int, int] = (220, 40, 40)
COLOR_FN: Tuple[int, int, int] = (40, 90, 220)
COLOR_BG: Tuple[int, int, int] = (0, 0, 0)


# ---- stability band (D-11) ----
def stability_band(
    case_f1_curves: Sequence[np.ndarray],
    out_png: str | Path,
    *,
    n_grid: int = 100,
    label: str = "median F1 (16 cases)",
) -> None:
    """Median + IQR stability band across cases (D-11).

    Each entry of ``case_f1_curves`` is a per-frame F1 curve of (possibly)
    different length. Resample each onto a common ``n_grid``-point [0,1] grid
    with ``np.interp``, then plot the per-grid-point median with a
    25-75 percentile ``fill_between`` band. ``label`` names the quantity on the
    y-axis (never mix BCE-from-probs with binarized F1 on one axis -- Pitfall 4).
    Writes the figure to ``out_png`` (Agg, dpi=200).
    """
    raise NotImplementedError("Plan 05: D-11 median+IQR stability band")


# ---- threshold-sensitivity curve (D-12) ----
def threshold_curve(
    probs: np.ndarray,
    gts: np.ndarray,
    out_png: str | Path,
    *,
    thr_grid: Optional[np.ndarray] = None,
    calibrated_thr: Optional[float] = None,
):
    """Threshold-sensitivity sweep over saved probability stacks (D-12).

    Vectorized ``probs[None] >= thr_grid[:, None]`` sweep, applying the
    no-healing rule ``np.maximum.accumulate(pred, axis=time)`` per threshold,
    re-using ``aggregate.frame_f1_d03`` for the per-frame F1. Marks
    ``calibrated_thr`` (the Phase-1 value from ``calibration.json`` -- do NOT
    re-derive calibration here). Returns ``(thr_grid, macro_f1)`` arrays and
    writes the figure to ``out_png``.
    """
    raise NotImplementedError("Plan 05: D-12 threshold-sensitivity curve")


# ---- FP/FN panel (D-10/D-14) ----
def fpfn_rgb(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Dark-background FP/FN RGB overlay for one frame (D-10/D-14).

    Returns an ``(H, W, 3)`` uint8 image: gray ``COLOR_TP`` where ``gt & pred``,
    red ``COLOR_FP`` where ``pred & ~gt``, blue ``COLOR_FN`` where ``gt & ~pred``,
    black ``COLOR_BG`` elsewhere. Both inputs MUST already be in the same
    orientation (Pitfall 2).
    """
    raise NotImplementedError("Plan 05: D-10 FP/FN RGB overlay")


def select_key_frames(gt_stack: np.ndarray, pred_stack: np.ndarray) -> Dict[str, int]:
    """Pick curated diagnostic frame indices from a rollout (D-10).

    Returns ``{"onset", "mid", "late", "max_div"}`` -> frame indices: crack
    onset, mid-rollout, late-rollout, and the frame of maximum GT/pred
    divergence. Feeds the curated FP/FN panels in the report (D-14).
    """
    raise NotImplementedError("Plan 05: D-10 key-frame selection")
