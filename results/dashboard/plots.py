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

    OQ3 (LOCKED): the x-axis is the NORMALIZED rollout fraction [0, 1] -- each
    variable-length case is resampled onto a common grid via ``np.interp`` so the
    median+IQR band is computed across cases at matched rollout fractions, never
    absolute frame index. Returns ``(grid, med, lo, hi)`` so callers/tests can
    assert shape without re-reading the PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_png = Path(out_png)
    curves = [np.asarray(c, dtype=float).ravel() for c in case_f1_curves]
    if not curves:
        raise ValueError("stability_band: case_f1_curves is empty")

    # Resample each case onto a common normalized rollout-fraction grid (OQ3).
    grid = np.linspace(0.0, 1.0, n_grid)
    resampled = np.vstack(
        [np.interp(grid, np.linspace(0.0, 1.0, len(c)), c) for c in curves]
    )
    # Median + IQR band across the cases (D-11): 25/50/75 percentiles, NOT mean+/-std.
    lo, med, hi = np.percentile(resampled, [25, 50, 75], axis=0)

    fig = plt.figure(figsize=(11, 6))
    ax = fig.gca()
    ax.plot(grid, med, lw=2, label=label)
    ax.fill_between(grid, lo, hi, alpha=0.3, label="IQR (25-75%)")
    ax.set_xlabel("rollout fraction", fontsize=16)
    ax.set_ylabel("foreground F1", fontsize=16)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return grid, med, lo, hi


def case_f1_curve_from_rows(rows: Sequence[Dict[str, float]]) -> np.ndarray:
    """Per-frame D-03-aware macro-F1 curve from a case's raw count rows.

    Maps each ``{tp, fp, fn, ...}`` row to :func:`aggregate.frame_f1_d03` so the
    resulting 1-D curve is a binarized-F1-over-time series ready to hand to
    :func:`stability_band` (never the stored ``f1`` column -- Pitfall 1).
    """
    from dashboard.aggregate import frame_f1_d03

    return np.array(
        [frame_f1_d03(float(r["tp"]), float(r["fp"]), float(r["fn"])) for r in rows],
        dtype=float,
    )


# ---- threshold-sensitivity curve (D-12) ----
def threshold_curve(
    probs,
    gts,
    out_png: str | Path,
    *,
    thr_grid: Optional[np.ndarray] = None,
    calibrated_thr: Optional[float] = None,
    no_healing: bool = True,
):
    """Threshold-sensitivity sweep over saved probability stacks (D-12).

    ``probs`` / ``gts`` are lists of per-case ``(n_frames, H, W)`` arrays (load
    via :func:`probs_io.load_case_probs_gt` for real data; tests pass synthetic
    stacks). For each threshold the prediction is ``(prob >= thr)``; when
    ``no_healing`` the per-case stack is monotonically accumulated over time via
    ``np.maximum.accumulate(pred, axis=0)`` (state = max(state, pred)), the exact
    rollout no-healing rule. Per-frame F1 re-uses ``aggregate.frame_f1_d03``
    (D-03-aware), is macro-averaged within a case, then averaged across cases.

    This is FractureTAU-ONLY (OQ4): the reference is pre-binarized / fixed-
    threshold and is NEVER re-rolled here. ``calibrated_thr`` is the Phase-1 value
    read from ``calibration.json`` (see :func:`read_calibrated_threshold`) -- do
    NOT re-derive calibration. Returns ``(thr_grid, macro_f1)`` arrays and writes
    the figure to ``out_png``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from dashboard.aggregate import frame_f1_d03

    out_png = Path(out_png)
    if thr_grid is None:
        thr_grid = np.arange(0.01, 1.00, 0.01)  # fine ~0.01 grid (D-12)
    thr_grid = np.asarray(thr_grid, dtype=float)

    # Normalize to a list-of-cases (each case is a (n_frames, H, W) stack).
    # A bare 3-D ndarray is a SINGLE case; a list/sequence is many cases.
    def _as_cases(x):
        if isinstance(x, np.ndarray) and x.ndim == 3:
            return [x]
        return list(x)

    prob_cases = _as_cases(probs)
    gt_cases = _as_cases(gts)
    if len(prob_cases) != len(gt_cases):
        raise ValueError("threshold_curve: probs/gts case-count mismatch")

    macro_f1 = np.zeros(thr_grid.shape, dtype=float)
    for ti, thr in enumerate(thr_grid):
        case_scores = []
        for prob, gt in zip(prob_cases, gt_cases):
            prob = np.asarray(prob, dtype=np.float32)
            gt = (np.asarray(gt) >= 1).astype(np.uint8)
            pred = (prob >= thr).astype(np.uint8)
            if no_healing:
                pred = np.maximum.accumulate(pred, axis=0)  # state = max(state, pred)
            # Per-frame D-03 F1, macro within the case.
            frame_f1s = []
            for f in range(pred.shape[0]):
                p = pred[f]
                g = gt[f]
                tp = int(((g == 1) & (p == 1)).sum())
                fp = int(((g == 0) & (p == 1)).sum())
                fn = int(((g == 1) & (p == 0)).sum())
                frame_f1s.append(frame_f1_d03(tp, fp, fn))
            case_scores.append(float(np.mean(frame_f1s)))
        macro_f1[ti] = float(np.mean(case_scores))

    fig = plt.figure(figsize=(11, 6))
    ax = fig.gca()
    ax.plot(thr_grid, macro_f1, lw=2, label="FractureTAU macro F1")
    if calibrated_thr is not None:
        ax.axvline(
            calibrated_thr, ls="--", color="gray",
            label=f"Phase-1 calibrated thr = {calibrated_thr:.3f}",
        )
    ax.set_xlabel("threshold", fontsize=16)
    ax.set_ylabel("macro foreground F1", fontsize=16)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return thr_grid, macro_f1


def read_calibrated_threshold(run_dir: str | Path) -> Optional[float]:
    """Read the Phase-1 calibrated threshold from ``run_dir/calibration.json``.

    Returns the ``["threshold"]`` float, or ``None`` if the file is absent. Does
    NOT recompute calibration -- the threshold curve only MARKS the Phase-1 value
    (D-12 reuse).
    """
    import json

    cal = Path(run_dir) / "calibration.json"
    if not cal.exists():
        return None
    with open(cal) as fh:
        data = json.load(fh)
    thr = data.get("threshold")
    return float(thr) if thr is not None else None


# ---- FP/FN panel (D-10/D-14) ----
def fpfn_rgb(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Dark-background FP/FN RGB overlay for one frame (D-10/D-14).

    Returns an ``(H, W, 3)`` uint8 image: gray ``COLOR_TP`` where ``gt & pred``,
    red ``COLOR_FP`` where ``pred & ~gt``, blue ``COLOR_FN`` where ``gt & ~pred``,
    black ``COLOR_BG`` elsewhere. Both inputs MUST already be in the same
    orientation (Pitfall 2): the caller passes GT from probs_io and the pred
    binarized from the SAME saved probs -- never a flipud'd PNG against an
    unflipped GT array.
    """
    gt = (np.asarray(gt) >= 1)
    pred = (np.asarray(pred) >= 1)
    H, W = gt.shape
    rgb = np.zeros((H, W, 3), np.uint8)            # black background (COLOR_BG)
    tp = gt & pred                                 # gt==1 & pred==1
    fp = pred & ~gt                                # pred==1 & gt==0
    fn = gt & ~pred                                # gt==1 & pred==0
    rgb[tp] = COLOR_TP                             # (160,160,160) gray
    rgb[fp] = COLOR_FP                             # (220,40,40) red
    rgb[fn] = COLOR_FN                             # (40,90,220) blue
    return rgb


def select_key_frames(gt_stack: np.ndarray, pred_stack: np.ndarray) -> Dict[str, int]:
    """Pick curated diagnostic frame indices from a rollout (D-10).

    Returns ``{"onset", "mid", "late", "max_div"}`` -> frame indices:
      * ``onset``   -- first frame with any GT foreground.
      * ``mid``     -- ``n // 2`` (mid-rollout).
      * ``late``    -- start of the last-20% window: ``n - max(1, round(0.20*n))``.
      * ``max_div`` -- argmax over frames of the (FP + FN) pixel count
                       (max pred-vs-GT divergence).
    Feeds the curated FP/FN panels in the report (D-14).
    """
    gt_stack = (np.asarray(gt_stack) >= 1).astype(np.uint8)
    pred_stack = (np.asarray(pred_stack) >= 1).astype(np.uint8)
    n = int(gt_stack.shape[0])
    if n == 0:
        raise ValueError("select_key_frames: empty stack")

    # onset: first frame with any GT foreground (default to 0 if never).
    fg_per_frame = gt_stack.reshape(n, -1).sum(axis=1)
    nz = np.nonzero(fg_per_frame)[0]
    onset = int(nz[0]) if nz.size else 0

    mid = n // 2
    late = n - max(1, round(0.20 * n))

    # max_div: frame of maximum (FP + FN) pixel count.
    fp = ((pred_stack == 1) & (gt_stack == 0)).reshape(n, -1).sum(axis=1)
    fn = ((gt_stack == 1) & (pred_stack == 0)).reshape(n, -1).sum(axis=1)
    max_div = int(np.argmax(fp + fn))

    return {"onset": onset, "mid": int(mid), "late": int(late), "max_div": max_div}


def save_fpfn_panel(
    out_png: str | Path,
    gt_stack: np.ndarray,
    pred_stack: np.ndarray,
    frames: Sequence[int],
    *,
    titles: Optional[Sequence[str]] = None,
) -> None:
    """Multi-frame FP/FN diagnostic panel on a dark figure (D-10/D-14).

    Renders :func:`fpfn_rgb` for each requested frame side-by-side (Agg idiom
    from ``utils.py``: ``mkdir(parents=True)``, ``dpi=200``, ``plt.close``). The
    all-16 per-case panels go under ``results/diagnostics/``; the curated subset
    is embedded by the report. GT and pred MUST share one orientation (Pitfall 2).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_png = Path(out_png)
    gt_stack = np.asarray(gt_stack)
    pred_stack = np.asarray(pred_stack)
    frames = list(frames)
    n = max(1, len(frames))

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), facecolor="black")
    if n == 1:
        axes = [axes]
    for i, f in enumerate(frames):
        ax = axes[i]
        ax.imshow(fpfn_rgb(gt_stack[f], pred_stack[f]), interpolation="nearest")
        title = titles[i] if titles is not None and i < len(titles) else f"frame {f}"
        ax.set_title(title, color="white", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, facecolor="black")
    plt.close(fig)
