"""Probability calibration for the physical-metrics paper: reliability + ECE.

This module measures whether FractureTAU's predicted fracture PROBABILITIES are
calibrated — i.e. whether, among pixels predicted at confidence ``p``, a fraction
``p`` are truly fractured. It implements the binned Expected Calibration Error
(Guo et al. 2017, "On Calibration of Modern Neural Networks") and the matching
reliability diagram over a rollout, computed from the saved raw probability
stacks (``probs.npz``) — NO Gilbreth dependency for the math.

DISTINCT from threshold calibration: this is NOT
``new_model/src/train.py::calibrate_threshold`` / ``tests/test_calibration.py``,
which pick a decision THRESHOLD on the val set. This module is about probability
reliability and is binned by predicted probability, never by a binarized mask.

Implements:
  * PHYS-05 -- reliability diagram + ECE over the rollout, from saved raw probs.

Pitfall 5 (hard): bins are formed by PREDICTED PROBABILITY (the raw fp32 values
from ``probs.npz``), NEVER by binarizing the mask first — binarizing collapses
the very confidence distribution calibration is supposed to inspect. For the
monotone-Δ head the fed-back probability is bounded (>= the no-healing input
mask), so its raw probs are the correct calibration target here too.

Security / load contract (S6, threat T-05-09 Tampering): probability stacks are
read ONLY through ``dashboard.probs_io.load_case_probs_gt`` (which enforces
``np.load(..., allow_pickle=False)``). This module NEVER hand-rolls
``np.load`` of ``probs.npz`` — no pickle surface is opened.

Framework rule (S1): numpy / stdlib ONLY for the math (matplotlib lazy-imported,
Agg, for the optional plot). NO torch, NO tensorflow.

Run:
    cd dynamic-fracture && python -c "import sys; sys.path.insert(0,'results'); \
import numpy as np; from phys_metrics.calibration import reliability_ece; \
p=np.array([0.1,0.9,0.6]); g=np.array([0,1,1]); print(reliability_ece(p,g))"
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ---- vector-PDF font embedding (D-06): Type-42 (TrueType), never Type-3 ----
# Set once at import so the reliability-diagram .pdf savefig embeds editable fonts
# (matplotlib defaults to publisher-rejected Type-3). Does NOT select a backend,
# so the lazy ``matplotlib.use("Agg")`` inside ``reliability_diagram`` still wins.
import matplotlib as mpl
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# ---- repo-root sys.path shim (S3) ----
# calibration.py -> phys_metrics -> results -> dynamic-fracture; add results/ so
# ``dashboard.probs_io`` (the allow_pickle=False seam, S6) imports cleanly.
RESULTS = Path(__file__).resolve().parents[1]
if str(RESULTS) not in sys.path:
    sys.path.insert(0, str(RESULTS))

# Re-exported so callers source probs through the safe seam (S6) — never
# hand-roll np.load of probs.npz (threat T-05-09).
from dashboard.probs_io import load_case_probs_gt  # noqa: E402,F401


# ---- per-bin statistics (binned by PREDICTED PROBABILITY, Pitfall 5) ----
def _bin_stats(
    probs: np.ndarray, gts: np.ndarray, n_bins: int
) -> Tuple[List[Tuple[float, float, int]], int]:
    """Return ``(records, N)`` where ``records[i] = (conf, acc, count)`` per bin.

    Bins partition ``[0, 1]`` into ``n_bins`` equal-width intervals; a sample is
    assigned by its PREDICTED PROBABILITY (lo-exclusive except the first bin is
    lo-inclusive, hi-inclusive — so 0.0 lands in bin 0 and 1.0 in the last bin).
    ``conf`` = mean predicted probability in the bin, ``acc`` = empirical positive
    fraction (``gt`` mean); an EMPTY bin yields ``(mid, np.nan, 0)``.
    """
    probs = probs.ravel().astype(np.float64)
    gts = gts.ravel().astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    records: List[Tuple[float, float, int]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        # bin by predicted PROBABILITY, not on a binarized mask (Pitfall 5)
        m = (probs > lo) & (probs <= hi) if lo > 0 else (probs >= lo) & (probs <= hi)
        count = int(m.sum())
        if count == 0:
            records.append((0.5 * (lo + hi), float("nan"), 0))
        else:
            records.append((float(probs[m].mean()), float(gts[m].mean()), count))
    return records, probs.size


# ---- PHYS-05: Expected Calibration Error + per-bin populations ----
def reliability_ece(
    probs: np.ndarray, gts: np.ndarray, n_bins: int = 15
) -> Tuple[float, List[int]]:
    """Binned Expected Calibration Error (Guo et al. 2017).

    ``probs`` / ``gts`` are array-likes (any shape; raveled). Returns
    ``(ece, populations)`` where ``ece = sum_b (count_b/N) * |acc_b - conf_b|``
    over non-empty bins, and ``populations`` is the per-bin sample count.

    NOTE on the return shape: the second element is the per-bin POPULATION list
    (counts), so ``sum(populations) == probs.size`` exactly — no sample dropped or
    double-counted (the contract pinned by ``test_calibration_ece.py``). The
    richer per-bin ``(conf, acc)`` records used to DRAW the reliability diagram
    are produced by ``_bin_stats`` / :func:`reliability_diagram`.
    """
    records, N = _bin_stats(probs, gts, n_bins)
    ece = 0.0
    populations: List[int] = []
    for conf, acc, count in records:
        populations.append(count)
        if count == 0:
            continue
        ece += (count / N) * abs(acc - conf)
    return ece, populations


# ---- optional reliability-diagram plot (lazy Agg, S7) ----
def reliability_diagram(
    probs: np.ndarray, gts: np.ndarray, out_png: str | Path, n_bins: int = 15
) -> Tuple[float, List[int]]:
    """Draw the reliability diagram (acc vs. conf) and return ``(ece, populations)``.

    Plots per-bin empirical accuracy against mean confidence with the ``y = x``
    perfect-calibration diagonal; mirrors ``dashboard.plots.threshold_curve``'s
    lazy-Agg / mkdir / savefig(dpi=200) / close idiom (S7). Returns the same
    ``(ece, populations)`` as :func:`reliability_ece` so callers (and tests) can
    assert WITHOUT reading the PNG back.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records, _N = _bin_stats(probs, gts, n_bins)
    ece, populations = reliability_ece(probs, gts, n_bins)

    confs = [c for (c, a, n) in records if n > 0]
    accs = [a for (c, a, n) in records if n > 0]

    out_png = Path(out_png)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca()
    ax.plot([0, 1], [0, 1], ls="--", color="gray", label="perfect calibration")
    ax.plot(confs, accs, "o-", lw=2, label=f"FractureTAU (ECE = {ece:.3f})")
    ax.set_xlabel("mean predicted probability (confidence)", fontsize=14)
    ax.set_ylabel("empirical fracture fraction (accuracy)", fontsize=14)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if str(out_png).lower().endswith(".pdf"):
        fig.savefig(out_png, format="pdf")   # vector; dpi only affects raster insets
    else:
        fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return ece, populations
