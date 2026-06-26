"""D-06 paper figure generators for the physical-metrics analysis.

This module turns the Wave-2 ``phys_metrics`` quantities into the actual paper
figures (D-06), all rendered headless (lazy ``matplotlib.use("Agg")`` inside the
function body, S7: ``figsize=(11,6)``, ``grid(alpha=0.3)``, ``mkdir(parents)``
before save, ``dpi=200``, ``plt.close(fig)``). The figures here are framed as
INTEGRATED physical-fidelity evidence (D-13), not a separate/appendix-only
section: PHYS-04 (TF-vs-AR gap) and PHYS-05 (reliability/ECE) sit beside the
physical-metric plots.

Figures produced:
  * ``length_over_time_fig`` -- crack length-over-time, pred vs GT, with the
    time-to-onset frame marked for each (PHYS-03, D-02).
  * ``f1_vs_horizon_fig``    -- per-frame F1 over the rollout horizon for the
    autoregressive (AR) and teacher-forced (TF) passes, with the AR-vs-TF gap
    shaded (PHYS-04, D-13).
  * ``reliability_fig``      -- reliability diagram + ECE over the rollout
    (PHYS-05, D-13).
  * the rollout-stability band and the FP/FN qualitative panels are NOT
    re-implemented here -- they are REUSED directly from
    ``dashboard.plots.stability_band`` / ``dashboard.plots.save_fpfn_panel``
    (imported + re-exported below; thin wrappers only).

Reuse-by-import (S4): the F1 curve / gap / crack-length / calibration math is
NEVER re-derived here -- it is imported from ``dashboard.plots`` (the D-03 F1
curve), ``phys_metrics.error_accum`` (the TF-vs-AR gap), ``phys_metrics.physical``
(crack length + onset) and ``phys_metrics.calibration`` (reliability/ECE).

Framework rule (S1): numpy / matplotlib only -- no torch, no tensorflow (these
figure helpers live on the numpy side of the hard cross-framework boundary; a
grep gate enforces it).

Run:
    cd dynamic-fracture && python -c "import sys; sys.path.insert(0,'results'); \
import numpy as np; from phys_metrics.figures import length_over_time_fig; \
s=np.zeros((4,8,8),np.uint8); s[:,0,:]=1; \
print(length_over_time_fig(s, s, '/tmp/len.png'))"
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

# ---- repo-root + results sys.path shim (S3) ----
# figures.py -> phys_metrics -> results -> dynamic-fracture; add results/ so the
# sibling ``dashboard`` package and the ``phys_metrics`` Wave-2 modules resolve
# whether imported as ``phys_metrics.figures`` or run as a bare script.
RESULTS = Path(__file__).resolve().parents[1]            # dynamic-fracture/results
REPO_ROOT = RESULTS.parent                               # dynamic-fracture
for _p in (str(REPO_ROOT), str(RESULTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---- reuse-by-import (S4): never re-derive these quantities here ----
from dashboard.plots import (  # noqa: E402
    case_f1_curve_from_rows,    # per-frame D-03 F1 curve from count rows
    save_fpfn_panel,            # FP/FN qualitative panel (reused, not rebuilt)
    stability_band,             # rollout-stability median+IQR band (reused)
)
from phys_metrics.calibration import reliability_diagram  # noqa: E402  (PHYS-05)
from phys_metrics.error_accum import tf_vs_ar_gap  # noqa: E402  (PHYS-04)
from phys_metrics.physical import (  # noqa: E402  (PHYS-03)
    crack_length_curve,
    time_to_onset,
)

# Re-export the reused dashboard helpers so the phys-report driver can pull the
# stability band + FP/FN panels from one figure module (thin wrappers only).
__all__ = [
    "length_over_time_fig",
    "f1_vs_horizon_fig",
    "reliability_fig",
    "stability_band",
    "save_fpfn_panel",
]


# ---- PHYS-03: crack length-over-time + onset ----
def length_over_time_fig(
    pred_stack: np.ndarray,
    gt_stack: np.ndarray,
    out_png: str | Path,
    *,
    pixel_pitch: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Crack length-over-time, prediction vs GT, with onset marked (PHYS-03/D-02).

    ``pred_stack`` / ``gt_stack`` are ``(n_frames, H, W)`` ALREADY-binarized mask
    stacks (binarize upstream at the val-calibrated threshold, Pitfall 3). The
    per-frame skeleton-pixel length curve is computed via
    :func:`phys_metrics.physical.crack_length_curve` (reuse, S4) for both stacks,
    and the time-to-fracture-onset frame (:func:`physical.time_to_onset`) is drawn
    as a vertical marker for each. Returns ``(pred_curve, gt_curve)`` so callers
    assert WITHOUT reading the PNG back (S7).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred_curve = crack_length_curve(pred_stack, pixel_pitch=pixel_pitch)
    gt_curve = crack_length_curve(gt_stack, pixel_pitch=pixel_pitch)
    onset_pred = time_to_onset(pred_stack)
    onset_gt = time_to_onset(gt_stack)

    frames = np.arange(len(gt_curve))
    ylab = "crack length (px)" if pixel_pitch == 1.0 else "crack length (phys. units)"

    out_png = Path(out_png)
    fig = plt.figure(figsize=(11, 6))
    ax = fig.gca()
    ax.plot(frames, gt_curve, lw=2, color="black", label="ground truth")
    ax.plot(np.arange(len(pred_curve)), pred_curve, lw=2, color="tab:red",
            label="FractureTAU (AR)")
    if onset_gt >= 0:
        ax.axvline(onset_gt, ls="--", color="black", alpha=0.6,
                   label=f"GT onset (frame {onset_gt})")
    if onset_pred >= 0:
        ax.axvline(onset_pred, ls=":", color="tab:red", alpha=0.8,
                   label=f"pred onset (frame {onset_pred})")
    ax.set_xlabel("rollout frame", fontsize=16)
    ax.set_ylabel(ylab, fontsize=16)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return pred_curve, gt_curve


# ---- PHYS-04: TF-vs-AR F1 over the rollout horizon ----
def f1_vs_horizon_fig(
    ar_rows: Sequence[Dict[str, float]],
    tf_rows: Sequence[Dict[str, float]],
    out_png: str | Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame F1 over the horizon for AR vs TF, with the gap shaded (PHYS-04).

    ``ar_rows`` / ``tf_rows`` are equal-length sequences of per-frame count rows
    (``tp/fp/fn`` keys, as read from ``per_frame_metrics.csv`` /
    ``per_frame_metrics_tf.csv``). The two D-03-aware F1 curves are built via
    :func:`dashboard.plots.case_f1_curve_from_rows` (reuse, S4 -- never the stored
    ``f1`` column) and the per-frame gap via
    :func:`phys_metrics.error_accum.tf_vs_ar_gap` (``gap = f1_tf - f1_ar``). The
    error-accumulation narrative (D-13): the gap grows as the AR rollout drifts
    from the teacher-forced pass over the horizon. Returns
    ``(ar_curve, tf_curve, gap)`` (S7).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ar_curve = case_f1_curve_from_rows(ar_rows)
    tf_curve = case_f1_curve_from_rows(tf_rows)
    gap = tf_vs_ar_gap(ar_rows, tf_rows)
    frames = np.arange(len(ar_curve))

    out_png = Path(out_png)
    fig = plt.figure(figsize=(11, 6))
    ax = fig.gca()
    ax.plot(frames, tf_curve, lw=2, color="tab:blue", label="teacher-forced (TF)")
    ax.plot(frames, ar_curve, lw=2, color="tab:red", label="autoregressive (AR)")
    ax.fill_between(frames, ar_curve, tf_curve, where=(tf_curve >= ar_curve),
                    alpha=0.25, color="tab:purple", label="TF - AR gap")
    ax.set_xlabel("rollout frame (horizon)", fontsize=16)
    ax.set_ylabel("foreground F1", fontsize=16)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return ar_curve, tf_curve, gap


# ---- PHYS-05: reliability diagram + ECE ----
def reliability_fig(
    probs: np.ndarray,
    gts: np.ndarray,
    out_png: str | Path,
    n_bins: int = 15,
) -> Tuple[float, List[int]]:
    """Reliability diagram + ECE over the rollout (PHYS-05/D-13).

    Thin wrapper around :func:`phys_metrics.calibration.reliability_diagram`
    (reuse, S4): bins are formed by PREDICTED PROBABILITY (never a binarized mask,
    Pitfall 5). ``probs`` / ``gts`` are array-likes (raveled inside). Returns
    ``(ece, populations)`` so callers assert WITHOUT reading the PNG back (S7).
    """
    return reliability_diagram(probs, gts, out_png, n_bins=n_bins)
