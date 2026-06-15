"""Single shared per-frame metric implementation for BOTH pipelines (D-11 seam).

This module is the one place per-frame fracture-segmentation metrics are
computed. The new PyTorch pipeline (``new_model/src/evaluate.py``) and the
reference TensorFlow/Keras pipeline
(``kathleens-model/source/predict_full_simulation_metrics.py``) both route their
per-frame emission through ``per_frame_metrics`` so the golden-input parity test
(CMP-01 / D-12) is a near-tautology: there is only one implementation, hence no
duplicated-metric baseline leak (threat T-03-01).

It consolidates three previously-duplicated code paths:
  * ``new_model/src/evaluate.py`` ``frame_metrics`` (numpy counts — the keeper),
  * ``kathleens-model/source/metrics.py`` ``metrics_from_binary`` (sklearn
    ``confusion_matrix`` path — retired in favor of these numpy counts),
  * ``kathleens-model/source/predict_full_simulation_metrics.py``
    ``compute_bce_per_frame`` (``eps = 1e-7``),
and ADDS IoU (D-13), which is computed nowhere in the legacy code.

Framework rule (hard): this module imports numpy + stdlib ONLY -- no PyTorch and
no TF/Keras -- which is what lets the same numbers be produced from either
pipeline.

References (provenance style): SimVP (CVPR 2022), TAU "Temporal Attention Unit"
(CVPR 2023) — see the FractureTAU model. This file itself contains no model
logic; it is the comparison-harness metric seam.

Run (smoke):
    cd dynamic-fracture && python -c "import frac_metrics as m; print(m.CANONICAL_COLUMNS)"
"""

from __future__ import annotations

import numpy as np
from typing import Dict

# ---- constants ----
# BCE probability clamp. MUST stay 1e-7 — both pipelines already agree on this
# value (new: evaluate.py:45; old: predict_full_simulation_metrics.py). A
# mismatch here silently shifts the BCE curves (threat T-03-02).
EPS: float = 1e-7

# Canonical per-frame CSV column order (D-13). Both pipelines emit these columns,
# in this order, with these exact names; extra provenance columns are allowed but
# these six must always be present. ``f1`` is foreground-only. ``frame_idx`` is
# supplied by each pipeline's emission loop, not by per_frame_metrics.
CANONICAL_COLUMNS = ("frame_idx", "precision", "recall", "f1", "iou", "bce")


# ---- per-frame metrics ----
def per_frame_metrics(
    gt_bin: np.ndarray,
    pred_bin: np.ndarray,
    prob: np.ndarray,
) -> Dict[str, float]:
    """Per-frame binary segmentation metrics for one frame.

    Args:
        gt_bin:   ground-truth binary fracture mask, any shape; values in {0, 1}.
        pred_bin: predicted binary mask, same shape; values in {0, 1}.
        prob:     raw predicted probability map (NOT binarized), values in [0, 1],
                  used only for the BCE term (BCE is computed from raw probs).

    Returns:
        dict with keys ``precision, recall, f1, iou, bce, tp, tn, fp, fn``
        (all floats). ``f1`` is foreground-only. The degenerate 0/0 convention
        resolves to ``0.0`` (never NaN): an all-zero gt/pred frame yields
        ``precision = recall = f1 = iou = 0.0``.

    Determinism: no RNG, pure arithmetic — identical inputs yield byte-identical
    outputs (CMP-01).
    """
    # flatten + uint8 binarize internally so callers may pass any shape / dtype
    g = gt_bin.reshape(-1).astype(np.uint8)
    p = pred_bin.reshape(-1).astype(np.uint8)

    # ---- confusion counts (copied from new_model/src/evaluate.py:48-58) ----
    tp = float(np.sum((p == 1) & (g == 1)))
    tn = float(np.sum((p == 0) & (g == 0)))
    fp = float(np.sum((p == 1) & (g == 0)))
    fn = float(np.sum((p == 0) & (g == 1)))

    # ---- precision / recall / f1 / iou (0/0 -> 0.0 convention preserved) ----
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0  # NEW (D-13)

    # ---- BCE from raw probs (eps=1e-7 clamp; predict_full_simulation_metrics.py:53-59) ----
    pr = np.clip(prob.reshape(-1), EPS, 1.0 - EPS)
    gt = gt_bin.reshape(-1).astype(np.float64)
    bce = float(-(gt * np.log(pr) + (1.0 - gt) * np.log(1.0 - pr)).mean())

    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "iou": iou,
        "bce": bce,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
