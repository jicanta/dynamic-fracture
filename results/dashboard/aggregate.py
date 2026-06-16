"""Dashboard aggregation contracts: macro / micro / degenerate / late-rollout.

This module is the AGGREGATION layer over the frozen per-frame metric seam
(``frac_metrics.per_frame_metrics``). It NEVER forks the seam: macro F1 is
re-derived from the per-frame ``tp/fp/fn`` count columns the CSV already carries
(``evaluate.py:117``). The ONE deliberate divergence from the seam is the
degenerate convention -- ``frac_metrics.py:82-86`` resolves ``0/0 -> 0.0`` while
D-03 requires ``0/0 -> 1.0`` when the prediction is also empty (a correctly
predicted "nothing" frame). That divergence is the entire reason this lives in
the aggregation layer and not in the seam.

Decision IDs implemented here:
  * D-01 -- macro F1 = mean of the D-03-aware per-frame F1 (NOT the stored ``f1``).
  * D-02 -- micro F1 secondary column (delegates to ``micro_metrics.micro``).
  * D-03 -- degenerate convention: empty-GT + empty-pred -> F1=P=R=IoU=1.0;
            empty-GT + non-empty-pred -> 0.0.
  * D-04 -- late-rollout window: per-case last 20% (k=max(1,round(0.2n))).
  * D-05/D-06 -- single val-only selection metric seam (Phase 3/4 imports this).

Framework rule (D-13): numpy / pandas / stdlib ONLY. NO torch, NO tensorflow.

This is a Wave-0 CONTRACT module: every public function raises
``NotImplementedError``. Plan 02 fills in the bodies and turns the RED tests
(``tests/test_dashboard_aggregate.py``) GREEN.

Run:
    cd dynamic-fracture && python -c "import sys; sys.path.insert(0,'results'); \
from dashboard.aggregate import frame_f1_d03; print(frame_f1_d03)"
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

# ---- repo-root sys.path shim ----
# dashboard -> results -> dynamic-fracture; makes frac_metrics / case_registry /
# results.scripts.micro_metrics importable from this subdir.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---- per-frame (D-03-aware) ----
def frame_f1_d03(tp: float, fp: float, fn: float) -> float:
    """Degenerate-aware per-frame F1 (D-03).

    Differs from ``frac_metrics`` ONLY on the all-zero frame: when
    ``tp + fp + fn == 0`` (empty GT AND empty pred) this returns ``1.0`` (the
    model correctly predicted nothing), whereas the frozen seam returns ``0.0``.
    All other branches match the seam's foreground-only F1.
    """
    raise NotImplementedError("Plan 02: D-03-aware per-frame F1 from counts")


def frame_metrics_d03(tp: float, fp: float, fn: float) -> Dict[str, float]:
    """Degenerate-aware per-frame precision/recall/f1/iou (D-03).

    Same ``0/0 -> 1.0`` rule as :func:`frame_f1_d03`, applied to each of
    precision, recall, f1, iou when the frame is all-zero.
    Returns ``{"precision", "recall", "f1", "iou"}``.
    """
    raise NotImplementedError("Plan 02: D-03-aware per-frame metric dict")


# ---- macro aggregation (D-01) ----
def macro_f1_from_counts(rows: Iterable[Dict[str, float]]) -> float:
    """Macro F1 = mean over frames of :func:`frame_f1_d03` (D-01).

    ``rows`` is an iterable of per-frame dicts carrying ``tp/fp/fn`` count keys
    (as read from a ``per_frame_metrics.csv``). The stored ``f1`` column is
    intentionally NOT used -- macro F1 is recomputed from counts under the D-03
    convention.
    """
    raise NotImplementedError("Plan 02: D-01 macro F1 = mean of frame_f1_d03")


def macro_metrics_from_csv(path: str | Path) -> Dict[str, float]:
    """Full aggregate summary for one case's ``per_frame_metrics.csv``.

    Returns ``{macro_f1, micro_f1, precision, recall, iou, frames}`` where
    ``macro_f1`` is the D-01/D-03 mean and ``micro_f1`` delegates to
    :func:`micro_f1_from_csv` (D-02). Reads counts via ``csv.DictReader`` so the
    canonical schema's extra provenance columns are ignored
    (mirrors ``micro_metrics.py:21-34``).
    """
    raise NotImplementedError("Plan 02: macro+micro summary from one CSV")


def micro_f1_from_csv(path: str | Path) -> float:
    """Micro F1 over all frames of one case CSV (D-02).

    Delegates to ``micro_metrics.micro`` (reuse, do NOT re-implement micro):
    sums tp/fp/fn over every frame then computes F1 from the totals.
    """
    raise NotImplementedError("Plan 02: D-02 micro F1 via micro_metrics.micro")


# ---- late-rollout window (D-04) + selection seam (D-05/D-06) ----
def late_rollout_macro_f1(rows: Sequence[Dict[str, float]], frac: float = 0.20) -> float:
    """Macro F1 over the last ``frac`` of one case's frames (D-04).

    Window size ``k = max(1, round(frac * n))`` frames taken from the END of the
    rollout; macro F1 (D-01/D-03) is computed within that window. Captures
    long-horizon stability where the autoregressive rollout degrades most.
    """
    raise NotImplementedError("Plan 02: D-04 per-case last-20% macro F1")


def late_rollout_selection_metric(val_csv_paths: Iterable[str | Path]) -> float:
    """The ONE val-only model-selection scalar (D-05/D-06).

    Given the val-set per-case ``per_frame_metrics.csv`` paths, computes the
    per-case late-rollout macro F1 (:func:`late_rollout_macro_f1`) then averages
    ACROSS cases, returning a single float. Inputs are VALIDATION CSVs only --
    test-set CSVs must never reach this function (no selection-time leakage).
    This is the stable seam Phase 3/4 imports for sweep/checkpoint selection.
    """
    raise NotImplementedError("Plan 02: D-05/D-06 val-only selection metric")
