"""CMP-03 -- canonical per-frame CSV schema (D-13).

Pins the single canonical column order both pipelines must emit and proves the
shared metric function actually provides every canonical field (except
``frame_idx``, which each pipeline's emission loop supplies, not the metric fn).
A drift in column names/order between the two pipelines is the schema-level form
of the duplicated-metric baseline leak; this test makes such a drift impossible
to merge silently.

Run:  cd dynamic-fracture && python -m pytest tests/test_schema.py -q
"""

from __future__ import annotations

# imported via the conftest.py repo-root sys.path shim
import frac_metrics


# ---- the canonical column tuple is pinned exactly ----
def test_canonical_columns():
    assert frac_metrics.CANONICAL_COLUMNS == (
        "frame_idx",
        "precision",
        "recall",
        "f1",
        "iou",
        "bce",
    )


# ---- the metric fn provides every canonical field except frame_idx ----
def test_metrics_provide_canonical_fields(golden_masks):
    """Every canonical column other than ``frame_idx`` must be a key returned by
    per_frame_metrics (``frame_idx`` is added by the per-frame emission loop)."""
    gt_bin, pred_bin, prob = golden_masks
    out = frac_metrics.per_frame_metrics(gt_bin, pred_bin, prob)

    for col in frac_metrics.CANONICAL_COLUMNS:
        if col == "frame_idx":
            continue
        assert col in out, f"canonical column {col!r} missing from per_frame_metrics output"
