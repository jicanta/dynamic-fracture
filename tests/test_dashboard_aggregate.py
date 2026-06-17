# tests/test_dashboard_aggregate.py
"""METR-01 / METR-03: dashboard aggregation contracts (macro / degenerate / late).

Pins the LOCKED conventions the Plan-02 implementation must satisfy:
  * METR-01      -- macro F1 = mean of the D-03-aware per-frame F1 from counts
                    (NOT the stored ``f1`` column).
  * METR-01/D-03 -- empty-GT + empty-pred frame -> F1=P=R=IoU=1.0;
                    empty-GT + non-empty-pred -> 0.0.
  * METR-03/D-04 -- late-rollout window = ``k = max(1, round(0.2 * n))`` frames.
  * METR-03/D-05/06 -- ``late_rollout_selection_metric(val_csvs)`` returns one
                    float (val-only inputs, no test-set leakage).

Wave-0 state: the ``aggregate`` bodies raise ``NotImplementedError``, so these
tests are RED (a runtime failure, NOT a collection/import error). Plan 02 turns
them GREEN.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

# ---- sys.path shim: repo root + results + results/dashboard ----
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
DASHBOARD = RESULTS / "dashboard"
for _p in (str(REPO_ROOT), str(RESULTS), str(DASHBOARD)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dashboard.aggregate import (  # noqa: E402
    frame_f1_d03,
    frame_metrics_d03,
    late_rollout_macro_f1,
    late_rollout_selection_metric,
    macro_f1_from_counts,
    macro_metrics_from_csv,
)
from dashboard import make_report  # noqa: E402


def test_macro_from_counts():
    # Three frames: all-zero (D-03 -> 1.0), perfect (1.0), half (0.5).
    rows = [
        {"tp": 0, "fp": 0, "fn": 0},     # empty GT + empty pred -> D-03 F1 = 1.0
        {"tp": 10, "fp": 0, "fn": 0},    # perfect -> F1 = 1.0
        {"tp": 5, "fp": 5, "fn": 5},     # prec=rec=0.5 -> F1 = 0.5
    ]
    expected = (1.0 + 1.0 + 0.5) / 3.0
    assert macro_f1_from_counts(rows) == pytest.approx(expected)


def test_degenerate_convention(golden_degenerate):
    gt, pred, _prob = golden_degenerate
    tp = int(((gt == 1) & (pred == 1)).sum())   # 0
    fp = int(((gt == 0) & (pred == 1)).sum())   # 0
    fn = int(((gt == 1) & (pred == 0)).sum())   # 0
    # empty GT + empty pred -> everything resolves to 1.0 (D-03), never NaN.
    m = frame_metrics_d03(tp, fp, fn)
    assert m["f1"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["iou"] == 1.0
    # empty GT + NON-empty pred (spurious crack) -> 0.0, not 1.0.
    assert frame_f1_d03(0, 5, 0) == 0.0


def test_late_rollout_window():
    # n=10 -> k = max(1, round(0.2*10)) = 2 ; macro over the LAST 2 frames.
    rows = [{"tp": 1, "fp": 0, "fn": 0} for _ in range(8)]
    rows.append({"tp": 5, "fp": 5, "fn": 0})    # prec=0.5, rec=1.0  -> f1 = 2/3
    rows.append({"tp": 3, "fp": 0, "fn": 3})    # prec=1.0, rec=0.5  -> f1 = 2/3
    expected = (2.0 / 3.0 + 2.0 / 3.0) / 2.0
    assert late_rollout_macro_f1(rows, frac=0.20) == pytest.approx(expected)


def test_selection_metric_signature(per_frame_csv):
    # val-only inputs: a list of per-case val CSV paths -> ONE float.
    csv1 = per_frame_csv([(1, 0, 0, 10), (2, 0, 0, 9)], name="val_a")
    csv2 = per_frame_csv([(3, 1, 1, 8), (0, 0, 0, 11)], name="val_b")
    result = late_rollout_selection_metric([csv1, csv2])
    assert isinstance(result, float)


# ---- MODEL-04 / D-01: the ConvLSTM reference headline cell is LATE-rollout ----
def _ref_cell(lines, case):
    """Extract the reference (3rd) cell from the headline row for ``case``.

    Headline row schema: ``| case | new macro | REF | micro | late |``.
    """
    row = next(l for l in lines if l.startswith(f"| {case} |"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    return cells[2]


def test_reference_cell_uses_late_rollout(per_frame_csv, tmp_path):
    # D-01: the reference column must mirror the FractureTAU late-rollout column
    # (late_rollout_macro_f1, frac=0.20) so the head-to-head is apples-to-apples
    # -- NOT the full-rollout macro. Build a CSV where the two differ so a
    # regression back to the full-rollout macro fails this test.
    # n=10 -> late window k = max(1, round(0.2*10)) = 2 (last 2 frames).
    # First 8 frames perfect (F1=1.0); last 2 prec=rec=0.5 (F1=0.5).
    # full-rollout macro = (8*1.0 + 2*0.5)/10 = 0.9 ; late (last 2) = 0.5.
    counts = [(10, 0, 0, 1) for _ in range(8)]
    counts += [(5, 5, 5, 1), (5, 5, 5, 1)]
    ref_src = per_frame_csv(counts, name="ref_src")

    full_macro = macro_metrics_from_csv(ref_src)["macro_f1"]
    late = late_rollout_macro_f1(make_report._read_rows(ref_src))
    assert full_macro != pytest.approx(late)     # guard: the two MUST differ

    case, ref_mode = "F_MS206_V1000_out", "BASE"
    eval_dir = tmp_path / "eval"
    (eval_dir / case).mkdir(parents=True)
    shutil.copy(ref_src, eval_dir / case / "per_frame_metrics.csv")
    old_root = tmp_path / "old"
    (old_root / ref_mode / case).mkdir(parents=True)
    shutil.copy(ref_src, old_root / ref_mode / case / "per_frame_metrics.csv")

    ref_cell = _ref_cell(
        make_report._headline_section([case], eval_dir, old_root, ref_mode), case
    )
    assert ref_cell == f"{late:.4f}"            # late-rollout, not full macro
    assert ref_cell != f"{full_macro:.4f}"


def test_reference_cell_absent_csv_not_yet_evaluated(per_frame_csv, tmp_path):
    # D-09 honesty: a missing reference CSV reads "not yet evaluated" -- never
    # blank, never fabricated -- even after the late-rollout rewiring.
    new_src = per_frame_csv([(10, 0, 0, 1), (5, 5, 5, 1)], name="new_src")

    case, ref_mode = "F_MS206_V1000_out", "BASE"
    eval_dir = tmp_path / "eval"
    (eval_dir / case).mkdir(parents=True)
    shutil.copy(new_src, eval_dir / case / "per_frame_metrics.csv")
    old_root = tmp_path / "old"                  # no reference CSV under here
    old_root.mkdir()

    ref_cell = _ref_cell(
        make_report._headline_section([case], eval_dir, old_root, ref_mode), case
    )
    assert ref_cell == "not yet evaluated"
