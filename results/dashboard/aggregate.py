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

import csv
import glob
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

# ---- repo-root sys.path shim ----
# dashboard -> results -> dynamic-fracture; makes frac_metrics / case_registry /
# results.scripts.micro_metrics importable from this subdir.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---- results/scripts sys.path shim (mirror compare_runs.py:23) ----
# Puts micro_metrics on the path so the D-02 secondary column reuses the frozen
# micro() implementation verbatim instead of re-deriving it here.
SCRIPTS_DIR = REPO_ROOT / "results" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from micro_metrics import micro  # noqa: E402  (D-02 reuse — do NOT re-implement)


# ---- per-frame (D-03-aware) ----
def frame_f1_d03(tp: float, fp: float, fn: float) -> float:
    """Degenerate-aware per-frame F1 (D-03).

    Differs from ``frac_metrics`` ONLY on the all-zero frame: when
    ``tp + fp + fn == 0`` (empty GT AND empty pred) this returns ``1.0`` (the
    model correctly predicted nothing), whereas the frozen seam returns ``0.0``.
    All other branches match the seam's foreground-only F1.
    """
    if tp + fp + fn == 0:          # GT empty AND pred empty -> predicted nothing
        return 1.0                 # D-03: F1 = 1  (frac_metrics.py:85 returns 0.0)
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0


def frame_metrics_d03(tp: float, fp: float, fn: float) -> Dict[str, float]:
    """Degenerate-aware per-frame precision/recall/f1/iou (D-03).

    Same ``0/0 -> 1.0`` rule as :func:`frame_f1_d03`, applied to each of
    precision, recall, f1, iou when the frame is all-zero.
    Returns ``{"precision", "recall", "f1", "iou"}``.
    """
    if tp + fp + fn == 0:          # fully degenerate frame: correctly predicted nothing
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "iou": 1.0}
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    iou = tp / (tp + fp + fn)      # tp+fp+fn > 0 guaranteed by the guard above
    f1 = frame_f1_d03(tp, fp, fn)
    return {"precision": prec, "recall": rec, "f1": f1, "iou": iou}


# ---- macro aggregation (D-01) ----
def macro_f1_from_counts(rows: Iterable[Dict[str, float]]) -> float:
    """Macro F1 = mean over frames of :func:`frame_f1_d03` (D-01).

    ``rows`` is an iterable of per-frame dicts carrying ``tp/fp/fn`` count keys
    (as read from a ``per_frame_metrics.csv``). The stored ``f1`` column is
    intentionally NOT used -- macro F1 is recomputed from counts under the D-03
    convention.
    """
    f1s = [frame_f1_d03(float(r["tp"]), float(r["fp"]), float(r["fn"])) for r in rows]
    if not f1s:
        raise ValueError("macro_f1_from_counts: no rows to aggregate")
    return sum(f1s) / len(f1s)


def macro_metrics_from_csv(path: str | Path) -> Dict[str, float]:
    """Full aggregate summary for one case's ``per_frame_metrics.csv``.

    Returns ``{macro_f1, micro_f1, precision, recall, iou, frames}`` where
    ``macro_f1`` is the D-01/D-03 mean and ``micro_f1`` delegates to
    :func:`micro_f1_from_csv` (D-02). Reads counts via ``csv.DictReader`` so the
    canonical schema's extra provenance columns are ignored
    (mirrors ``micro_metrics.py:21-34``).
    """
    rows = _read_count_rows(path)
    if not rows:
        raise ValueError(f"macro_metrics_from_csv: no rows in {path}")
    per_frame = [
        frame_metrics_d03(float(r["tp"]), float(r["fp"]), float(r["fn"])) for r in rows
    ]
    n = len(per_frame)
    return {
        "macro_f1": sum(m["f1"] for m in per_frame) / n,
        "precision": sum(m["precision"] for m in per_frame) / n,
        "recall": sum(m["recall"] for m in per_frame) / n,
        "iou": sum(m["iou"] for m in per_frame) / n,
        "frames": n,
        "micro_f1": micro_f1_from_csv(path),     # D-02 labelled secondary value
    }


def micro_f1_from_csv(path: str | Path) -> float:
    """Micro F1 over all frames of one case CSV (D-02).

    Delegates to ``micro_metrics.micro`` (reuse, do NOT re-implement micro):
    sums tp/fp/fn over every frame then computes F1 from the totals.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"micro_f1_from_csv: no CSV at {p}")
    return float(micro(str(p))["f1"])


# ---- CSV helper (mirror micro_metrics.py:25-28) ----
def _read_count_rows(path: str | Path) -> List[Dict[str, int]]:
    """Read the tp/fp/fn(/tn) integer count columns from one per-frame CSV.

    Keyed by header via ``csv.DictReader`` (mirror ``micro_metrics.py:25-28``) so
    the canonical schema's extra provenance columns are ignored. Fails loud with
    ``FileNotFoundError`` on a missing CSV (threat T-02D-05: malformed/absent CSV
    surfaces immediately, never silently).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"_read_count_rows: no CSV at {p}")
    with open(p, newline="") as f:
        rows = [
            {"tp": int(r["tp"]), "fp": int(r["fp"]), "fn": int(r["fn"]),
             "tn": int(r["tn"])}
            for r in csv.DictReader(f)
        ]
    return rows


# ---- late-rollout window (D-04) + selection seam (D-05/D-06) ----
def late_rollout_macro_f1(rows: Sequence[Dict[str, float]], frac: float = 0.20) -> float:
    """Macro F1 over the last ``frac`` of one case's frames (D-04).

    Window size ``k = max(1, round(frac * n))`` frames taken from the END of the
    rollout; macro F1 (D-01/D-03) is computed within that window. Captures
    long-horizon stability where the autoregressive rollout degrades most.
    """
    rows = list(rows)
    n = len(rows)
    if n == 0:
        raise ValueError("late_rollout_macro_f1: no rows to aggregate")
    k = max(1, round(frac * n))          # D-04: per-case last-frac, never 0
    tail = rows[-k:]
    return macro_f1_from_counts(tail)


def late_rollout_selection_metric(val_csv_paths: Iterable[str | Path]) -> float:
    """The ONE val-only model-selection scalar (D-05/D-06).

    Given the val-set per-case ``per_frame_metrics.csv`` paths, computes the
    per-case late-rollout macro F1 (:func:`late_rollout_macro_f1`) then averages
    ACROSS cases, returning a single float. Inputs are VALIDATION CSVs only --
    test-set CSVs must never reach this function (no selection-time leakage).
    This is the stable seam Phase 3/4 imports for sweep/checkpoint selection.

    D-05: the late-rollout (last-20%) AR macro F1 is the PRIMARY model-selection
    criterion exposed for Phase 3/4 (short-rollout val F1 is only a
    guardrail/tiebreaker).
    D-06: the inputs MUST be VALIDATION-split per-case CSVs ONLY — never the 16
    held-out test cases. Passing test-set CSVs here is selection-time leakage and
    is the caller's contract to avoid; this function makes no test-set access.
    """
    paths = list(val_csv_paths)
    if not paths:
        raise ValueError("late_rollout_selection_metric: empty val_csv_paths "
                         "(no validation per_frame_metrics.csv provided)")
    per_case = [late_rollout_macro_f1(_read_count_rows(p)) for p in paths]
    return sum(per_case) / len(per_case)


def selection_metric_from_eval_dir(eval_dir: str | Path) -> float:
    """Directory convenience over :func:`late_rollout_selection_metric` (D-05/D-06).

    Globs ``<case>/per_frame_metrics.csv`` under ``eval_dir`` (mirror
    ``micro_metrics.collect``) and feeds the discovered paths to the val-only
    selection metric. The SAME D-06 contract applies: ``eval_dir`` must contain
    ONLY validation-split cases, never the held-out test cases. Fails loud (mirror
    ``micro_metrics.main:52-53``) when no per-case CSV is found.
    """
    paths = sorted(glob.glob(os.path.join(str(eval_dir), "*", "per_frame_metrics.csv")))
    if not paths:
        raise ValueError(f"no <case>/per_frame_metrics.csv found under {eval_dir}")
    return late_rollout_selection_metric(paths)
