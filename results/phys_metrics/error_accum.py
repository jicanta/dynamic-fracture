"""Error-accumulation: teacher-forced vs autoregressive F1-over-horizon gap.

The TF-vs-AR curve is the paper's *fidelity evidence* (D-13): how much the
autoregressive (AR) rollout degrades relative to a teacher-forced (TF) pass as
the prediction horizon grows. That curve is NOT stored in any existing CSV
(RESEARCH Pitfall 2) — it must be GENERATED from two parallel per-frame metric
files:

  * ``per_frame_metrics.csv``     — the AR rollout (model feeds its own pred back)
  * ``per_frame_metrics_tf.csv``  — the teacher-forced rollout (GT channel-0 fed
    back each step), produced by ``new_model/src/evaluate.py --teacher-forced``.
    That teacher-forced pass is inference-only and freeze-legal per D-14; this
    module merely CONSUMES its CSV output (it never trains or touches weights).

For each frame this computes the degenerate-aware per-frame F1 (D-03) on BOTH
rows from their tp/fp/fn COUNTS — reusing the single canonical
``dashboard.aggregate.frame_f1_d03`` (S4). The stored ``f1`` column is NEVER
read (a second F1 derivation would silently diverge from the dashboard
headline, threat T-05-13). The reported gap is ``gap[i] = f1_tf[i] - f1_ar[i]``:
zero where the two rollouts agree, positive where AR has degraded below TF.

Implements:
  * PHYS-04 — ``tf_vs_ar_gap`` + ``gap_from_csvs``.
  * D-13    — fidelity-evidence framing (TF-vs-AR is the degradation curve).
  * D-14    — the teacher-forced pass that produces ``per_frame_metrics_tf.csv``
              is freeze-legal (inference only); this module consumes its output.

Framework rule (S1): numpy / stdlib ONLY. NO torch, NO tensorflow.

Run:
    cd dynamic-fracture && python -c "import sys; sys.path.insert(0,'results'); \
from phys_metrics.error_accum import tf_vs_ar_gap; \
ar=[{'tp':10,'fp':0,'fn':0},{'tp':5,'fp':5,'fn':5}]; \
tf=[{'tp':10,'fp':0,'fn':0},{'tp':10,'fp':0,'fn':0}]; \
print(tf_vs_ar_gap(ar, tf))"
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

# ---- sys.path shim: make the repo-root `results/` package importable so
# `dashboard.aggregate` resolves whether imported as `phys_metrics.error_accum`
# or run as a script (mirrors physical.py / the test shim). ----
_RESULTS = Path(__file__).resolve().parents[1]          # dynamic-fracture/results
if str(_RESULTS) not in sys.path:
    sys.path.insert(0, str(_RESULTS))

# Reuse the ONE D-03-aware per-frame F1 + the canonical count-row CSV reader
# (S4 — do NOT re-implement F1, NEVER read the stored `f1` column).
from dashboard.aggregate import frame_f1_d03, _read_count_rows  # noqa: E402


def _frame_f1s(rows: Sequence[Dict[str, float]]) -> List[float]:
    """Per-frame D-03 F1 list from tp/fp/fn count rows (reuse ``frame_f1_d03``)."""
    return [
        frame_f1_d03(float(r["tp"]), float(r["fp"]), float(r["fn"]))
        for r in rows
    ]


def tf_vs_ar_gap(
    ar_rows: Sequence[Dict[str, float]],
    tf_rows: Sequence[Dict[str, float]],
) -> np.ndarray:
    """Per-frame TF-vs-AR F1 gap ``gap[i] = f1_tf[i] - f1_ar[i]`` (PHYS-04).

    ``ar_rows`` / ``tf_rows`` are equal-length sequences of per-frame dicts
    carrying ``tp/fp/fn`` count keys (as read from a ``per_frame_metrics.csv``).
    Each frame's F1 is recomputed from counts under the D-03 convention via
    :func:`frame_f1_d03` (S4) — the stored ``f1`` column is intentionally never
    used. Identical inputs -> an all-zero gap; a per-frame AR degradation ->
    a positive gap at that frame.

    Raises ``ValueError`` (S8) if the two sequences differ in length, since a
    horizon-aligned gap is undefined when the rollouts cover different frames.
    """
    if len(ar_rows) != len(tf_rows):
        raise ValueError(
            f"tf_vs_ar_gap: AR/TF length mismatch "
            f"({len(ar_rows)} != {len(tf_rows)}) — cannot align by horizon"
        )
    ar_f1 = np.asarray(_frame_f1s(ar_rows), dtype=float)
    tf_f1 = np.asarray(_frame_f1s(tf_rows), dtype=float)
    return tf_f1 - ar_f1


def gap_from_csvs(ar_csv: str | Path, tf_csv: str | Path) -> np.ndarray:
    """Convenience: read both per-frame CSVs and return :func:`tf_vs_ar_gap`.

    Reads tp/fp/fn count rows via the canonical ``_read_count_rows`` (S4) — fails
    loud (``FileNotFoundError``) on a missing CSV — then delegates to
    :func:`tf_vs_ar_gap`. ``ar_csv`` is the AR ``per_frame_metrics.csv``;
    ``tf_csv`` is the teacher-forced ``per_frame_metrics_tf.csv`` (D-14 output).
    """
    return tf_vs_ar_gap(_read_count_rows(ar_csv), _read_count_rows(tf_csv))
