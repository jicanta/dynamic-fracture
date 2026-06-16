# tests/test_dashboard_matrix.py
"""METR-04: honest 16x4 case x mode coverage matrix (D-09).

Pins the LOCKED convention the Plan-04 implementation must satisfy: absent cells
render the explicit string "not yet evaluated" (never blank), a present BASE
cell renders as evaluated, and the matrix enumerates all 16 canonical cases x 4
modes. Mirrors the honest-coverage assertions in
``test_compare_runs.py::test_coverage_reports_all_cases_honestly``.

Wave-0 state: the ``matrix`` bodies raise ``NotImplementedError`` -> RED (runtime
failure, NOT a collection/import error). Plan 04 turns them GREEN.
"""
from __future__ import annotations

import sys
from pathlib import Path

# ---- sys.path shim: repo root + results + results/dashboard ----
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
DASHBOARD = RESULTS / "dashboard"
for _p in (str(REPO_ROOT), str(RESULTS), str(DASHBOARD)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from case_registry import TEST_CASE_FOLDERS  # noqa: E402
from dashboard.matrix import (  # noqa: E402
    MODES,
    NOT_EVALUATED,
    build_matrix,
    render_matrix_md,
)


def test_honest_matrix():
    # With nothing on disk every cell is honest "not yet evaluated".
    rows = build_matrix({}, {})
    assert len(rows) == len(TEST_CASE_FOLDERS) * len(MODES) == 16 * 4
    md = render_matrix_md(rows)
    assert NOT_EVALUATED in md

    # A present BASE cell for one canonical case renders as evaluated, not absent.
    one = next(iter(TEST_CASE_FOLDERS))
    rows2 = build_matrix({one: {"f1": 0.5}}, {("BASE", one): {"f1": 0.5}})
    md2 = render_matrix_md(rows2)
    # the evaluated value surfaces; the present cell is not the absent label
    assert "0.5" in md2
