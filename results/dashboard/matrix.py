"""Dashboard coverage-matrix contracts: honest 16x4 case x mode grid (D-09).

Extends -- does NOT rebuild -- ``results/scripts/compare_runs.build_coverage``
(lines 84-99), which already emits a 16x4 honest ``baseline_present`` /
``new_present`` yes/no table over all canonical cases. This module renders that
dict-of-rows into the METR-04 Markdown grid, drawing ABSENT cells as
``"not yet evaluated"`` (D-09) rather than blank, so missing baselines are stated
explicitly. Only BASE x 16 is populated in Phase 1; the structure already
tolerates partial population.

Reuses the canonical case registry (``case_registry.TEST_CASE_FOLDERS`` via
``compare_runs.CASE_MAP``) and the ``MODES`` ordering -- never re-list the cases.

Framework rule (D-13): stdlib + (optionally) pandas ONLY. NO torch, NO tensorflow.

This is a Wave-0 CONTRACT module: every public function raises
``NotImplementedError``. Plan 04 fills in the bodies and turns the RED test
(``tests/test_dashboard_matrix.py::test_honest_matrix``) GREEN.

Run:
    cd dynamic-fracture && python -c "import sys; sys.path.insert(0,'results'); \
from dashboard.matrix import MODES; print(MODES)"
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ---- repo-root sys.path shim ----
# dashboard -> results -> dynamic-fracture.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---- mode ordering (matches compare_runs.MODES:37) ----
MODES: List[str] = ["BASE", "SED", "VON", "PRESSURE"]

# ---- honest absent-cell label (D-09) ----
NOT_EVALUATED: str = "not yet evaluated"


# ---- matrix build (D-09) ----
def build_matrix(new: Dict[str, dict], old: Dict[Tuple[str, str], dict]) -> List[dict]:
    """Honest per-(case, mode) status matrix over all 16 canonical cases (D-09).

    Extends ``compare_runs.build_coverage(new, old)``: in addition to the
    ``baseline_present`` / ``new_present`` yes/no flags, adds a ``status`` cell
    that is the evaluated metric when present and :data:`NOT_EVALUATED` when the
    cell is absent. ``new`` is ``{case: metrics}``; ``old`` is
    ``{(mode, case): metrics}``. Returns one dict per (case, mode) -- 16 * 4 rows.
    """
    raise NotImplementedError("Plan 04: D-09 honest status matrix (extends build_coverage)")


def render_matrix_md(matrix_rows: List[dict]) -> str:
    """Render the status matrix as a Markdown table (D-09).

    Header + ``|---|...|`` separator + one row per case (mode columns), drawing
    absent cells as :data:`NOT_EVALUATED`. Mirrors the Markdown table idiom in
    ``compare_runs.py:161-167``. This is one section of the report (D-14 order).
    """
    raise NotImplementedError("Plan 04: D-09 render matrix to Markdown")
