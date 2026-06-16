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

# ---- repo-root + scripts sys.path shim ----
# dashboard -> results -> dynamic-fracture; results/scripts holds compare_runs.
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "results" / "scripts"
for _p in (str(REPO_ROOT), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse -- do NOT rebuild -- the already-tested coverage seam and the canonical
# case/mode ordering (build_coverage emits the 16x4 yes/no grid; CASE_MAP is the
# identity map over case_registry.TEST_CASE_FOLDERS; MODES is the 4-mode order).
from compare_runs import CASE_MAP, MODES, build_coverage  # noqa: E402
from case_registry import TEST_CASE_FOLDERS  # noqa: E402

# ---- honest absent-cell label (D-09) ----
NOT_EVALUATED: str = "not yet evaluated"


# ---- matrix build (D-09) ----
def build_matrix(new: Dict[str, dict], old: Dict[Tuple[str, str], dict]) -> List[dict]:
    """Honest per-(case, mode) status matrix over all 16 canonical cases (D-09).

    Extends ``compare_runs.build_coverage(new, old)``: in addition to the
    ``baseline_present`` / ``new_present`` yes/no flags, adds a ``status`` cell
    that surfaces the evaluated F1 when present and :data:`NOT_EVALUATED` when the
    cell is absent (never blank/omitted). ``new`` is ``{case: metrics}``; ``old``
    is ``{(mode, case): metrics}``. Returns one dict per (case, mode) -- 16 * 4
    rows, in the same order build_coverage emits them.

    Status derivation (D-09 honest cell):
      * baseline + new present -> evaluated head-to-head: the new-model F1.
      * new present, baseline absent -> ``"new only (<F1>)"``.
      * baseline present, new absent -> ``"baseline only (<F1>)"``.
      * neither -> :data:`NOT_EVALUATED`.
    """
    rows: List[dict] = []
    for r in build_coverage(new, old):
        case, mode = r["case"], r["mode"]
        baseline = r["baseline_present"] == "yes"
        new_present = r["new_present"] == "yes"
        if baseline and new_present:
            status = f"{new[case]['f1']:.4f}"
        elif new_present:
            status = f"new only ({new[case]['f1']:.4f})"
        elif baseline:
            status = f"baseline only ({old[(mode, case)]['f1']:.4f})"
        else:
            status = NOT_EVALUATED
        rows.append({**r, "status": status})
    return rows


def render_matrix_md(matrix_rows: List[dict]) -> str:
    """Render the status matrix as a Markdown table (D-09).

    Header + ``|---|...|`` separator + one row per canonical case (mode columns),
    drawing absent cells as :data:`NOT_EVALUATED`. Mirrors the Markdown table
    idiom in ``compare_runs.py:161-167``. Cases follow the canonical
    ``TEST_CASE_FOLDERS`` order. This is one section of the report (D-14 order).
    """
    status_by_cell = {(r["case"], r["mode"]): r["status"] for r in matrix_rows}
    lines = [
        "| Case | " + " | ".join(MODES) + " |",
        "|---|" + "|".join(["---"] * len(MODES)) + "|",
    ]
    for case in TEST_CASE_FOLDERS:
        cells = [status_by_cell.get((case, m), NOT_EVALUATED) for m in MODES]
        lines.append(f"| {case} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
