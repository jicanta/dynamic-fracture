"""One-command dashboard report driver contract (D-08/D-14).

Assembles the full multi-metric dashboard into a single ``results/REPORT.md``
plus committed figures, in ONE command (D-08). Section order is LOCKED by D-14:
headline macro-F1 table -> stability band -> 16x4 honest matrix -> threshold
curve -> curated FP/FN panels.

Mirrors the multi-output write driver of ``compare_runs.main`` (102-168) and the
Markdown-assembly-from-artifacts idiom of ``evaluate.write_reports`` (build a
``lines = [...]`` list, append rows in a loop, ``"\n".join`` -> ``write_text``).
Path constants live at module top like ``compare_runs.py:32-35``; never
hard-code ``/scratch/gilbreth/<user>/`` paths (Security) -- derive from
``case_registry`` + config defaults. CLI via ``argparse`` like
``micro_metrics.main:45-49``.

Outputs (RESEARCH Recommended Project Structure):
    results/REPORT.md
    results/figures/*.png        (committed)
    results/diagnostics/*        (per-case)

Framework rule (D-13): numpy / matplotlib / pandas / Pillow / stdlib ONLY.
NO torch, NO tensorflow.

This is a Wave-0 CONTRACT module: ``main`` raises ``NotImplementedError``.
Plan 06 fills in the body.

Run:
    cd dynamic-fracture && python -m results.dashboard.make_report --help
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---- repo-root sys.path shim ----
# dashboard -> results -> dynamic-fracture.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---- output path constants (mirror compare_runs.py:32-35) ----
REPORT_MD = "results/REPORT.md"
FIGURES_DIR = "results/figures"
DIAGNOSTICS_DIR = "results/diagnostics"


# ---- one-command driver (D-08/D-14) ----
def main() -> None:
    """Build the full dashboard report in one command (D-08/D-14).

    Parses CLI args (argparse), then assembles the locked-order sections
    (headline macro-F1 table, stability band, 16x4 honest matrix, threshold
    curve, curated panels) into :data:`REPORT_MD`, writing figures under
    :data:`FIGURES_DIR` and per-case diagnostics under :data:`DIAGNOSTICS_DIR`.
    """
    raise NotImplementedError("Plan 06: D-08/D-14 one-command REPORT.md driver")


if __name__ == "__main__":
    main()
