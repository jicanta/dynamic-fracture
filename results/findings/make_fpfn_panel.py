#!/usr/bin/env python3
"""Generate the deferred FP/FN qualitative panel (D-07) from REAL v1.0 AR artifacts.

Closes the paper's one open figure gap HONESTLY: the false-positive / false-negative
overlay panel for the representative failure case ``test_inclusions_1_2`` is rendered
directly from the confirmed-local, first-party rollout artifacts
(``probs.npz`` / ``gt.npz``) — it is NEVER synthesized. If an artifact were missing the
loader (``load_case_probs_gt``, allow_pickle=False) fails loudly; this driver adds no
fallback that would fabricate masks.

This is a THIN driver: it reuses the audited helpers and adds no new metric/overlay
math. Threshold is read from the case's own ``summary.json`` (0.275 here — never
hardcoded to 0.5), and binarization follows the no-healing rollout rule
``pred = np.maximum.accumulate((probs >= thr), axis=0)`` (matches
``_binarize_no_healing`` in ``results/phys_metrics/make_phys_report.py``). GT and pred
come from the SAME ``load_case_probs_gt`` seam so orientation matches (Pitfall 3).

Self-sufficient PDF fonts: sets ``pdf.fonttype`` / ``ps.fonttype`` = 42 (Type-42, no
Type-3) at import time so the emitted vector PDF embeds proper fonts regardless of any
other module's rcParams edits.

Run from dynamic-fracture/:
    python results/findings/make_fpfn_panel.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ---- self-sufficient vector-PDF fonts (Type-42; D-06) ----
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# ---- repo-root sys.path shim (findings -> results -> dynamic-fracture) ----
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
for _p in (str(REPO_ROOT), str(RESULTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---- reuse-by-import: never reimplement the loader or the overlay renderer ----
from dashboard.probs_io import load_case_probs_gt  # noqa: E402  (allow_pickle=False)
from dashboard.plots import save_fpfn_panel  # noqa: E402  (dark facecolor; .pdf-aware)

CASE_DIR = REPO_ROOT / "new_model" / "OUTPUTS" / "headline_s42" / "eval" / "test_inclusions_1_2"
OUT_PDF = REPO_ROOT / "paper" / "figures" / "fpfn_inclusions_1_2.pdf"


def _read_threshold(case_dir: Path) -> float:
    """Read the calibrated threshold from the case's own summary.json (fail loud)."""
    summary_path = case_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"no summary.json under {case_dir}")
    with summary_path.open() as fh:
        summary = json.load(fh)
    if "threshold" not in summary:
        raise KeyError(f"summary.json missing 'threshold': {summary_path}")
    return float(summary["threshold"])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Render the FP/FN qualitative panel from real probs.npz/gt.npz.")
    ap.add_argument(
        "--case-dir", default=str(CASE_DIR),
        help="case dir holding probs.npz/gt.npz/summary.json "
             f"(default: {CASE_DIR})")
    ap.add_argument(
        "--out", default=str(OUT_PDF),
        help=f"output PDF path (default: {OUT_PDF})")
    args = ap.parse_args(argv)
    case_dir = Path(args.case_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # (1) load real probs + GT from the SAME seam (orientation-consistent).
    probs, gts = load_case_probs_gt(case_dir)  # (T,H,W) float32, (T,H,W) uint8

    # (2) threshold from THIS case's summary.json (0.275; never hardcode 0.5).
    thr = _read_threshold(case_dir)

    # (3) no-healing binarization (monotone over time) — matches _binarize_no_healing.
    pred = (probs >= thr).astype(np.uint8)
    pred = np.maximum.accumulate(pred, axis=0)

    # (4) three curated frames across the rollout.
    T = len(gts)
    frames = [int(0.25 * T), int(0.5 * T), int(0.85 * T)]
    titles = [f"frame {f}" for f in frames]

    # (5) render the FP/FN panel as a vector PDF via the audited helper.
    save_fpfn_panel(out, gts, pred, frames, titles=titles)
    print(f"[fpfn] wrote {out} (thr={thr}, T={T}, frames={frames})")


if __name__ == "__main__":
    main()
