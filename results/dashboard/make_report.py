"""One-command dashboard report driver (D-08/D-14).

Assembles the full multi-metric dashboard into a single ``results/REPORT.md``
plus committed figures, in ONE command (D-08). Section order is LOCKED by D-14:

    1. Headline (macro F1)        -- both-models macro/micro/late-rollout table
    2. Rollout stability          -- median+IQR band over normalized rollout
    3. Coverage matrix (16x4)     -- honest case x mode grid (D-09)
    4. Threshold sensitivity      -- FractureTAU-only sweep, calibrated thr marked
    5. Qualitative panels         -- curated FP/FN panels (all-16 under diagnostics)

Orchestrates the aggregation core (Plan 02 ``aggregate``), the figures (Plan 05
``plots``), the coverage matrix (Plan 04 ``matrix``), and the saved-prob loader
(Plan 03 ``probs_io``) -- it computes nothing new, it only assembles. Mirrors the
multi-output write driver of ``compare_runs.main`` (102-168) and the Markdown-
assembly-from-artifacts idiom of ``evaluate.write_reports`` (build a
``lines = [...]`` list, append rows in a loop, ``"\n".join`` -> ``write_text``).

Path constants live at module top like ``compare_runs.py:32-35``; NEVER hard-code
absolute Gilbreth user scratch paths (Security T-02D-14) -- everything is derived
from the canonical ``case_registry`` + ``compare_runs`` roots + the run dir + the
module constants below, resolved against ``REPO_ROOT``. CLI via ``argparse`` like
``micro_metrics.main:45-49`` / ``compare_runs.main:103-105``.

Honest degradation (D-09): the headline reference column, the threshold curve and
the qualitative panels each require artifacts (reference CSVs / saved
``probs.npz`` + ``gt.npz``) that may not be present for a given run. When absent
the report states ``not yet evaluated`` / ``not yet available`` explicitly rather
than emitting a blank or a fabricated figure.

Framework rule (D-13): numpy / matplotlib / pandas / Pillow / stdlib ONLY.
NO torch, NO tensorflow.

Run:
    cd dynamic-fracture && python results/dashboard/make_report.py --run-name tau_base
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---- repo-root + scripts sys.path shim ----
# dashboard -> results -> dynamic-fracture; results/scripts holds compare_runs /
# micro_metrics, the repo root holds case_registry.
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
SCRIPTS = RESULTS_DIR / "scripts"
# RESULTS_DIR makes `import dashboard.*` resolve when run as a bare script
# (python results/dashboard/make_report.py); SCRIPTS holds compare_runs /
# micro_metrics; REPO_ROOT holds case_registry.
for _p in (str(REPO_ROOT), str(RESULTS_DIR), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---- the producers this driver orchestrates (Plans 02/03/04/05) ----
from dashboard.aggregate import (  # noqa: E402
    late_rollout_macro_f1,
    macro_metrics_from_csv,
)
from dashboard.matrix import build_matrix, render_matrix_md  # noqa: E402
from dashboard.plots import (  # noqa: E402
    case_f1_curve_from_rows,
    read_calibrated_threshold,
    save_fpfn_panel,
    select_key_frames,
    stability_band,
    threshold_curve,
)
from dashboard.probs_io import load_case_probs_gt  # noqa: E402

# Reference (old-side / ConvLSTM) roots + the honest 16x4 coverage seam.
from compare_runs import OLD_ROOT, build_coverage, collect_old  # noqa: E402
from micro_metrics import collect  # noqa: E402

# ---- output path constants (mirror compare_runs.py:32-35) ----
NEW_ROOT = "new_model/OUTPUTS"
FIG_DIR = "results/figures"
DIAG_DIR = "results/diagnostics"
REPORT_PATH = "results/REPORT.md"

DEFAULT_THRESHOLD = 0.5            # fallback binarization thr when no calibration
NOT_AVAILABLE = "not yet available"


# ---- path helpers (everything resolved against REPO_ROOT; no scratch paths) ----
def _abs(rel: str | Path) -> Path:
    """Resolve a repo-relative path against REPO_ROOT (cwd-independent)."""
    p = Path(rel)
    return p if p.is_absolute() else (REPO_ROOT / p)


def _read_rows(csv_path: str | Path) -> List[Dict[str, str]]:
    """Per-frame rows keyed by header (tp/fp/fn carried as strings).

    The aggregation helpers coerce ``float(r["tp"])`` internally, so the raw
    ``csv.DictReader`` rows are fed straight through (extra provenance columns
    are simply ignored).
    """
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _ref_csv_path(old_root: Path, ref_mode: str, case: str) -> Path:
    """Reference per-case CSV path, resolved like ``compare_runs`` old-side.

    ``<OLD_ROOT>/<MODE>/<case>/per_frame_metrics.csv`` (METR-01 both-models).
    """
    return old_root / ref_mode / case / "per_frame_metrics.csv"


# ---- section 1: headline (macro F1) ----
def _headline_section(
    cases: List[str], eval_dir: Path, old_root: Path, ref_mode: str
) -> List[str]:
    """Both-models headline table: FractureTAU vs ConvLSTM reference (METR-01).

    Macro F1 (D-01) is the headline win claim; micro F1 (D-02) is a clearly
    labelled secondary column; the late-rollout F1 (D-04) is the last-20% window.
    The reference column calls ``macro_metrics_from_csv`` on the old-side CSV when
    present and reads ``not yet evaluated`` otherwise (D-09 honesty).
    """
    lines = [
        "## Headline (macro F1)",
        "",
        "Macro F1 (D-01) is the head-to-head **win claim** (per-frame F1 averaged "
        "over the rollout); micro F1 (D-02) is a labelled **secondary** column. "
        "Degenerate convention: an empty-GT frame scores 1.0 when the prediction "
        "is also empty (0/0 -> 1.0).",
        "",
        "| Case | FractureTAU macro F1 | ConvLSTM-ref macro F1 | micro F1 (secondary) "
        "| late-rollout F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for case in cases:
        new_csv = eval_dir / case / "per_frame_metrics.csv"
        m = macro_metrics_from_csv(new_csv)
        late = late_rollout_macro_f1(_read_rows(new_csv))

        ref_csv = _ref_csv_path(old_root, ref_mode, case)
        if ref_csv.exists():
            ref_cell = f"{macro_metrics_from_csv(ref_csv)['macro_f1']:.4f}"
        else:
            ref_cell = "not yet evaluated"

        lines.append(
            f"| {case} | {m['macro_f1']:.4f} | {ref_cell} | "
            f"{m['micro_f1']:.4f} | {late:.4f} |"
        )
    lines.append("")
    return lines


# ---- section 2: rollout stability ----
def _stability_section(cases: List[str], eval_dir: Path, fig_dir: Path) -> List[str]:
    """Median+IQR rollout-stability band over the cases (D-11/METR-02)."""
    curves = []
    for case in cases:
        rows = _read_rows(eval_dir / case / "per_frame_metrics.csv")
        curves.append(case_f1_curve_from_rows(rows))

    out_png = fig_dir / "stability.png"
    stability_band(curves, out_png, label=f"median F1 ({len(curves)} cases)")
    return [
        "## Rollout stability",
        "",
        "Per-case foreground-F1 curves are resampled onto a common normalized "
        "**rollout fraction** [0, 1] grid (`np.interp`); the band is the "
        "**median + IQR (25-75%)** across cases at matched rollout fractions "
        "(OQ3/D-11) -- never absolute frame index, never mean +/- std.",
        "",
        "![rollout stability band](figures/stability.png)",
        "",
    ]


# ---- section 3: coverage matrix (16x4) ----
def _matrix_section(eval_dir: Path, old_root: Path, new_mode: str = "BASE") -> List[str]:
    """Honest 16x4 case x mode coverage grid (D-09/METR-04).

    ``new_mode`` is the single mode the FractureTAU run was evaluated in; the
    new-model F1 only populates that column (CR-01 fix: no cross-mode fabrication).
    """
    new = collect(str(eval_dir))
    old = collect_old() if old_root == _abs(OLD_ROOT) else _collect_old_under(old_root)
    matrix_rows = build_matrix(new, old, new_mode=new_mode)
    return [
        "## Coverage matrix (16x4)",
        "",
        "Every canonical case x mode cell is reported explicitly; cells with no "
        "evaluation read **not yet evaluated** (D-09) rather than being dropped. "
        "Only BASE is fully regenerated in Phase 1.",
        "",
        render_matrix_md(matrix_rows),
    ]


def _collect_old_under(old_root: Path) -> Dict[Tuple[str, str], dict]:
    """``{(mode, case): metrics}`` for a non-default reference root.

    Mirrors ``compare_runs.collect_old`` but parameterized on ``old_root`` so a
    test/override root resolves without monkeypatching the module constant.
    """
    import os

    out: Dict[Tuple[str, str], dict] = {}
    if not old_root.is_dir():
        return out
    for mode in sorted(os.listdir(old_root)):
        mode_dir = old_root / mode
        if not mode_dir.is_dir():
            continue
        for case, m in collect(str(mode_dir)).items():
            out[(mode, case)] = m
    return out


# ---- section 4: threshold sensitivity ----
def _load_probs_gt(cases: List[str], eval_dir: Path) -> Tuple[List, List, List[str]]:
    """Load every case that has saved ``probs.npz`` + ``gt.npz`` (Plan 03)."""
    probs_list, gt_list, have = [], [], []
    for case in cases:
        try:
            probs, gt = load_case_probs_gt(eval_dir / case)
        except FileNotFoundError:
            continue
        probs_list.append(probs)
        gt_list.append(gt)
        have.append(case)
    return probs_list, gt_list, have


def _threshold_section(
    probs_list: List, gt_list: List, have: List[str], run_dir: Path, fig_dir: Path
) -> List[str]:
    """FractureTAU-only threshold-sensitivity curve (D-12/OQ4/METR-07)."""
    if not have:
        return [
            "## Threshold sensitivity",
            "",
            f"_{NOT_AVAILABLE}: no per-case `probs.npz`/`gt.npz` found for this run. "
            "Re-run `python -m src.evaluate` (the Plan-03 raw-probability emission) "
            "to populate the saved-prob stacks this curve sweeps._",
            "",
        ]
    calibrated = read_calibrated_threshold(run_dir)
    out_png = fig_dir / "threshold.png"
    threshold_curve(probs_list, gt_list, out_png, calibrated_thr=calibrated)
    marker = (
        f"the Phase-1 calibrated threshold ({calibrated:.3f}) is marked (dashed)"
        if calibrated is not None
        else "no `calibration.json` was found, so no calibrated-threshold marker is drawn"
    )
    return [
        "## Threshold sensitivity",
        "",
        f"FractureTAU macro F1 swept over the binarization threshold on the saved "
        f"probabilities ({len(have)} case(s)), no-healing applied. This is "
        "**FractureTAU-only** (OQ4): the ConvLSTM reference is pre-binarized / "
        f"fixed-threshold and is never re-rolled. The threshold is reused, not "
        f"recomputed -- {marker}.",
        "",
        "![threshold sensitivity curve](figures/threshold.png)",
        "",
    ]


# ---- section 5: qualitative panels ----
def _panels_section(
    probs_list: List,
    gt_list: List,
    have: List[str],
    macro_by_case: Dict[str, float],
    run_dir: Path,
    fig_dir: Path,
    diag_dir: Path,
) -> List[str]:
    """All-16 FP/FN diagnostics + a curated embedded subset (D-10/D-14/METR-06)."""
    if not have:
        return [
            "## Qualitative panels",
            "",
            f"_{NOT_AVAILABLE}: FP/FN panels require per-case `probs.npz`/`gt.npz`. "
            "Re-run `python -m src.evaluate` to populate them._",
            "",
        ]
    thr = read_calibrated_threshold(run_dir)
    thr = DEFAULT_THRESHOLD if thr is None else thr

    # All-16 (all-available) per-case panels into diagnostics/.
    for probs, gt, case in zip(probs_list, gt_list, have):
        pred = (np.asarray(probs) >= thr).astype(np.uint8)
        pred = np.maximum.accumulate(pred, axis=0)     # no-healing, rollout rule
        keys = select_key_frames(gt, pred)
        frames = [keys["onset"], keys["mid"], keys["late"], keys["max_div"]]
        titles = [f"onset ({keys['onset']})", f"mid ({keys['mid']})",
                  f"late ({keys['late']})", f"max-div ({keys['max_div']})"]
        save_fpfn_panel(diag_dir / f"{case}.png", gt, pred, frames, titles=titles)

    # Curated subset: best / worst / representative (median) by macro F1.
    ranked = sorted(have, key=lambda c: macro_by_case.get(c, 0.0))
    curated: List[Tuple[str, str]] = []
    if ranked:
        curated.append((ranked[-1], "best"))
        if len(ranked) > 1:
            curated.append((ranked[0], "worst"))
        if len(ranked) > 2:
            curated.append((ranked[len(ranked) // 2], "representative"))

    lines = [
        "## Qualitative panels",
        "",
        "FP/FN overlays on a dark background: **TP gray (160,160,160)**, "
        "**FP red (220,40,40)**, **FN blue (40,90,220)** (D-14). GT and prediction "
        "are binarized from the SAME saved probabilities in one orientation "
        "(no mirror-flip; Pitfall 2). All-case panels are written under "
        f"`{DIAG_DIR}/`; a curated subset (best / worst / representative by macro "
        "F1) is embedded below.",
        "",
    ]
    idx = {case: i for i, case in enumerate(have)}
    for case, kind in curated:
        probs = probs_list[idx[case]]
        gt = gt_list[idx[case]]
        pred = (np.asarray(probs) >= thr).astype(np.uint8)
        pred = np.maximum.accumulate(pred, axis=0)
        keys = select_key_frames(gt, pred)
        frames = [keys["onset"], keys["mid"], keys["late"], keys["max_div"]]
        titles = [f"onset ({keys['onset']})", f"mid ({keys['mid']})",
                  f"late ({keys['late']})", f"max-div ({keys['max_div']})"]
        panel_png = fig_dir / f"panel_{kind}_{case}.png"
        save_fpfn_panel(panel_png, gt, pred, frames, titles=titles)
        lines += [
            f"**{kind}: {case}** (macro F1 = {macro_by_case.get(case, float('nan')):.4f})",
            "",
            f"![{kind} panel for {case}](figures/{panel_png.name})",
            "",
        ]
    return lines


# ---- one-command driver (D-08/D-14) ----
def build_report(
    run_name: str,
    *,
    new_root: str | Path = NEW_ROOT,
    old_root: str | Path = OLD_ROOT,
    ref_mode: str = "BASE",
    report_path: str | Path = REPORT_PATH,
    fig_dir: str | Path = FIG_DIR,
    diag_dir: str | Path = DIAG_DIR,
    provenance_note: Optional[str] = None,
) -> Path:
    """Assemble the locked-order report for ``run_name`` and write REPORT.md.

    Returns the path of the written report. Fails loud (``FileNotFoundError``)
    when the run eval dir is missing, mirroring ``compare_runs.py:123-124``.
    """
    new_root = _abs(new_root)
    old_root = _abs(old_root)
    report_path = _abs(report_path)
    fig_dir = _abs(fig_dir)
    diag_dir = _abs(diag_dir)

    run_dir = new_root / run_name
    eval_dir = run_dir / "eval"
    if not eval_dir.is_dir():
        raise FileNotFoundError(
            f"[dashboard] run eval dir not found: {eval_dir} "
            f"(expected {NEW_ROOT}/{run_name}/eval)"
        )

    cases = sorted(p.parent.name for p in eval_dir.glob("*/per_frame_metrics.csv"))
    if not cases:
        raise FileNotFoundError(
            f"[dashboard] no <case>/per_frame_metrics.csv under {eval_dir}"
        )
    print(f"[dashboard] {run_name}: {len(cases)} eval cases")

    macro_by_case = {
        c: macro_metrics_from_csv(eval_dir / c / "per_frame_metrics.csv")["macro_f1"]
        for c in cases
    }
    probs_list, gt_list, have = _load_probs_gt(cases, eval_dir)
    print(f"[dashboard] saved-prob stacks available for {len(have)}/{len(cases)} cases")

    lines: List[str] = [
        f"# FractureTAU Dashboard -- {run_name}",
        "",
        "Authoritative, git-diffable static report (D-08). Regenerate with "
        "`python results/dashboard/make_report.py --run-name "
        f"{run_name}`. Sections follow the locked D-14 order.",
        "",
    ]
    if provenance_note:
        lines += [f"> **Provenance:** {provenance_note}", ""]
    # LOCKED D-14 section order.
    lines += _headline_section(cases, eval_dir, old_root, ref_mode)
    lines += _stability_section(cases, eval_dir, fig_dir)
    lines += _matrix_section(eval_dir, old_root, new_mode=ref_mode)
    lines += _threshold_section(probs_list, gt_list, have, run_dir, fig_dir)
    lines += _panels_section(
        probs_list, gt_list, have, macro_by_case, run_dir, fig_dir, diag_dir
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")
    print(f"[dashboard] wrote {report_path}")
    return report_path


def main() -> None:
    """Parse CLI args and build the one-command report (D-08/D-14)."""
    ap = argparse.ArgumentParser(description="Assemble the static dashboard REPORT.md")
    ap.add_argument("--run-name", default="tau_base",
                    help="run dir under new_model/OUTPUTS (default: tau_base)")
    ap.add_argument("--ref-mode", default="BASE",
                    help="reference (old-side) mode for the both-models column")
    ap.add_argument("--new-root", default=NEW_ROOT)
    ap.add_argument("--old-root", default=OLD_ROOT)
    ap.add_argument("--report", default=REPORT_PATH)
    ap.add_argument("--figures", default=FIG_DIR)
    ap.add_argument("--diagnostics", default=DIAG_DIR)
    ap.add_argument("--provenance-note", default=None,
                    help="optional data-provenance blockquote inserted under the title")
    args = ap.parse_args()

    build_report(
        args.run_name,
        new_root=args.new_root,
        old_root=args.old_root,
        ref_mode=args.ref_mode,
        report_path=args.report,
        fig_dir=args.figures,
        diag_dir=args.diagnostics,
        provenance_note=args.provenance_note,
    )


if __name__ == "__main__":
    main()
