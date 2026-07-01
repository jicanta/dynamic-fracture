"""One-command physical-metrics report driver (PHYS-02..06 / D-05/D-13/D-14).

Computes the Phase-5 physical-fidelity tables + figures on the REAL pulled
artifacts of a run (default ``tau_refined``) and assembles them into a single
``results/phys_metrics/PHYS_REPORT.md`` plus committed figures. Mirrors
``results/dashboard/make_report.py`` (argparse ``--run-name``, path constants at
module top resolved against ``REPO_ROOT``, ``lines = [...]`` -> ``write_text``,
honest-degradation strings rather than fabricated numbers). It computes the
physical/boundary/calibration metrics here, but REUSES the Phase-4 headline /
significance scripts (``seed_aggregate`` / ``significance``) and the Wave-2
``phys_metrics`` math by import -- it re-derives no F1 / no Wilcoxon (S4).

Sections (D-13 -- physical fidelity is integrated, not appendix-only):
  1. Headline + 3-seed mean+/-std + Wilcoxon (D-05; via the Phase-4 scripts)
  2. Boundary metrics: HD95 + boundary-F1 (PHYS-02 / D-03)
  3. Physical metrics: crack length-over-time + onset (PHYS-03 / D-02)
  4. Error accumulation: TF-vs-AR F1 over the horizon (PHYS-04 / D-13/D-14)
  5. Calibration: reliability diagram + ECE (PHYS-05 / D-13)
  6. Rollout stability band (median + IQR over cases)
  7. Compute efficiency: params / MACs / FLOPs / A100-hours (PHYS-06)

The teacher-forced ``per_frame_metrics_tf.csv`` (D-14, inference-only,
freeze-legal) is a HARD prerequisite for section 4 -- it is produced by the
teacher-forced eval pass BEFORE this driver runs. The honest-degradation
"teacher-forced pass not yet run" string is kept as DEFENSE ONLY so a misordered
run fails informatively instead of fabricating; it is not the intended end state.

Provenance (DATA-04 honesty): the winning checkpoint SHA (``best.pt``, prefix
``f1a0786c``) is recorded next to the headline numbers. The from-scratch ConvLSTM
reproduction scope is reported HONESTLY as 4 BASE / 2 SED (the corrupt MS206 SED
reference exports are excluded) per D-08.

Security (T-05-15): NO absolute cluster / user-scratch paths are hard-coded --
every path is derived from the run-dir args + the module constants below,
resolved against ``REPO_ROOT``. Saved-array reads go through the
``load_case_probs_gt`` allow_pickle=False seam (T-05-17).

Framework rule: numpy / matplotlib / pandas / stdlib for the assembly; the ONE
torch touch is the efficiency introspection (``phys_metrics.efficiency``), which
imports torch for FractureTAU only -- never tensorflow.

Run:
    cd dynamic-fracture && python results/phys_metrics/make_phys_report.py --run-name tau_refined
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---- repo-root + results + new_model + scripts sys.path shim ----
# make_phys_report -> phys_metrics -> results -> dynamic-fracture; results/ makes
# `dashboard.*` / `phys_metrics.*` resolve; new_model + new_model/scripts make the
# reused Phase-4 `seed_aggregate` / `significance` (and src.model for efficiency)
# importable whether run as a bare script or imported.
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
NEW_MODEL = REPO_ROOT / "new_model"
SCRIPTS = NEW_MODEL / "scripts"
for _p in (str(REPO_ROOT), str(RESULTS), str(NEW_MODEL), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---- reuse-by-import (S4): never re-derive F1 / gap / metrics here ----
from dashboard.aggregate import (  # noqa: E402
    _read_count_rows,
    late_rollout_macro_f1,
    macro_metrics_from_csv,
)
from dashboard.plots import (  # noqa: E402
    case_f1_curve_from_rows,
    read_calibrated_threshold,
    save_fpfn_panel,
    select_key_frames,
    stability_band,
)
from dashboard.probs_io import load_case_probs_gt  # noqa: E402  (allow_pickle=False)
from phys_metrics.boundary import boundary_f1, hd95  # noqa: E402  (PHYS-02)
from phys_metrics.calibration import reliability_ece  # noqa: E402  (PHYS-05)
from phys_metrics.error_accum import gap_from_csvs  # noqa: E402  (PHYS-04)
from phys_metrics.figures import (  # noqa: E402
    f1_vs_horizon_fig,
    length_over_time_fig,
    reliability_fig,
)
from phys_metrics.physical import crack_length_curve, time_to_onset  # noqa: E402  (PHYS-03)

# ---- output path constants (mirror make_report.py; all repo-relative) ----
NEW_ROOT = "new_model/OUTPUTS"
FIG_DIR = "results/phys_metrics/figures"
REPORT_PATH = "results/phys_metrics/PHYS_REPORT.md"
DEFAULT_RUN = "tau_refined"
DEFAULT_THRESHOLD = 0.5
# Phase-4 headline seed runs (under NEW_ROOT) + the frozen ConvLSTM regen dir.
SEED_RUN_NAMES = ("headline_s42", "headline_s43", "headline_s44")
CNN_REF_REL = "kathleens-model/OUTPUTS/base_regen"   # 16-case frozen ConvLSTM regen (relative to REPO_ROOT=dynamic-fracture/)
NOT_AVAILABLE = "not yet available"
# D-08: honest from-scratch reproduction scope (corrupt MS206 SED refs excluded).
REPRO_SCOPE = "4 BASE / 2 SED"


# ---- path + io helpers (everything resolved against REPO_ROOT) ----
def _abs(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (REPO_ROOT / p)


def _fig_rel(out_png: Path, fig_dir: Path) -> str:
    """Markdown-embedded relative path for a figure.

    Default: relative to ``results/`` (unchanged byte-for-byte). When the report
    is written OUTSIDE the source ``results/`` tree (e.g. the self-contained
    ``archive/v1.0/`` light-repro target), that relative_to raises — fall back to
    a path relative to the figures dir's parent (the report's own directory), so
    the embedded link resolves from wherever the report lives.
    """
    try:
        return out_png.relative_to(_abs("results")).as_posix()
    except ValueError:
        return out_png.relative_to(fig_dir.parent).as_posix()


def _read_rows(csv_path: str | Path) -> List[Dict[str, str]]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _sha256_short(path: str | Path, n: int = 12) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def _binarize_no_healing(probs: np.ndarray, thr: float) -> np.ndarray:
    """(prob >= thr) with the no-healing rollout rule (monotone over time)."""
    pred = (np.asarray(probs) >= thr).astype(np.uint8)
    return np.maximum.accumulate(pred, axis=0)


def _gt_binary(gt: np.ndarray) -> np.ndarray:
    return (np.asarray(gt) >= 1).astype(np.uint8)


# ---- section 1: headline + 3-seed mean+/-std + Wilcoxon (D-05, Phase-4 reuse) ----
def _headline_section(new_root: Path, provenance_sha: str) -> List[str]:
    """3-seed late-rollout macro F1 mean+/-std + one-sided Wilcoxon (reuse S4).

    The mean+/-std comes from ``seed_aggregate.per_case_meanstd`` over the headline
    seed runs (computed on the pulled artifacts). The one-sided Wilcoxon
    (TAU > frozen ConvLSTM) is run via ``significance`` ONLY when the 16-case
    from-scratch ConvLSTM reference is present in this checkout; otherwise an
    honest-degradation note is emitted (no fabricated p-value, T-05-16).
    """
    lines = ["## 1. Headline result (3-seed mean +/- std + significance)", ""]
    seed_dirs = [new_root / s for s in SEED_RUN_NAMES if (new_root / s / "eval").is_dir()]
    if len(seed_dirs) < 2:
        lines += [
            f"_{NOT_AVAILABLE}: fewer than 2 headline seed runs found under "
            f"`{NEW_ROOT}` (expected {', '.join(SEED_RUN_NAMES)}). The 3-seed "
            "mean+/-std cannot be formed._",
            "",
        ]
        return lines

    try:
        import seed_aggregate
        rows = seed_aggregate.per_case_meanstd([str(d) for d in seed_dirs])
        shas = seed_aggregate.seed_shas([str(d) for d in seed_dirs])
    except SystemExit as e:
        lines += [f"_{NOT_AVAILABLE}: seed aggregation failed -- {e}_", ""]
        return lines

    overall = statistics.mean(m for _c, m, _s, _n in rows)
    n_seeds = rows[0][3]
    lines += [
        f"Late-rollout (last-20%) macro F1, mean +/- std over **{n_seeds} seeds** "
        f"{', '.join(SEED_RUN_NAMES)} (D-05). bf16 AMP + cuDNN are not "
        "byte-deterministic, so the seed spread IS the honest run-to-run variation "
        "(Pitfall 9 -- mean+/-std, never bit-determinism).",
        "",
        "| Case | late-rollout F1 (mean +/- std) | n seeds |",
        "|---|---:|---:|",
    ]
    for case, mean, std, n in rows:
        lines.append(f"| {case} | {mean:.4f} +/- {std:.4f} | {n} |")
    lines += [
        f"| **OVERALL (per-case mean)** | **{overall:.4f}** | {n_seeds} |",
        "",
    ]

    # One-sided Wilcoxon TAU > frozen ConvLSTM (16 BASE cases), via Phase-4 reuse.
    cnn_dir = _abs(CNN_REF_REL)
    wil_line = None
    try:
        import significance
        cnn_map = significance._cnn_percase(cnn_dir)
        tau_pc = seed_aggregate.tau_percase([str(d) for d in seed_dirs])
        paired = sorted(set(dict(tau_pc)) & set(cnn_map))
        if cnn_map and len(paired) == significance.N_BASE_CASES:
            cases, tau, cnn = significance.pair_cases(tau_pc, cnn_map)
            res, d = significance.wilcoxon_tau_gt_cnn(tau, cnn)
            wil_line = (
                f"**One-sided Wilcoxon** (TAU > frozen ConvLSTM, n="
                f"{len(d)}): W={res.statistic}, p={res.pvalue:.4g}, "
                f"median delta={statistics.median(d):+.4f} (D-06, exact small-n)."
            )
        else:
            wil_line = (
                f"_{NOT_AVAILABLE}: the {significance.N_BASE_CASES}-case from-scratch "
                "ConvLSTM reference (`base_regen`) is not present in this checkout "
                f"(found {len(paired)} paired case(s)); the Wilcoxon p is computed in "
                "the on-cluster significance run. No p-value is fabricated here "
                "(T-05-16)._"
            )
    except Exception as e:  # noqa: BLE001 -- honest degradation, never fabricate
        wil_line = (
            f"_{NOT_AVAILABLE}: significance step unavailable ({type(e).__name__}); "
            "no p-value fabricated (T-05-16)._"
        )
    lines += [wil_line, ""]
    lines += [
        f"> **Provenance (DATA-04):** winning `tau_refined/checkpoints/best.pt` "
        f"sha256 `{provenance_sha}`. Per-seed `best.pt` sha256:",
        "",
    ]
    for name, sha in shas.items():
        lines.append(f"> - `{name}`: `{sha[:12]}`")
    lines += [
        "",
        f"> **From-scratch ConvLSTM reproduction scope (D-08):** {REPRO_SCOPE} "
        "(corrupt MS206 SED reference exports excluded -- honest verifiable scope).",
        "",
    ]
    return lines


# ---- per-case loaded artifacts ----
def _load_cases(eval_dir: Path) -> List[str]:
    return sorted(p.parent.name for p in eval_dir.glob("*/per_frame_metrics.csv"))


def _representative_case(eval_dir: Path, cases: Sequence[str]) -> str:
    """Median-macro-F1 case (a representative, not the best/worst), for figures."""
    scored = []
    for c in cases:
        try:
            m = macro_metrics_from_csv(eval_dir / c / "per_frame_metrics.csv")
            scored.append((m["macro_f1"], c))
        except Exception:  # noqa: BLE001
            continue
    if not scored:
        return cases[0]
    scored.sort()
    return scored[len(scored) // 2][1]


# ---- section 2: boundary metrics (PHYS-02 / D-03) ----
def _boundary_section(eval_dir: Path, cases: Sequence[str], thr: float) -> List[str]:
    lines = [
        "## 2. Boundary metrics (HD95 + boundary-F1)",
        "",
        "Crack-front localization on the binarized (no-healing) AR prediction vs "
        "GT, per D-03: **HD95** (reviewer-familiar 95th-pct Hausdorff; empty-mask "
        "-> NaN, excluded from the mean) and **boundary-F1** (BF score within 2 px). "
        "Per-frame metrics are averaged within each case.",
        "",
        "| Case | mean HD95 (px) | mean boundary-F1 |",
        "|---|---:|---:|",
    ]
    hd_all, bf_all = [], []
    for case in cases:
        try:
            probs, gt = load_case_probs_gt(eval_dir / case)
        except FileNotFoundError:
            lines.append(f"| {case} | _{NOT_AVAILABLE}_ | _{NOT_AVAILABLE}_ |")
            continue
        pred = _binarize_no_healing(probs, thr)
        gtb = _gt_binary(gt)
        n = min(pred.shape[0], gtb.shape[0])
        hds = [hd95(pred[i], gtb[i]) for i in range(n)]
        bfs = [boundary_f1(pred[i], gtb[i]) for i in range(n)]
        hd_mean = float(np.nanmean(hds)) if np.any(~np.isnan(hds)) else float("nan")
        bf_mean = float(np.mean(bfs))
        hd_all.append(hd_mean)
        bf_all.append(bf_mean)
        hd_cell = "NaN" if np.isnan(hd_mean) else f"{hd_mean:.3f}"
        lines.append(f"| {case} | {hd_cell} | {bf_mean:.4f} |")
    if hd_all:
        macro_hd = float(np.nanmean(hd_all))
        macro_bf = float(np.mean(bf_all))
        lines.append(
            f"| **MACRO (over {len(bf_all)} cases)** | **{macro_hd:.3f}** | "
            f"**{macro_bf:.4f}** |"
        )
    lines.append("")
    return lines


# ---- section 3: physical metrics (PHYS-03 / D-02) ----
def _physical_section(
    eval_dir: Path, cases: Sequence[str], thr: float, fig_dir: Path, rep_case: str
) -> List[str]:
    lines = [
        "## 3. Physical metrics (crack length-over-time + onset)",
        "",
        "Crack length = skeleton-pixel count per frame (reported in PIXELS; "
        "pixel-pitch left at 1.0 -- A3 honesty); time-to-onset = first frame with "
        "any predicted crack. GT vs FractureTAU (AR). The length-over-time figure "
        "below is the representative case (median macro F1).",
        "",
        "| Case | onset (GT) | onset (pred) | final length GT (px) | final length pred (px) |",
        "|---|---:|---:|---:|---:|",
    ]
    rep_png_rel = None
    for case in cases:
        try:
            probs, gt = load_case_probs_gt(eval_dir / case)
        except FileNotFoundError:
            lines.append(f"| {case} | _{NOT_AVAILABLE}_ | | | |")
            continue
        pred = _binarize_no_healing(probs, thr)
        gtb = _gt_binary(gt)
        pred_len = crack_length_curve(pred)
        gt_len = crack_length_curve(gtb)
        o_gt = time_to_onset(gtb)
        o_pred = time_to_onset(pred)
        lines.append(
            f"| {case} | {o_gt} | {o_pred} | {gt_len[-1]:.1f} | {pred_len[-1]:.1f} |"
        )
        if case == rep_case:
            out_png = fig_dir / f"length_{case}.png"
            length_over_time_fig(pred, gtb, out_png)
            length_over_time_fig(pred, gtb, out_png.with_suffix(".pdf"))  # D-06 vector
            rep_png_rel = _fig_rel(out_png, fig_dir)
    lines.append("")
    if rep_png_rel:
        lines += [
            f"![crack length-over-time ({rep_case})]({rep_png_rel})",
            "",
        ]
    return lines


# ---- section 4: error accumulation TF-vs-AR (PHYS-04 / D-13/D-14) ----
def _error_accum_section(
    eval_dir: Path, cases: Sequence[str], fig_dir: Path, rep_case: str
) -> List[str]:
    lines = [
        "## 4. Error accumulation (teacher-forced vs autoregressive F1)",
        "",
        "Per-frame F1 gap `TF - AR` over the rollout horizon (D-13 fidelity "
        "evidence): how far the autoregressive rollout drifts from the "
        "teacher-forced pass as the horizon grows. Computed from the REAL "
        "`per_frame_metrics_tf.csv` (D-14 teacher-forced pass, inference-only / "
        "freeze-legal) against the AR `per_frame_metrics.csv`, reusing the D-03 F1 "
        "(never the stored `f1` column, S4).",
        "",
        "| Case | mean gap (TF-AR) | late-rollout gap (last 20%) |",
        "|---|---:|---:|",
    ]
    rep_png_rel = None
    any_real = False
    for case in cases:
        ar_csv = eval_dir / case / "per_frame_metrics.csv"
        tf_csv = eval_dir / case / "per_frame_metrics_tf.csv"
        if not tf_csv.exists():
            # Honest degradation -- DEFENSE ONLY (the TF pass must run first, D-14).
            lines.append(
                f"| {case} | _teacher-forced pass not yet run_ | |"
            )
            continue
        try:
            gap = gap_from_csvs(ar_csv, tf_csv)
        except ValueError:
            lines.append(f"| {case} | _AR/TF horizon length mismatch_ | |")
            continue
        any_real = True
        late_n = max(1, int(round(0.20 * len(gap))))
        mean_gap = float(np.mean(gap))
        late_gap = float(np.mean(gap[-late_n:]))
        lines.append(f"| {case} | {mean_gap:+.4f} | {late_gap:+.4f} |")
        if case == rep_case:
            out_png = fig_dir / f"f1_horizon_{case}.png"
            ar_rows = _read_count_rows(ar_csv)
            tf_rows = _read_count_rows(tf_csv)
            f1_vs_horizon_fig(ar_rows, tf_rows, out_png)
            f1_vs_horizon_fig(ar_rows, tf_rows, out_png.with_suffix(".pdf"))  # D-06 vector
            rep_png_rel = _fig_rel(out_png, fig_dir)
    lines.append("")
    # PHYS-04 must-have: the figure is emitted from the REAL TF CSV. If the
    # representative case mismatched, fall back to the first case that has a real
    # gap so the figure is always present when TF data exists.
    if rep_png_rel is None and any_real:
        for case in cases:
            ar_csv = eval_dir / case / "per_frame_metrics.csv"
            tf_csv = eval_dir / case / "per_frame_metrics_tf.csv"
            if not tf_csv.exists():
                continue
            try:
                ar_rows = _read_count_rows(ar_csv)
                tf_rows = _read_count_rows(tf_csv)
                out_png = fig_dir / f"f1_horizon_{case}.png"
                f1_vs_horizon_fig(ar_rows, tf_rows, out_png)
                f1_vs_horizon_fig(ar_rows, tf_rows, out_png.with_suffix(".pdf"))  # D-06 vector
                rep_png_rel = _fig_rel(out_png, fig_dir)
                rep_case = case
                break
            except ValueError:
                continue
    if rep_png_rel:
        lines += [
            f"![TF vs AR F1 over horizon ({rep_case})]({rep_png_rel})",
            "",
        ]
    else:
        lines += [
            f"_{NOT_AVAILABLE}: no `per_frame_metrics_tf.csv` found -- run the "
            "teacher-forced eval pass (D-14) before this driver (defense only)._",
            "",
        ]
    return lines


# ---- section 5: calibration (PHYS-05 / D-13) ----
def _calibration_section(
    eval_dir: Path, cases: Sequence[str], fig_dir: Path
) -> List[str]:
    lines = [
        "## 5. Calibration (reliability + ECE)",
        "",
        "Reliability of the raw predicted probabilities over the rollout, binned by "
        "PREDICTED PROBABILITY (never a binarized mask, Pitfall 5). ECE per case + "
        "an aggregate reliability diagram over all cases.",
        "",
        "| Case | ECE |",
        "|---|---:|",
    ]
    all_probs, all_gts = [], []
    for case in cases:
        try:
            probs, gt = load_case_probs_gt(eval_dir / case)
        except FileNotFoundError:
            lines.append(f"| {case} | _{NOT_AVAILABLE}_ |")
            continue
        ece, _pop = reliability_ece(probs, _gt_binary(gt))
        lines.append(f"| {case} | {ece:.4f} |")
        all_probs.append(np.asarray(probs).ravel())
        all_gts.append(_gt_binary(gt).ravel())
    lines.append("")
    if all_probs:
        cat_p = np.concatenate(all_probs)
        cat_g = np.concatenate(all_gts)
        out_png = fig_dir / "reliability.png"
        ece_all, _pop = reliability_fig(cat_p, cat_g, out_png)
        reliability_fig(cat_p, cat_g, out_png.with_suffix(".pdf"))  # D-06 vector
        rel_rel = _fig_rel(out_png, fig_dir)
        lines += [
            f"Aggregate ECE over {len(all_probs)} cases: **{ece_all:.4f}**.",
            "",
            f"![reliability diagram]({rel_rel})",
            "",
        ]
    return lines


# ---- section 6: rollout stability band ----
def _stability_section(
    eval_dir: Path, cases: Sequence[str], fig_dir: Path
) -> List[str]:
    curves = []
    for case in cases:
        rows = _read_rows(eval_dir / case / "per_frame_metrics.csv")
        curves.append(case_f1_curve_from_rows(rows))
    out_png = fig_dir / "stability.png"
    stability_band(curves, out_png, label=f"median F1 ({len(curves)} cases)")
    stability_band(curves, out_png.with_suffix(".pdf"),
                   label=f"median F1 ({len(curves)} cases)")  # D-06 vector
    rel = _fig_rel(out_png, fig_dir)
    return [
        "## 6. Rollout stability",
        "",
        "Per-case foreground-F1 curves resampled onto a common normalized rollout "
        "fraction [0,1] grid; band = median + IQR (25-75%) across cases (reused from "
        "`dashboard.plots.stability_band`, S4).",
        "",
        f"![rollout stability band]({rel})",
        "",
    ]


# ---- section 7: compute efficiency (PHYS-06) ----
def _efficiency_section(run_dir: Path) -> List[str]:
    lines = [
        "## 7. Compute efficiency (params / MACs / FLOPs / A100-h)",
        "",
    ]
    ckpt = run_dir / "checkpoints" / "best.pt"
    if not ckpt.exists():
        lines += [f"_{NOT_AVAILABLE}: no `checkpoints/best.pt` under {run_dir.name}._", ""]
        return lines

    params = macs = flops = None
    note = ""
    try:
        import torch
        from src.config import Config
        from src.model import build_model
        from phys_metrics.efficiency import torch_efficiency

        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = state["model"]
        train_cfg = Config(**{k: v for k, v in state["config"].items()
                              if k in Config().__dict__})
        # c_in = in-channels of the first 4-D conv weight in the encoder.
        c_in = next(int(v.shape[1]) for v in sd.values()
                    if hasattr(v, "ndim") and v.ndim == 4)
        # H, W from a saved probs stack (cwd-independent, no grid path needed).
        H = W = None
        for case_dir in (run_dir / "eval").glob("*/"):
            try:
                probs, _gt = load_case_probs_gt(case_dir)
                H, W = int(probs.shape[1]), int(probs.shape[2])
                break
            except FileNotFoundError:
                continue
        model = build_model(train_cfg, c_in)
        model.load_state_dict(sd)
        T = int(train_cfg.sequence_length)
        eff = torch_efficiency(model, (1, T, c_in, H, W))
        params, macs, flops = eff["params"], eff["macs"], eff["flops"]
    except Exception as e:  # noqa: BLE001 -- honest degradation
        note = f" (FLOP introspection unavailable: {type(e).__name__})"
        try:
            import torch
            sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
            params = int(sum(int(np.prod(v.shape)) for v in sd.values()
                             if hasattr(v, "shape")))
        except Exception:  # noqa: BLE001
            params = None

    a100h = None
    log_csv = run_dir / "logs" / "train_log.csv"
    try:
        from phys_metrics.efficiency import a100_hours
        if log_csv.exists():
            a100h = a100_hours(log_csv, col="epoch_time_s")
    except Exception:  # noqa: BLE001
        a100h = None

    lines += [
        "FractureTAU compute, introspected from the frozen `best.pt`. MACs via "
        "fvcore/ptflops (manual conv/linear fallback if absent); **FLOPs = 2 x MACs** "
        "(Pitfall 4 -- report tool + convention). The frozen ConvLSTM reference is "
        "profiled in its OWN TensorFlow env and tabulated separately (cross-framework "
        "FLOP counters are not directly comparable).",
        "",
        "| Model | params | MACs (1 fwd) | FLOPs (1 fwd) | A100-hours |",
        "|---|---:|---:|---:|---:|",
    ]
    p_cell = f"{params:,}" if params is not None else NOT_AVAILABLE
    m_cell = f"{macs:.3e}" if macs is not None else f"{NOT_AVAILABLE}{note}"
    f_cell = f"{flops:.3e}" if flops is not None else NOT_AVAILABLE
    a_cell = f"{a100h:.3f}" if a100h is not None else NOT_AVAILABLE
    lines += [
        f"| FractureTAU (PyTorch) | {p_cell} | {m_cell} | {f_cell} | {a_cell} |",
        f"| ConvLSTM (frozen ref) | _tabulated in TF env_ | _tabulated_ | _tabulated_ | _tabulated_ |",
        "",
    ]
    return lines


# ---- driver ----
def build_report(
    run_name: str,
    *,
    run_dir: Optional[str | Path] = None,
    new_root: str | Path = NEW_ROOT,
    report_path: str | Path = REPORT_PATH,
    fig_dir: str | Path = FIG_DIR,
) -> Path:
    """Assemble the physical-metrics report for ``run_name`` and write PHYS_REPORT.md."""
    new_root = _abs(new_root)
    report_path = _abs(report_path)
    fig_dir = _abs(fig_dir)
    run_dir = _abs(run_dir) if run_dir else (new_root / run_name)
    eval_dir = run_dir / "eval"
    if not eval_dir.is_dir():
        raise FileNotFoundError(
            f"[phys] run eval dir not found: {eval_dir} "
            f"(expected {NEW_ROOT}/{run_name}/eval)"
        )

    cases = _load_cases(eval_dir)
    if not cases:
        raise FileNotFoundError(f"[phys] no <case>/per_frame_metrics.csv under {eval_dir}")
    print(f"[phys] {run_name}: {len(cases)} eval cases")

    ckpt = run_dir / "checkpoints" / "best.pt"
    provenance_sha = _sha256_short(ckpt) if ckpt.exists() else "unknown"
    thr = read_calibrated_threshold(run_dir)
    thr = DEFAULT_THRESHOLD if thr is None else thr
    print(f"[phys] calibrated threshold = {thr} ; best.pt sha {provenance_sha}")

    rep_case = _representative_case(eval_dir, cases)
    print(f"[phys] representative case (median macro F1) = {rep_case}")

    lines: List[str] = [
        f"# FractureTAU Physical-Metrics Report -- {run_name}",
        "",
        "Static, git-diffable physical-fidelity report (D-13). Regenerate with "
        f"`python results/phys_metrics/make_phys_report.py --run-name {run_name}`. "
        "Every reported number traces to a measured artifact "
        "(correctness-over-convenience mandate).",
        "",
    ]
    lines += _headline_section(new_root, provenance_sha)
    lines += _boundary_section(eval_dir, cases, thr)
    lines += _physical_section(eval_dir, cases, thr, fig_dir, rep_case)
    lines += _error_accum_section(eval_dir, cases, fig_dir, rep_case)
    lines += _calibration_section(eval_dir, cases, fig_dir)
    lines += _stability_section(eval_dir, cases, fig_dir)
    lines += _efficiency_section(run_dir)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")
    print(f"[phys] wrote {report_path}")
    return report_path


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Assemble the physical-metrics PHYS_REPORT.md (PHYS-02..06)."
    )
    ap.add_argument("--run-name", default=DEFAULT_RUN,
                    help=f"run dir under {NEW_ROOT} (default: {DEFAULT_RUN})")
    ap.add_argument("--run-dir", default=None,
                    help="explicit run dir (overrides --run-name resolution)")
    ap.add_argument("--new-root", default=NEW_ROOT)
    ap.add_argument("--report", default=REPORT_PATH)
    ap.add_argument("--figures", default=FIG_DIR)
    args = ap.parse_args(argv)
    build_report(
        args.run_name,
        run_dir=args.run_dir,
        new_root=args.new_root,
        report_path=args.report,
        fig_dir=args.figures,
    )


if __name__ == "__main__":
    main()
