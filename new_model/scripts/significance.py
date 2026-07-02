#!/usr/bin/env python3
"""One-sided Wilcoxon signed-rank test that FractureTAU > frozen ConvLSTM (HPC-03 / D-06).

D-06 asks whether FractureTAU's per-case late-rollout macro F1 is significantly
greater than the frozen ConvLSTM reference across the 16 BASE held-out cases. The
paired, distribution-free test for this is the one-sided Wilcoxon signed-rank:

    d = [t - c for t, c in zip(tau_percase, cnn_percase)]   # n = 16, sign = TAU - CNN
    res = scipy.stats.wilcoxon(d, alternative="greater", zero_method="wilcox", mode="auto")

The TAU side is the 3-seed MEAN per case (from ``seed_aggregate.tau_percase`` — D-05),
the CNN side is the deterministic frozen ConvLSTM. With n=16 and no ties, ``mode="auto"``
uses the EXACT distribution, so the reported p-value is exact (not a normal approx).

Provenance (D-14 / S5, threat T-04-02): the p-value is recorded alongside the 3 TAU
seed ``sha256_file(best.pt)`` hashes and the frozen CNN reference path, so the test
traces to specific checkpoints + the unmodified baseline (threat T-04-04).

Framework-free: stdlib + scipy + the framework-free ``verdict`` metric only — NO torch,
NO tensorflow. Runs on a login node in either conda env.

Run from new_model/:
    python -m scripts.significance \
        --tau OUTPUTS \
        --cnn ../kathleens-model/OUTPUTS/base_regen
  (``--tau`` is the parent holding headline_s{42,43,44}; ``--tau-csv`` accepts a
   precomputed seed_aggregate seed_meanstd.csv instead.)
"""
from __future__ import annotations

import argparse
import glob
import os
import statistics
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import scipy.stats as st  # NEW dep (scipy>=1.11); exact small-n Wilcoxon

# ---- repo-root + scripts + new_model sys.path shim (PATTERNS S1) ----
SCRIPTS_DIR = Path(__file__).resolve().parent
NEW_MODEL = SCRIPTS_DIR.parent
REPO_ROOT = NEW_MODEL.parent
for _p in (str(REPO_ROOT), str(NEW_MODEL), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse the ONE D-03-aware per-case scalar + the 3-seed TAU aggregation (no fork).
from verdict import late_rollout_macro_f1, _read_rows  # noqa: E402
import seed_aggregate  # noqa: E402  (tau_percase, seed_shas, discover_seed_runs)

# RESEARCH A5 / OQ2: this local checkout has only the 5-case base_repro/; the
# 16-case frozen ConvLSTM regen (base_regen) lives on Gilbreth and MUST be confirmed
# (gated in plan 06) before a real significance run. The CLI default mirrors
# verdict.py:60 so an on-cluster run is a no-flag invocation.
DEFAULT_CNN = "../kathleens-model/OUTPUTS/base_regen"
N_BASE_CASES = 16


def wilcoxon_tau_gt_cnn(
    tau_percase: Sequence[float], cnn_percase: Sequence[float]
):
    """One-sided Wilcoxon signed-rank test that median(TAU - CNN) > 0.

    ``tau_percase`` / ``cnn_percase`` are paired per-case scalars (same case order,
    same length). Returns ``(result, d)`` where ``result`` is the scipy
    ``WilcoxonResult`` and ``d = [TAU - CNN]`` is the paired-difference vector.

    Sign convention: ``d = TAU - CNN`` so ``alternative="greater"`` tests H1 that
    TAU beats CNN (a positive median delta). ``zero_method="wilcox"`` discards exact
    ties (rare with continuous F1; a degenerate case scoring 1.0 on both sides gives
    d=0). ``mode="auto"`` -> exact distribution for n<=25 with no ties.
    """
    if len(tau_percase) != len(cnn_percase):
        raise ValueError(
            f"[sig] unpaired vectors: len(tau)={len(tau_percase)} != "
            f"len(cnn)={len(cnn_percase)}")
    d = [float(t) - float(c) for t, c in zip(tau_percase, cnn_percase)]
    res = st.wilcoxon(d, alternative="greater", zero_method="wilcox", mode="auto")
    return res, d


def _cnn_percase(cnn_dir: str | Path, frac: float = 0.20) -> dict:
    """Map case-name -> late-rollout macro F1 for the frozen ConvLSTM regen dir
    (reuse verdict.py's glob/keying + the ONE shared metric)."""
    matches = glob.glob(
        os.path.join(str(cnn_dir), "**", "per_frame_metrics.csv"), recursive=True)
    out = {}
    for p in matches:
        case = os.path.basename(os.path.dirname(p))
        out[case] = late_rollout_macro_f1(_read_rows(p), frac=frac)
    return out


def _read_tau_csv(path: str | Path) -> List[Tuple[str, float]]:
    """Read a precomputed seed_aggregate seed_meanstd.csv -> [(case, mean), ...]."""
    import csv
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"[sig] empty tau CSV: {path}")
    return [(r["case"], float(r["late_rollout_f1_mean"])) for r in rows]


def pair_cases(
    tau_percase: Sequence[Tuple[str, float]], cnn_map: dict,
    expect_cases: int = N_BASE_CASES,
) -> Tuple[List[str], List[float], List[float]]:
    """Pair TAU and CNN per-case scalars by case name (sorted). Fails loud unless
    exactly ``expect_cases`` cases pair up (default 16 = the v1.0 BASE roster; the
    new-case run passes 3). The assert stays an EXACT-count fail-loud (``!=``, never
    relaxed to ``>=``/``>``) so a missing paired case still aborts (D-02/Pitfall 6)."""
    tau_map = dict(tau_percase)
    common = sorted(set(tau_map) & set(cnn_map))
    if len(common) != expect_cases:
        raise SystemExit(
            f"[sig] expected {expect_cases} paired cases, got {len(common)} "
            f"(tau={sorted(tau_map)}, cnn={sorted(cnn_map)}). The reference case set "
            f"must be confirmed (16 = v1.0 BASE; 3 = the new-case roster).")
    tau = [tau_map[c] for c in common]
    cnn = [cnn_map[c] for c in common]
    return common, tau, cnn


def main(argv: Sequence[str] | None = None):
    ap = argparse.ArgumentParser(
        description="One-sided Wilcoxon TAU>CNN over the 16 BASE cases (HPC-03/D-06).")
    ap.add_argument(
        "--tau", default="OUTPUTS",
        help="Parent dir holding headline_s{42,43,44} seed runs (TAU side = 3-seed "
             "mean per case, via seed_aggregate). Default: OUTPUTS.")
    ap.add_argument(
        "--tau-csv", default=None,
        help="Precomputed seed_aggregate seed_meanstd.csv (used instead of --tau).")
    ap.add_argument(
        "--cnn", default=DEFAULT_CNN,
        help=f"Frozen ConvLSTM regen dir (<case>/per_frame_metrics.csv). "
             f"Default: {DEFAULT_CNN} (RESEARCH A5/OQ2: confirm the 16-case "
             f"base_regen path before a real run — plan-06 gate).")
    ap.add_argument("--frac", type=float, default=0.20, help="late-rollout window")
    ap.add_argument(
        "--expect-cases", type=int, default=N_BASE_CASES,
        help="Exact number of paired cases required (16 = v1.0 BASE; 3 = the "
             "new-case roster for --case-set new).")
    ap.add_argument(
        "--eval-subdir", default="eval",
        help="Eval subdir under each seed run to read per_frame_metrics.csv from "
             "(eval = v1.0 canonical; eval_newcase = the Phase-9 new-case eval).")
    args = ap.parse_args(argv)

    # TAU side: 3-seed mean per case (or a precomputed CSV).
    if args.tau_csv:
        tau_pc = _read_tau_csv(args.tau_csv)
        seed_runs: List[Path] = []
    else:
        seed_runs = seed_aggregate.discover_seed_runs(
            args.tau, eval_subdir=args.eval_subdir)
        if not seed_runs:
            sys.exit(f"[sig] no headline_s* seed dirs under {args.tau!r}")
        tau_pc = seed_aggregate.tau_percase(
            args.tau, frac=args.frac, eval_subdir=args.eval_subdir)

    # CNN side: frozen ConvLSTM per case.
    cnn_map = _cnn_percase(args.cnn, frac=args.frac)
    if not cnn_map:
        sys.exit(f"[sig] no frozen-ConvLSTM CSVs under {args.cnn!r}")

    cases, tau, cnn = pair_cases(tau_pc, cnn_map, expect_cases=args.expect_cases)
    res, d = wilcoxon_tau_gt_cnn(tau, cnn)
    median_delta = statistics.median(d)
    n_nonzero = sum(1 for x in d if x != 0.0)

    print(f"{'case':30s} {'TAU':>8s} {'CNN':>8s} {'delta':>8s}")
    print("-" * 58)
    for c, t, k in zip(cases, tau, cnn):
        print(f"{c:30s} {t:8.4f} {k:8.4f} {t - k:+8.4f}")
    print("-" * 58)
    print(f"[sig] W={res.statistic}  p(one-sided TAU>CNN)={res.pvalue:.4g}  "
          f"n={len(d)}  (nonzero={n_nonzero})  median_delta={median_delta:+.4f}")

    # Provenance (D-14 / T-04-02 + T-04-04): hash the TAU seed checkpoints and
    # record the frozen CNN reference path next to the p-value.
    if seed_runs:
        print("[sig] TAU provenance (sha256 best.pt per seed):")
        for name, sha in seed_aggregate.seed_shas(seed_runs).items():
            print(f"[sig]   {name}: {sha}")
    print(f"[sig] CNN reference (frozen baseline): {os.path.abspath(args.cnn)}")
    return res


if __name__ == "__main__":
    main()
