#!/usr/bin/env python3
"""Blunt head-to-head: FractureTAU vs frozen ConvLSTM on late-rollout macro F1.

The D-01 win bar = mean per-case late-rollout (last-20%, frac=0.20) macro F1 over
the 16 BASE cases. Reads each side's per_frame_metrics.csv and prints per-case
TAU/CNN plus the bottom-line means + delta (positive => FractureTAU wins).

Run from new_model/ (login node is fine -- pure CSV math, no GPU/torch):
    python scripts/verdict.py
Optional overrides:
    python scripts/verdict.py --tau OUTPUTS/tau_refined/eval --cnn ../kathleens-model/OUTPUTS/base_regen
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics as st
import sys


# ---- inlined from results/dashboard/aggregate.py (D-03 / D-04) so this script
# has ZERO non-stdlib deps and runs anywhere without syncing the dashboard pkg.
def _frame_f1_d03(tp: float, fp: float, fn: float) -> float:
    """Degenerate-aware per-frame F1: empty-GT AND empty-pred frame scores 1.0."""
    if tp + fp + fn == 0:
        return 1.0
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0


def _macro_f1_from_counts(rows) -> float:
    f1s = [_frame_f1_d03(float(r["tp"]), float(r["fp"]), float(r["fn"])) for r in rows]
    if not f1s:
        raise ValueError("no rows to aggregate")
    return sum(f1s) / len(f1s)


def late_rollout_macro_f1(rows, frac: float = 0.20) -> float:
    """Macro F1 over the last round(frac*n) frames (D-04), min 1 frame."""
    rows = list(rows)
    n = len(rows)
    if n == 0:
        raise ValueError("no rows to aggregate")
    k = max(1, round(frac * n))
    return _macro_f1_from_counts(rows[-k:])


def _read_rows(path: str) -> list:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", default="OUTPUTS/tau_refined/eval",
                    help="FractureTAU eval dir (contains <case>/per_frame_metrics.csv)")
    ap.add_argument("--cnn", default="../kathleens-model/OUTPUTS/base_regen",
                    help="frozen ConvLSTM regen dir (<case>/per_frame_metrics.csv)")
    ap.add_argument("--frac", type=float, default=0.20, help="late-rollout window")
    args = ap.parse_args()

    new = {
        os.path.basename(os.path.dirname(p)): p
        for p in glob.glob(os.path.join(args.tau, "**", "per_frame_metrics.csv"),
                           recursive=True)
    }
    if not new:
        sys.exit(f"[verdict] no FractureTAU CSVs under {args.tau!r}")

    rows = []
    for case, tau_csv in sorted(new.items()):
        cnn_csv = os.path.join(args.cnn, case, "per_frame_metrics.csv")
        tau_f1 = late_rollout_macro_f1(_read_rows(tau_csv), frac=args.frac)
        cnn_f1 = (late_rollout_macro_f1(_read_rows(cnn_csv), frac=args.frac)
                  if os.path.exists(cnn_csv) else None)
        rows.append((case, tau_f1, cnn_f1))

    print(f"{'case':30s} {'TAU':>8s} {'CNN':>8s} {'delta':>8s}")
    print("-" * 58)
    for case, a, b in rows:
        if b is None:
            print(f"{case:30s} {a:8.4f} {'NA':>8s} {'--':>8s}")
        else:
            print(f"{case:30s} {a:8.4f} {b:8.4f} {a-b:+8.4f}")

    paired = [(a, b) for _, a, b in rows if b is not None]
    n = len(paired)
    if n == 0:
        sys.exit("[verdict] no paired cases (ConvLSTM CSVs missing)")
    tau_mean = st.mean(a for a, _ in paired)
    cnn_mean = st.mean(b for _, b in paired)
    delta = tau_mean - cnn_mean
    print("-" * 58)
    print(f"{'MEAN (N=%d)' % n:30s} {tau_mean:8.4f} {cnn_mean:8.4f} {delta:+8.4f}")
    wins = sum(1 for _, a, b in rows if b is not None and a > b)
    print(f"\nFractureTAU wins {wins}/{n} cases.")
    if delta > 0.01:
        verdict = "WIN (clear)"
    elif delta > 0:
        verdict = "WIN (within-noise squeaker)"
    elif delta > -0.01:
        verdict = "LOSE (within-noise)"
    else:
        verdict = "LOSE (clear)"
    print(f"D-01 verdict: {verdict}  (TAU {tau_mean:.4f} vs CNN {cnn_mean:.4f}, delta {delta:+.4f})")


if __name__ == "__main__":
    main()
