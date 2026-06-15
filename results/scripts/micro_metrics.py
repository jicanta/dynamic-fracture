#!/usr/bin/env python3
"""Per-case micro-averaged metrics from per_frame_metrics.csv trees.

Usage:
    python3 results/scripts/micro_metrics.py <root> [--csv out.csv]

<root> is any directory containing <case>/per_frame_metrics.csv subdirs, e.g.
    new_model/OUTPUTS/tau_base/eval
    ALL_OUTPUTS/VISUALS/Prediction_Binary_Masks/BASE

Micro protocol: sum tp/fp/fn over every frame of the rollout, then compute
precision/recall/F1 from the sums. This matches the old pipeline's totals.
"""
import argparse
import csv
import glob
import os
import sys


def micro(path):
    # DictReader keys rows by header, so the canonical schema's extra provenance
    # columns (iou, frame_idx, ... ; A3) are simply ignored here — we read only
    # the tp/fp/fn/tn counts needed for the micro totals.
    tp = fp = fn = tn = 0
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        tp += int(r["tp"]); fp += int(r["fp"]); fn += int(r["fn"]); tn += int(r["tn"])
    p = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * rec / (p + rec) if p + rec else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)
    return dict(frames=len(rows), tp=tp, fp=fp, fn=fn,
                precision=p, recall=rec, f1=f1, accuracy=acc)


def collect(root):
    out = {}
    for path in sorted(glob.glob(os.path.join(root, "*", "per_frame_metrics.csv"))):
        case = os.path.basename(os.path.dirname(path))
        out[case] = micro(path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--csv", help="also write results to this CSV file")
    args = ap.parse_args()

    cases = collect(args.root)
    if not cases:
        sys.exit(f"no <case>/per_frame_metrics.csv found under {args.root}")

    cols = ["case", "frames", "tp", "fp", "fn", "precision", "recall", "f1", "accuracy"]
    print(f"{'case':30s} {'frames':>6s} {'P':>7s} {'R':>7s} {'F1':>7s} {'acc':>7s}")
    T = dict(tp=0, fp=0, fn=0)
    for case, m in cases.items():
        print(f"{case:30s} {m['frames']:6d} {m['precision']:7.4f} {m['recall']:7.4f} "
              f"{m['f1']:7.4f} {m['accuracy']:7.4f}")
        for k in T:
            T[k] += m[k]
    p = T["tp"] / (T["tp"] + T["fp"]) if T["tp"] + T["fp"] else 0.0
    r = T["tp"] / (T["tp"] + T["fn"]) if T["tp"] + T["fn"] else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    print(f"{'TOTAL (micro)':30s} {'':6s} {p:7.4f} {r:7.4f} {f1:7.4f}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for case, m in cases.items():
                w.writerow({"case": case, **m})
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
