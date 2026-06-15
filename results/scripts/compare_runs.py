#!/usr/bin/env python3
"""Compare a new-model run against the original ConvLSTM pipeline.

Usage:
    python3 results/scripts/compare_runs.py <run_name> [<run_name2> ...]

Reads:
    new_model/OUTPUTS/<run_name>/eval/<case>/per_frame_metrics.csv
    ALL_OUTPUTS/VISUALS/Prediction_Binary_Masks/<MODE>/<case>/per_frame_metrics.csv

Writes:
    results/baselines/convlstm_per_case_micro.csv      (refreshed every run)
    results/comparisons/<run_name>_vs_convlstm.csv
    results/comparisons/<run_name>_vs_convlstm.md

Run from the repo root.
"""
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from micro_metrics import collect, micro  # noqa: E402

# Repo root on sys.path so the canonical case registry (Plan 02) drives CASE_MAP.
REPO_ROOT = Path(__file__).resolve().parents[2]   # scripts -> results -> dynamic-fracture
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from case_registry import TEST_CASE_FOLDERS  # noqa: E402

OLD_ROOT = "ALL_OUTPUTS/VISUALS/Prediction_Binary_Masks"
NEW_ROOT = "new_model/OUTPUTS"
OUT_DIR = "results/comparisons"
BASELINE_CSV = "results/baselines/convlstm_per_case_micro.csv"

MODES = ["BASE", "SED", "VON", "PRESSURE"]

# Canonical-key identity map (D-03/C-03): regenerated baselines (Plan 08) live
# under the SAME 16 canonical keys as the new pipeline, so old<->new case names
# are 1:1. Replaces the 8 idiosyncratic legacy output-folder names.
CASE_MAP = {k: k for k in TEST_CASE_FOLDERS}


def collect_old():
    """{(mode, old_case): metrics} for every old-pipeline result."""
    out = {}
    for mode in sorted(os.listdir(OLD_ROOT)):
        mode_dir = os.path.join(OLD_ROOT, mode)
        if not os.path.isdir(mode_dir):
            continue
        for case, m in collect(mode_dir).items():
            out[(mode, case)] = m
    return out


def build_comparison_rows(new, old):
    """Comparison rows for cases present in BOTH pipelines (pure; testable).

    new: {case: metrics} from the new pipeline. old: {(mode, case): metrics}.
    Returns a list of dicts. Empty if there is no overlap (the caller must guard
    against indexing rows[0] — see the `if not rows` guard in main).
    """
    rows = []
    for old_case, new_case in CASE_MAP.items():
        if new_case not in new:
            continue
        nm = new[new_case]
        for mode in MODES:
            om = old.get((mode, old_case))
            if om is None:
                continue
            rows.append({
                "case": new_case, "old_mode": mode,
                "baseline_present": "yes",
                "old_f1": om["f1"], "new_f1": nm["f1"],
                "old_precision": om["precision"], "new_precision": nm["precision"],
                "old_recall": om["recall"], "new_recall": nm["recall"],
                "delta_f1": nm["f1"] - om["f1"],
            })
    return rows


def build_coverage(new, old):
    """Honest per-(case, mode) coverage over ALL 16 canonical cases (DATA-03).

    Reports baseline_present / new_present yes/no for every canonical case and
    mode, so missing baselines are stated explicitly rather than silently
    dropped (BASE is the only fully-regenerated mode in Phase 1)."""
    coverage = []
    for old_case, new_case in CASE_MAP.items():
        new_present = new_case in new
        for mode in MODES:
            coverage.append({
                "case": new_case, "mode": mode,
                "baseline_present": "yes" if (mode, old_case) in old else "no",
                "new_present": "yes" if new_present else "no",
            })
    return coverage


def main():
    runs = sys.argv[1:]
    if not runs:
        sys.exit(__doc__)

    old = collect_old()

    # refresh the baseline extract
    os.makedirs(os.path.dirname(BASELINE_CSV), exist_ok=True)
    with open(BASELINE_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "case", "frames", "precision", "recall", "f1", "accuracy"])
        for (mode, case), m in sorted(old.items()):
            w.writerow([mode, case, m["frames"],
                        f"{m['precision']:.4f}", f"{m['recall']:.4f}",
                        f"{m['f1']:.4f}", f"{m['accuracy']:.4f}"])
    print(f"wrote {BASELINE_CSV} ({len(old)} mode/case rows)")

    os.makedirs(OUT_DIR, exist_ok=True)
    for run in runs:
        eval_dir = os.path.join(NEW_ROOT, run, "eval")
        if not os.path.isdir(eval_dir):
            print(f"skipping {run}: {eval_dir} not found", file=sys.stderr)
            continue
        new = collect(eval_dir)

        # Honest coverage table over all 16 canonical cases x modes (DATA-03).
        coverage = build_coverage(new, old)
        cov_path = os.path.join(OUT_DIR, f"{run}_coverage.csv")
        with open(cov_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["case", "mode", "baseline_present", "new_present"])
            w.writeheader()
            for c in coverage:
                w.writerow(c)
        n_base = sum(1 for c in coverage if c["mode"] == "BASE" and c["baseline_present"] == "yes")
        print(f"wrote {cov_path} (BASE baseline present for {n_base}/{len(CASE_MAP)} cases)")

        rows = build_comparison_rows(new, old)

        # Empty-overlap guard (CMP-02): never IndexError on list(rows[0].keys()).
        if not rows:
            print(f"[compare] no overlapping cases for run '{run}'; "
                  f"new cases={sorted(new)} vs CASE_MAP keys={sorted(CASE_MAP)}",
                  file=sys.stderr)
            continue

        csv_path = os.path.join(OUT_DIR, f"{run}_vs_convlstm.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                            for k, v in r.items()})

        md_path = os.path.join(OUT_DIR, f"{run}_vs_convlstm.md")
        with open(md_path, "w") as f:
            f.write(f"# {run} vs original ConvLSTM (micro F1, full AR rollout)\n\n")
            f.write("Only cases present in both pipelines. `extra: none` runs are\n"
                    "input-equivalent to old BASE; SED/VON/PRESSURE had an extra channel.\n\n")
            f.write("| Case | Old mode | Old F1 | New F1 | ΔF1 | Old P/R | New P/R |\n")
            f.write("|---|---|---:|---:|---:|---|---|\n")
            for r in rows:
                f.write(f"| {r['case']} | {r['old_mode']} | {r['old_f1']:.4f} | "
                        f"{r['new_f1']:.4f} | {r['delta_f1']:+.4f} | "
                        f"{r['old_precision']:.2f}/{r['old_recall']:.2f} | "
                        f"{r['new_precision']:.2f}/{r['new_recall']:.2f} |\n")
        print(f"wrote {csv_path} and {md_path} ({len(rows)} comparison rows)")


if __name__ == "__main__":
    main()
