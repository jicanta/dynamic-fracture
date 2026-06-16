# new_model/scripts/rank_variants.py
"""Rank Stage-2 sweep variants on the VAL-only late-rollout selection metric.

Companion to ``sweep_stage2.sbatch``. After the directional sweep runs each
variant as a Stage-2-only fine-tune and evaluates it on the VALIDATION split
into ``OUTPUTS/sweep/<variant>/val_eval/<case>/per_frame_metrics.csv``, this
script ranks the variants by the single importable selection seam and writes a
sorted ``ranking.csv`` (variant, late_rollout_f1).

Decision IDs honored here:
  * D-05 -- ranking criterion is the late-rollout (last-20%) AR macro F1,
            re-derived via ``aggregate.selection_metric_from_eval_dir``. This
            module NEVER forks the metric — it imports the one seam.
  * D-06 -- inputs are VALIDATION per-case CSVs only. The 16 BASE cases are the
            held-out TEST set; ranking on them is selection-time leakage
            (threat T-03-13), so the script refuses a ``val_eval`` dir that
            actually holds the BASE test cases.

Framework rule (D-13): numpy/pandas/stdlib + the framework-free aggregate seam
ONLY. NO torch, NO tensorflow — runs in either conda env.

Run (after the sweep produces OUTPUTS/sweep/):
    cd dynamic-fracture/new_model && \
        python -m scripts.rank_variants --sweep-root OUTPUTS/sweep
    # or:  python scripts/rank_variants.py --sweep-root OUTPUTS/sweep
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

# ---- repo-root sys.path shim ----
# scripts -> new_model -> dynamic-fracture; makes the framework-free
# results.dashboard.aggregate seam and case_registry importable.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The ONE val-only selection seam (do NOT re-implement the metric here, D-05).
from results.dashboard.aggregate import (  # noqa: E402
    late_rollout_selection_metric,
    selection_metric_from_eval_dir,
)
from case_registry import TEST_CASE_FOLDERS  # noqa: E402

DEFAULT_EVAL_SUBDIR = "val_eval"
# >= half of the 16 held-out cases overlapping -> almost certainly the TEST set.
_LEAKAGE_OVERLAP_THRESHOLD = 8


def _looks_like_test_set(case_names: Sequence[str]) -> bool:
    """True if the case dirs look like the 16 BASE held-out TEST cases (D-06).

    Matches against both the short keys and the folder-name values of
    ``case_registry.TEST_CASE_FOLDERS`` so neither naming convention slips a
    test-set eval dir past the leakage guard.
    """
    base = set(TEST_CASE_FOLDERS.keys()) | set(TEST_CASE_FOLDERS.values())
    overlap = set(case_names) & base
    return len(overlap) >= _LEAKAGE_OVERLAP_THRESHOLD


def rank_variants(sweep_root: str | Path,
                  eval_subdir: str = DEFAULT_EVAL_SUBDIR) -> List[Tuple[str, float]]:
    """Rank each ``<sweep_root>/<variant>/<eval_subdir>`` by the val-only seam.

    Returns ``[(variant, late_rollout_f1), ...]`` sorted DESCENDING (winner
    first). Refuses (``ValueError``) any variant whose ``eval_subdir`` looks like
    the 16 BASE held-out TEST cases (D-06 / T-03-13). Fails loud if no variant
    with the expected layout is found.
    """
    root = Path(sweep_root)
    if not root.is_dir():
        raise SystemExit(f"[rank] --sweep-root is not a directory: {root}")

    results: List[Tuple[str, float]] = []
    for variant_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        eval_dir = variant_dir / eval_subdir
        if not eval_dir.is_dir():
            continue
        case_names = [c.name for c in eval_dir.iterdir() if c.is_dir()]
        if _looks_like_test_set(case_names):
            raise ValueError(
                f"[rank] refusing to rank '{variant_dir.name}': {eval_dir} looks "
                f"like the 16 BASE held-out TEST cases — ranking on them is "
                f"selection-time leakage (D-06 / T-03-13). Point at VAL cases only."
            )
        score = selection_metric_from_eval_dir(eval_dir)
        results.append((variant_dir.name, score))

    if not results:
        raise ValueError(
            f"[rank] no <variant>/{eval_subdir}/<case>/per_frame_metrics.csv "
            f"found under {root}"
        )
    results.sort(key=lambda t: t[1], reverse=True)
    return results


def write_ranking(results: Sequence[Tuple[str, float]], out_path: str | Path) -> Path:
    """Write the sorted (variant, late_rollout_f1) ranking to a CSV."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "late_rollout_f1"])
        for name, score in results:
            w.writerow([name, f"{score:.6f}"])
    return path


def main(argv: Sequence[str] | None = None) -> List[Tuple[str, float]]:
    ap = argparse.ArgumentParser(
        description="Rank Stage-2 sweep variants by the val-only late-rollout "
                    "selection metric (D-05/D-06)."
    )
    ap.add_argument(
        "--sweep-root", required=True,
        help="Dir whose children are <variant>/<eval-subdir>/<case>/"
             "per_frame_metrics.csv (e.g. OUTPUTS/sweep). VAL cases only.",
    )
    ap.add_argument(
        "--eval-subdir", default=DEFAULT_EVAL_SUBDIR,
        help=f"Per-variant eval subdir holding the VAL cases "
             f"(default: {DEFAULT_EVAL_SUBDIR}).",
    )
    ap.add_argument(
        "--out", default=None,
        help="Output ranking.csv path (default: <sweep-root>/ranking.csv).",
    )
    args = ap.parse_args(argv)

    if not args.sweep_root:
        raise SystemExit("[rank] --sweep-root is required")

    results = rank_variants(args.sweep_root, args.eval_subdir)
    out_path = Path(args.out) if args.out else Path(args.sweep_root) / "ranking.csv"
    write_ranking(results, out_path)

    print(f"[rank] wrote {out_path}")
    for i, (name, score) in enumerate(results, start=1):
        print(f"[rank] {i}. {name}: late_rollout_f1={score:.6f}")
    print(f"[rank] WINNER: {results[0][0]} "
          f"(late_rollout_f1={results[0][1]:.6f})")
    return results


if __name__ == "__main__":
    main()
