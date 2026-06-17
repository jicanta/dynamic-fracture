# new_model/scripts/rank_variants.py
"""Rank Stage-2 sweep variants on the VAL-only selection metric.

Companion to ``sweep_stage2.sbatch``. After the directional sweep runs each
variant as a Stage-2-only fine-tune, this script ranks the variants and writes a
sorted ``ranking.csv``.

Two selection sources (``--source``):

  * ``train-log`` (DEFAULT, leakage-free) -- rank by each variant's BEST logged
    ``val_f1_ar`` in ``OUTPUTS/sweep/<variant>/logs/train_log.csv``. That scalar
    is the AR-rollout F1 on the temporal-tail validation split (``val_fraction``)
    -- the SAME signal ``best.pt`` is selected on. It needs no separate eval and
    never touches the 16 BASE TEST cases, so it cannot leak (D-06 / T-03-13).
    This replaces the original ``val_eval`` path, which could only be produced by
    ``src.evaluate`` for the 16 TEST cases (selection-time leakage).

  * ``val-eval`` (legacy) -- rank by the late-rollout (last-20%) AR macro F1
    re-derived via ``aggregate.selection_metric_from_eval_dir`` over
    ``OUTPUTS/sweep/<variant>/val_eval/<case>/per_frame_metrics.csv``. Retained
    for the case where a genuine (non-TEST) per-case val set is registered; the
    leakage guard refuses a ``val_eval`` dir that actually holds the BASE cases.

Decision IDs honored here:
  * D-05 -- the criterion is an AR-rollout val F1; the metric is never forked
            (train-log reuses the trainer's ``val_f1_ar``; val-eval imports the
            one ``aggregate`` seam).
  * D-06 -- selection never runs on the 16 BASE held-out TEST cases.

Framework rule (D-13): numpy/pandas/stdlib + the framework-free aggregate seam
ONLY. NO torch, NO tensorflow -- runs in either conda env.

Run (after the sweep produces OUTPUTS/sweep/):
    cd dynamic-fracture/new_model && \
        python -m scripts.rank_variants --sweep-root OUTPUTS/sweep
    # legacy per-case val set:
    #   python -m scripts.rank_variants --sweep-root OUTPUTS/sweep --source val-eval
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
# Imported lazily-tolerant: the DEFAULT train-log ranking does NOT need the
# dashboard package, and on the cluster only new_model/ is synced (no results/),
# so a hard top-level import would break `--source train-log`. Defer the failure
# to the --source val-eval path, which is the only consumer.
try:
    from results.dashboard.aggregate import (  # noqa: E402
        late_rollout_selection_metric,
        selection_metric_from_eval_dir,
    )
except ModuleNotFoundError:
    late_rollout_selection_metric = None       # type: ignore
    selection_metric_from_eval_dir = None       # type: ignore
from case_registry import TEST_CASE_FOLDERS  # noqa: E402

DEFAULT_EVAL_SUBDIR = "val_eval"
DEFAULT_LOG_SUBPATH = "logs/train_log.csv"
# >= half of the 16 held-out cases overlapping -> almost certainly the TEST set.
_LEAKAGE_OVERLAP_THRESHOLD = 8


# ---- train-log source (DEFAULT, leakage-free): best logged val_f1_ar ----
def _best_val_f1_ar(train_log: str | Path) -> float:
    """Max ``val_f1_ar`` over all logged epochs in a variant's train_log.csv.

    ``val_f1_ar`` is the AR-rollout F1 on the temporal-tail validation split
    (``val_fraction``) -- the very scalar ``best.pt`` is selected on. Taking the
    max over the run mirrors best-checkpoint selection. Leakage-free: it is
    computed entirely on the trainer's val split, never the 16 BASE TEST cases.
    """
    path = Path(train_log)
    with open(path, newline="") as f:
        vals = [
            float(r["val_f1_ar"])
            for r in csv.DictReader(f)
            if r.get("val_f1_ar") not in (None, "")
        ]
    if not vals:
        raise ValueError(f"[rank] no val_f1_ar values in {path}")
    return max(vals)


def rank_variants_trainlog(
    sweep_root: str | Path, log_subpath: str = DEFAULT_LOG_SUBPATH
) -> List[Tuple[str, float]]:
    """Rank each ``<sweep_root>/<variant>`` by its best logged ``val_f1_ar``.

    Returns ``[(variant, best_val_f1_ar), ...]`` sorted DESCENDING (winner
    first). Fails loud if no variant has a ``<log_subpath>`` train log.
    """
    root = Path(sweep_root)
    if not root.is_dir():
        raise SystemExit(f"[rank] --sweep-root is not a directory: {root}")

    results: List[Tuple[str, float]] = []
    for variant_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        log = variant_dir / log_subpath
        if not log.is_file():
            continue
        try:
            score = _best_val_f1_ar(log)
        except ValueError:
            # A variant that crashed before its first epoch leaves a header-only
            # (or empty) train_log.csv. Skip it with a warning instead of aborting
            # the whole ranking — the other variants are still valid.
            print(f"[rank] WARNING: {variant_dir.name} has no logged val_f1_ar "
                  f"(crashed/empty) — skipping")
            continue
        results.append((variant_dir.name, score))

    if not results:
        raise ValueError(
            f"[rank] no <variant>/{log_subpath} with a usable val_f1_ar under {root}"
        )
    results.sort(key=lambda t: t[1], reverse=True)
    return results


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
    if selection_metric_from_eval_dir is None:
        raise SystemExit(
            "[rank] --source val-eval needs results.dashboard.aggregate, which is "
            "not importable here (the dashboard package isn't on this machine). Use "
            "the default --source train-log, or run from a checkout that includes "
            "results/.")

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


def write_ranking(
    results: Sequence[Tuple[str, float]],
    out_path: str | Path,
    score_label: str = "late_rollout_f1",
) -> Path:
    """Write the sorted (variant, <score_label>) ranking to a CSV."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", score_label])
        for name, score in results:
            w.writerow([name, f"{score:.6f}"])
    return path


def main(argv: Sequence[str] | None = None) -> List[Tuple[str, float]]:
    ap = argparse.ArgumentParser(
        description="Rank Stage-2 sweep variants by the val-only selection "
                    "metric (D-05/D-06)."
    )
    ap.add_argument(
        "--sweep-root", required=True,
        help="Sweep dir whose children are the per-variant run dirs "
             "(e.g. OUTPUTS/sweep).",
    )
    ap.add_argument(
        "--source", choices=("train-log", "val-eval"), default="train-log",
        help="Selection source. 'train-log' (default, leakage-free): best logged "
             "val_f1_ar per variant. 'val-eval': late-rollout macro F1 over a "
             "registered per-case VAL set (legacy; never the 16 BASE TEST cases).",
    )
    ap.add_argument(
        "--eval-subdir", default=DEFAULT_EVAL_SUBDIR,
        help=f"(--source val-eval) per-variant eval subdir holding the VAL cases "
             f"(default: {DEFAULT_EVAL_SUBDIR}).",
    )
    ap.add_argument(
        "--log-subpath", default=DEFAULT_LOG_SUBPATH,
        help=f"(--source train-log) per-variant train-log path "
             f"(default: {DEFAULT_LOG_SUBPATH}).",
    )
    ap.add_argument(
        "--out", default=None,
        help="Output ranking.csv path (default: <sweep-root>/ranking.csv).",
    )
    args = ap.parse_args(argv)

    if not args.sweep_root:
        raise SystemExit("[rank] --sweep-root is required")

    if args.source == "train-log":
        results = rank_variants_trainlog(args.sweep_root, args.log_subpath)
        score_label = "best_val_f1_ar"
    else:
        results = rank_variants(args.sweep_root, args.eval_subdir)
        score_label = "late_rollout_f1"

    out_path = Path(args.out) if args.out else Path(args.sweep_root) / "ranking.csv"
    write_ranking(results, out_path, score_label=score_label)

    print(f"[rank] wrote {out_path}  (source={args.source})")
    for i, (name, score) in enumerate(results, start=1):
        print(f"[rank] {i}. {name}: {score_label}={score:.6f}")
    print(f"[rank] WINNER: {results[0][0]} ({score_label}={results[0][1]:.6f})")
    return results


if __name__ == "__main__":
    main()
