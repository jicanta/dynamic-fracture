# tests/test_rank_variants.py
"""MODEL-01/02/04 (D-05/D-06): rank_variants ranks sweep variants on the
val-only late-rollout selection seam — and refuses test-set leakage.

Pins the locked contract for ``new_model/scripts/rank_variants.py``:
  * D-05 -- variants are ranked by the late-rollout (last-20%) AR macro F1
            re-derived through ``aggregate.selection_metric_from_eval_dir``
            (the ONE importable seam — rank_variants must NOT fork the metric).
  * D-06 -- inputs are VALIDATION-split per-case CSVs only; pointing the ranker
            at the 16 BASE held-out TEST cases is selection-time leakage and
            must raise (threat T-03-13).

Framework rule: NO torch. Uses only stdlib + the framework-free aggregate seam,
so it runs in either conda env and in CI without the PyTorch stack.

Run:  cd dynamic-fracture && python -m pytest tests/test_rank_variants.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import pytest

# ---- sys.path shims: repo root (case_registry / results) + scripts dir ----
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "new_model" / "scripts"
for _p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rank_variants  # noqa: E402  (module under test)
from case_registry import TEST_CASE_FOLDERS  # noqa: E402

# Canonical per-frame schema (matches conftest.PER_FRAME_COLUMNS / evaluate.py).
_COLUMNS = ["frame_idx", "precision", "recall", "f1", "iou", "bce",
            "tp", "tn", "fp", "fn", "accuracy"]


def _write_case_csv(case_dir: Path, counts: Sequence[Tuple[int, int, int, int]]) -> None:
    """Write a minimal per_frame_metrics.csv (only tp/tn/fp/fn carry signal).

    The aggregate seam re-derives F1 from the count columns under D-03, so the
    precision/recall/f1/iou/bce/accuracy columns are written as 0.0 placeholders
    — the ranker never reads the stored ``f1`` column (D-01).
    """
    import csv

    case_dir.mkdir(parents=True, exist_ok=True)
    path = case_dir / "per_frame_metrics.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLUMNS)
        w.writeheader()
        for i, (tp, fp, fn, tn) in enumerate(counts):
            row = {c: 0.0 for c in _COLUMNS}
            row.update({"frame_idx": i, "tp": int(tp), "tn": int(tn),
                        "fp": int(fp), "fn": int(fn)})
            w.writerow(row)


def _make_variant(sweep_root: Path, variant: str, case: str,
                  counts: Sequence[Tuple[int, int, int, int]]) -> None:
    """Lay out <sweep_root>/<variant>/val_eval/<case>/per_frame_metrics.csv."""
    _write_case_csv(sweep_root / variant / "val_eval" / case, counts)


# Five frames per case -> late-rollout window k = max(1, round(0.2*5)) = 1
# (only the LAST frame is scored), so the late-rollout F1 is controlled by the
# final tuple alone — making the expected ranking exact and obvious.
_HEAD = [(1, 0, 0, 10)] * 4
_PERFECT_TAIL = (10, 0, 0, 10)     # prec=rec=1 -> late F1 = 1.0
_HALF_TAIL = (5, 5, 5, 10)         # prec=rec=0.5 -> late F1 = 0.5


def test_ranking_orders_by_late_rollout_and_picks_winner(tmp_path):
    sweep_root = tmp_path / "sweep"
    # "noisy" has a degraded last frame; "pushforward" stays perfect long-horizon.
    _make_variant(sweep_root, "pushforward", "val_run_a", _HEAD + [_PERFECT_TAIL])
    _make_variant(sweep_root, "noisy", "val_run_a", _HEAD + [_HALF_TAIL])

    ranked: List[Tuple[str, float]] = rank_variants.rank_variants(sweep_root)

    # higher late-rollout F1 first; winner is the long-horizon-stable variant.
    assert [name for name, _ in ranked] == ["pushforward", "noisy"]
    assert ranked[0][0] == "pushforward"
    assert ranked[0][1] == pytest.approx(1.0)
    assert ranked[1][1] == pytest.approx(0.5)


def test_uses_the_val_only_seam_not_a_forked_metric():
    # rank_variants must delegate to the importable aggregate seam.
    assert hasattr(rank_variants, "selection_metric_from_eval_dir") or \
        hasattr(rank_variants, "late_rollout_selection_metric")


def test_writes_sorted_ranking_csv(tmp_path):
    import csv

    sweep_root = tmp_path / "sweep"
    _make_variant(sweep_root, "pushforward", "val_run_a", _HEAD + [_PERFECT_TAIL])
    _make_variant(sweep_root, "noisy", "val_run_a", _HEAD + [_HALF_TAIL])

    out = tmp_path / "ranking.csv"
    rank_variants.main(["--sweep-root", str(sweep_root), "--out", str(out)])

    assert out.exists()
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["variant"] for r in rows] == ["pushforward", "noisy"]
    assert float(rows[0]["late_rollout_f1"]) == pytest.approx(1.0)


def test_refuses_test_set_leakage(tmp_path):
    # A variant whose val_eval dir actually holds the 16 BASE TEST cases is
    # selection-time leakage (D-06 / T-03-13) and must raise, not rank.
    sweep_root = tmp_path / "sweep"
    for case in TEST_CASE_FOLDERS.keys():
        _write_case_csv(sweep_root / "leaky" / "val_eval" / case,
                        _HEAD + [_PERFECT_TAIL])
    with pytest.raises((ValueError, SystemExit)):
        rank_variants.rank_variants(sweep_root)


def test_missing_sweep_root_raises_systemexit():
    with pytest.raises(SystemExit):
        rank_variants.main([])
