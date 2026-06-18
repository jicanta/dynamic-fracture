# tests/test_seed_aggregate.py
"""HPC-02 (D-05): seed_aggregate computes per-case late-rollout macro F1
mean±std over the 3 headline seeds {42,43,44} — reusing the ONE shared metric.

Pins the locked contract for ``new_model/scripts/seed_aggregate.py``:
  * D-05 -- the headline number is a mean±std over 3 seeds (the seed spread is the
            honest run-to-run variation under non-byte-deterministic AMP/cuDNN).
  * Metric reuse -- the per-seed per-case scalar is ``verdict.late_rollout_macro_f1``
            (the ONE D-03-aware definition); ``seed_aggregate`` must NOT fork F1.
  * D-14 / S5 -- each seed dir is tagged with ``sha256_file(<run>/checkpoints/best.pt)``
            so every reported number traces to a checkpoint (threat T-04-02).

Framework rule: NO torch. stdlib + the framework-free verdict metric only, so it
runs in either conda env / CI without the PyTorch stack.

Run:  cd dynamic-fracture && python -m pytest tests/test_seed_aggregate.py -q
"""
from __future__ import annotations

import csv
import hashlib
import statistics
import sys
from pathlib import Path
from typing import Sequence, Tuple

import pytest

# ---- sys.path shims: repo root + scripts dir (module under test + verdict) ----
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "new_model" / "scripts"
for _p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import seed_aggregate  # noqa: E402  (module under test)

# Canonical per-frame schema (matches conftest.PER_FRAME_COLUMNS / evaluate.py).
_COLUMNS = ["frame_idx", "precision", "recall", "f1", "iou", "bce",
            "tp", "tn", "fp", "fn", "accuracy"]


def _write_case_csv(case_dir: Path, counts: Sequence[Tuple[int, int, int, int]]) -> None:
    """Write a minimal per_frame_metrics.csv (only tp/tn/fp/fn carry signal)."""
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


# Five frames per case -> late-rollout window k = max(1, round(0.2*5)) = 1, so the
# late-rollout macro F1 is controlled by the FINAL frame alone (exact + obvious).
_HEAD = [(1, 0, 0, 10)] * 4
_PERFECT = (10, 0, 0, 10)     # prec=rec=1  -> late F1 = 1.0
_HALF = (5, 5, 5, 10)         # prec=rec=.5 -> late F1 = 0.5
_ZERO = (0, 0, 10, 10)        # empty pred, nonempty GT -> late F1 = 0.0


def _layout_three_seeds(root: Path, per_case_tails: dict) -> None:
    """Build root/headline_s{42,43,44}/eval/<case>/per_frame_metrics.csv.

    ``per_case_tails`` maps case -> {seed: final-frame (tp,fp,fn,tn)}.
    """
    for case, by_seed in per_case_tails.items():
        for seed, tail in by_seed.items():
            _write_case_csv(root / f"headline_s{seed}" / "eval" / case,
                            _HEAD + [tail])


def test_per_case_meanstd_matches_pstdev_of_three_seeds(tmp_path):
    # caseA: seeds spread 1.0 / 0.5 / 0.0 ; caseB: all perfect -> std 0.
    _layout_three_seeds(tmp_path, {
        "caseA": {42: _PERFECT, 43: _HALF, 44: _ZERO},
        "caseB": {42: _PERFECT, 43: _PERFECT, 44: _PERFECT},
    })

    rows = seed_aggregate.per_case_meanstd(tmp_path)
    by_case = {r[0]: r for r in rows}

    a_f1s = [1.0, 0.5, 0.0]
    assert by_case["caseA"][1] == pytest.approx(statistics.mean(a_f1s))   # mean
    assert by_case["caseA"][2] == pytest.approx(statistics.pstdev(a_f1s))  # std
    assert by_case["caseA"][3] == 3                                        # n_seeds

    assert by_case["caseB"][1] == pytest.approx(1.0)
    assert by_case["caseB"][2] == pytest.approx(0.0)
    assert by_case["caseB"][3] == 3


def test_tau_percase_is_the_3seed_mean_for_significance(tmp_path):
    # The (case, mean) list significance.py pairs against the frozen ConvLSTM.
    _layout_three_seeds(tmp_path, {
        "caseA": {42: _PERFECT, 43: _HALF, 44: _ZERO},   # mean 0.5
        "caseB": {42: _PERFECT, 43: _PERFECT, 44: _PERFECT},  # mean 1.0
    })
    tp = dict(seed_aggregate.tau_percase(tmp_path))
    assert tp["caseA"] == pytest.approx(0.5)
    assert tp["caseB"] == pytest.approx(1.0)


def test_reuses_shared_verdict_metric_not_a_fork():
    # seed_aggregate must delegate the per-case scalar to verdict.late_rollout_macro_f1.
    assert hasattr(seed_aggregate, "late_rollout_macro_f1")


def test_missing_case_in_one_seed_raises(tmp_path):
    # caseA present in only 2 of 3 seeds -> cannot form a 3-seed mean -> fail loud.
    _write_case_csv(tmp_path / "headline_s42" / "eval" / "caseA", _HEAD + [_PERFECT])
    _write_case_csv(tmp_path / "headline_s43" / "eval" / "caseA", _HEAD + [_PERFECT])
    (tmp_path / "headline_s44" / "eval").mkdir(parents=True)
    with pytest.raises((SystemExit, ValueError)):
        seed_aggregate.per_case_meanstd(tmp_path)


def test_missing_seed_dir_raises(tmp_path):
    # No headline_s* dirs at all -> fail loud rather than silent empty output.
    (tmp_path / "nothing").mkdir()
    with pytest.raises((SystemExit, ValueError)):
        seed_aggregate.per_case_meanstd(tmp_path)


def test_seed_shas_hash_each_best_pt(tmp_path):
    # Provenance (D-14/S5): every seed dir's checkpoints/best.pt is sha256-hashed.
    expected = {}
    for seed in (42, 43, 44):
        ckpt = tmp_path / f"headline_s{seed}" / "checkpoints" / "best.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        payload = f"weights-seed-{seed}".encode()
        ckpt.write_bytes(payload)
        expected[f"headline_s{seed}"] = hashlib.sha256(payload).hexdigest()

    runs = [tmp_path / f"headline_s{s}" for s in (42, 43, 44)]
    shas = seed_aggregate.seed_shas(runs)
    for name, sha in expected.items():
        assert shas[name] == sha
