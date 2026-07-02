# tests/test_significance.py
"""HPC-03 (D-06): significance runs the one-sided Wilcoxon signed-rank test that
FractureTAU > frozen ConvLSTM across the 16 BASE cases.

Pins the locked contract for ``new_model/scripts/significance.py``:
  * Sign convention -- the paired difference is ``d = TAU - CNN`` (a positive d
    means TAU wins), so an all-positive ``d`` yields a SMALL one-sided p-value.
  * ``alternative="greater"`` + ``zero_method="wilcox"`` -- H1 is median(d) > 0.
  * Exact small-n p-value -- for n=16 with no ties, ``mode="auto"`` uses the exact
    distribution; an all-positive length-16 vector gives the hand-checkable
    p = 1/2**16 (W+ = 16*17/2 = 136). An all-negative vector gives p = 1.0.

Pure scipy unit test on synthetic paired data — no torch, no fixtures beyond stdlib.

Run:  cd dynamic-fracture && python -m pytest tests/test_significance.py -q
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

# ---- sys.path shims: repo root + scripts dir (module under test) ----
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "new_model" / "scripts"
for _p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import significance  # noqa: E402  (module under test)

# 16 paired cases, the real n for the BASE held-out test set.
_N = 16


def test_sign_is_tau_minus_cnn_and_all_positive_gives_exact_small_p():
    # TAU strictly beats CNN on every case (distinct deltas -> no ties).
    tau = [0.50 + (i + 1) * 0.01 for i in range(_N)]
    cnn = [0.50] * _N

    res, d = significance.wilcoxon_tau_gt_cnn(tau, cnn)

    assert all(x > 0 for x in d)                       # sign convention TAU - CNN
    assert res.statistic == pytest.approx(136.0)       # W+ = 16*17/2 (all positive)
    assert res.pvalue == pytest.approx(1.0 / 2 ** _N)  # exact small-n one-sided p


def test_all_negative_gives_p_near_one():
    # CNN beats TAU everywhere -> the one-sided "TAU > CNN" p-value is ~1.
    tau = [0.50] * _N
    cnn = [0.50 + (i + 1) * 0.01 for i in range(_N)]

    res, d = significance.wilcoxon_tau_gt_cnn(tau, cnn)

    assert all(x < 0 for x in d)
    assert res.pvalue == pytest.approx(1.0)


def test_uses_greater_alternative_and_wilcox_zero_method():
    # Pin the exact one-sided API contract in the source (not two-sided / pratt).
    src = inspect.getsource(significance.wilcoxon_tau_gt_cnn)
    assert 'alternative="greater"' in src
    assert 'zero_method="wilcox"' in src


def test_unpaired_lengths_raise():
    with pytest.raises((ValueError, SystemExit)):
        significance.wilcoxon_tau_gt_cnn([0.6, 0.7], [0.5])


# ---- retargeted exact-count pairing (09-03): the new-case roster is n=3 ----

def test_pair_cases_wrong_count_raises():
    # Only 2 cases pair up, but the new roster demands exactly 3 -> fail loud.
    tau_pc = [("c1", 0.70), ("c2", 0.72)]
    cnn_map = {"c1": 0.50, "c2": 0.51}
    with pytest.raises(SystemExit):
        significance.pair_cases(tau_pc, cnn_map, expect_cases=3)


def test_pair_cases_matches_new_roster():
    # Exactly 3 common cases (the real new roster) -> pairs cleanly, no raise.
    tau_pc = [("c1", 0.70), ("c2", 0.72), ("c3", 0.68)]
    cnn_map = {"c1": 0.50, "c2": 0.51, "c3": 0.49}

    common, tau, cnn = significance.pair_cases(tau_pc, cnn_map, expect_cases=3)

    assert len(common) == 3 and len(tau) == 3 and len(cnn) == 3
    assert common == ["c1", "c2", "c3"]          # sorted case order
    assert tau == [0.70, 0.72, 0.68]
    assert cnn == [0.50, 0.51, 0.49]
