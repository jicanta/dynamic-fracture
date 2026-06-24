# tests/test_efficiency.py
"""PHYS-06: parameter-count / FLOPs / A100-hours golden test (RED, Wave 0).

Pins the contract Wave-2 ``results/phys_metrics/efficiency.py`` must satisfy:
  * ``torch_efficiency(model, input_shape)`` reports the exact trainable
    parameter count of a KNOWN tiny module (``Conv2d(1,2,3)`` -> 20) and a
    positive MAC/FLOP count for a given input shape.
  * ``a100_hours(training_log_csv, col)`` sums an epoch-time column (seconds) of
    a training log and returns wall-clock hours (7200 s -> 2.0 h).

Wave-0 state: ``phys_metrics`` does not exist yet -> COLLECTION ImportError
(the intended RED). Wave 2 turns it GREEN.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest  # noqa: F401

# ---- sys.path shim: repo root + results + results/phys_metrics ----
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
for _p in (str(REPO_ROOT), str(RESULTS), str(RESULTS / "phys_metrics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from phys_metrics.efficiency import a100_hours, torch_efficiency  # noqa: E402


def test_param_count_and_positive_flops(tiny_torch_module):
    module, expected_params = tiny_torch_module
    result = torch_efficiency(module, input_shape=(1, 1, 8, 8))
    assert int(result["params"]) == expected_params      # exact known count (20)
    assert float(result["macs"]) > 0.0                   # nonzero compute
    assert float(result["flops"]) > 0.0


def test_a100_hours_sums_epoch_time_column(tmp_path):
    log = tmp_path / "training_log.csv"
    with open(log, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "epoch_time_sec"])
        w.writeheader()
        for epoch, secs in enumerate([1800, 1800, 3600]):   # 7200 s total -> 2.0 h
            w.writerow({"epoch": epoch, "epoch_time_sec": secs})
    hours = a100_hours(str(log), "epoch_time_sec")
    assert hours == pytest.approx(2.0)
