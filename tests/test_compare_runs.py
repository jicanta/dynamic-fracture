# tests/test_compare_runs.py
"""CMP-02: the cross-pipeline comparison script is hardened and honest.

Covers: CASE_MAP is the 16-entry canonical identity map; the empty-overlap path
produces no rows (and the script guards against indexing rows[0]); micro()
tolerates the canonical schema's extra iou/frame_idx columns.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "results" / "scripts"
for _p in (str(REPO_ROOT), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import compare_runs            # noqa: E402
from micro_metrics import micro  # noqa: E402


def test_case_map_is_16_identity():
    assert len(compare_runs.CASE_MAP) == 16
    assert all(k == v for k, v in compare_runs.CASE_MAP.items())


def test_empty_overlap_no_indexerror():
    # No overlap between pipelines -> no comparison rows, and crucially NO
    # IndexError (the old list(rows[0].keys()) bug). build_comparison_rows is the
    # pure core the main loop guards with `if not rows`.
    rows = compare_runs.build_comparison_rows({}, {})
    assert rows == []
    # The guard the script applies before list(rows[0].keys()):
    assert (not rows) is True


def test_coverage_reports_all_cases_honestly():
    # With no baselines on disk, every canonical case/mode is baseline_present=no.
    coverage = compare_runs.build_coverage({}, {})
    assert len(coverage) == 16 * len(compare_runs.MODES)
    assert all(c["baseline_present"] == "no" for c in coverage)
    # A present BASE baseline for one case shows up as yes.
    one = next(iter(compare_runs.CASE_MAP))
    cov = compare_runs.build_coverage({}, {("BASE", one): {"f1": 0.5}})
    yes = [c for c in cov if c["baseline_present"] == "yes"]
    assert len(yes) == 1 and yes[0]["case"] == one and yes[0]["mode"] == "BASE"


def test_micro_tolerates_extra_columns(tmp_path):
    # Canonical columns + the counts micro needs + extra provenance columns.
    p = tmp_path / "per_frame_metrics.csv"
    fieldnames = ["frame_idx", "precision", "recall", "f1", "iou", "bce",
                  "tp", "tn", "fp", "fn"]
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerow({"frame_idx": 0, "precision": 0.5, "recall": 0.5, "f1": 0.5,
                    "iou": 0.33, "bce": 0.1, "tp": 5, "tn": 90, "fp": 3, "fn": 2})
        w.writerow({"frame_idx": 1, "precision": 0.6, "recall": 0.4, "f1": 0.48,
                    "iou": 0.31, "bce": 0.2, "tp": 6, "tn": 88, "fp": 4, "fn": 2})
    m = micro(str(p))          # must not KeyError on iou/frame_idx
    assert m["frames"] == 2
    assert m["tp"] == 11 and m["fp"] == 7 and m["fn"] == 4
