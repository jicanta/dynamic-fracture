"""Shared pytest fixtures for the data-integrity / comparison-harness phase.

This conftest is the Wave-0 prerequisite for the whole phase. It provides:

  * a ``sys.path`` shim so the framework-free shared modules that live at the
    repo root (``dynamic-fracture/``) -- e.g. ``case_registry`` and
    ``frac_metrics``, created in Wave 2 -- import cleanly from tests;
  * a ``synth_runs`` fixture: a tiny in-memory, multi-run dataset shaped like
    the real grid data (channel 0 = binary fracture mask), with KNOWN and
    DISTINCT per-run frame counts so val-split disjointness can be asserted
    deterministically -- without touching Gilbreth scratch;
  * ``golden_masks`` / ``golden_degenerate`` fixtures: deterministic
    ``gt_bin`` / ``pred_bin`` / ``prob`` numpy arrays for the metrics-parity
    test (a real partial-overlap case and the degenerate all-zero 0/0 -> 0.0
    case).

All fixtures are SEEDED (numpy seed 42 / ``np.random.default_rng(42)``) so the
downstream parity and disjointness evidence is reproducible across runs
(threat T-01-01: a non-seeded fixture would make parity tests flaky and yield a
false integrity signal). The array internals mirror
``new_model/scripts/make_synth_data.py`` (tiny grid, binary mask in channel 0,
extra physics channel) but are built in-memory rather than shelled out.

Run:  cd dynamic-fracture && python -m pytest tests/ -q
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pytest

# ---- repo-root sys.path shim ----
# repo root = dynamic-fracture/ (parent of tests/); makes case_registry /
# frac_metrics importable once those modules land in Wave 2.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---- synthetic grid geometry ----
# Small stand-in for the real 321x161 grid; large enough to exercise
# scatter/crop logic, small enough to keep fixtures sub-millisecond.
SYNTH_H: int = 33          # grid height (rows)
SYNTH_W: int = 17          # grid width (cols)
SYNTH_C: int = 2           # channels: 0 = binary fracture mask, 1 = physics field
SEED: int = 42

# Three runs with DISTINCT frame counts so frame-boundary val splits are
# exercised (the split logic must not assume equal-length runs).
SYNTH_RUN_SPECS: List[Tuple[str, int]] = [
    ("run_V100", 20),
    ("run_V150", 26),
    ("run_V200", 31),
]


# ---- synth multi-run dataset fixture ----
@pytest.fixture
def synth_runs() -> List[Dict[str, object]]:
    """A tiny multi-run dataset shaped like the real preprocessed grids.

    Returns a list of per-run dicts::

        {"name": str, "n_frames": int, "features": np.ndarray}

    where ``features`` has shape ``(n_frames, C, H, W)`` (float32). Channel 0 is
    a binary fracture mask in {0, 1} that grows monotonically over frames (a
    no-healing crack, mirroring the real data's monotone fracture growth);
    channel 1 is a deterministic float physics field. Frame counts are KNOWN
    and DISTINCT (20, 26, 31) so val-split disjointness is assertable.
    """
    rng = np.random.default_rng(SEED)
    runs: List[Dict[str, object]] = []
    for name, n_frames in SYNTH_RUN_SPECS:
        features = np.zeros((n_frames, SYNTH_C, SYNTH_H, SYNTH_W), dtype=np.float32)
        # channel 0: monotonically growing binary mask (no-healing crack)
        for t in range(n_frames):
            grown_rows = int(round((t + 1) / n_frames * SYNTH_H))
            features[t, 0, :grown_rows, :] = 1.0
        # channel 1: deterministic physics-like float field in [0, 1)
        features[:, 1, :, :] = rng.random((n_frames, SYNTH_H, SYNTH_W)).astype(np.float32)
        runs.append({"name": name, "n_frames": n_frames, "features": features})
    return runs


# ---- golden mask fixtures (metrics parity) ----
@pytest.fixture
def golden_masks() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic ``(gt_bin, pred_bin, prob)`` for the metrics-parity test.

    Each array is shape ``(H, W)``:
      * ``gt_bin`` / ``pred_bin`` -- uint8 in {0, 1}, two PARTIALLY OVERLAPPING
        rectangles so tp/fp/fn are all nonzero (exercises a real F1, not a
        degenerate 0/0).
      * ``prob`` -- float in [0, 1] built so that
        ``(prob >= 0.5).astype(uint8) == pred_bin`` exactly, i.e. binarizing the
        probability map at 0.5 reproduces ``pred_bin``.
    """
    rng = np.random.default_rng(SEED)

    gt_bin = np.zeros((SYNTH_H, SYNTH_W), dtype=np.uint8)
    pred_bin = np.zeros((SYNTH_H, SYNTH_W), dtype=np.uint8)
    # gt rectangle and a shifted pred rectangle -> overlap (tp), gt-only (fn),
    # pred-only (fp) all guaranteed nonzero.
    gt_bin[5:20, 3:12] = 1
    pred_bin[8:23, 5:14] = 1

    # prob: < 0.5 where pred_bin == 0, >= 0.5 where pred_bin == 1.
    prob = np.where(
        pred_bin == 1,
        rng.uniform(0.5, 1.0, size=(SYNTH_H, SYNTH_W)),
        rng.uniform(0.0, 0.5, size=(SYNTH_H, SYNTH_W)),
    ).astype(np.float64)
    # invariant the parity test relies on
    assert np.array_equal((prob >= 0.5).astype(np.uint8), pred_bin)

    return gt_bin, pred_bin, prob


@pytest.fixture
def golden_degenerate() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All-zero ``(gt_bin, pred_bin, prob)`` to pin the 0/0 -> 0.0 convention.

    With no positives anywhere, tp = fp = fn = 0; precision/recall/F1 must
    resolve to 0.0 (not NaN). ``prob`` is all-zero so binarization at 0.5 also
    yields the all-zero ``pred_bin``.
    """
    gt_bin = np.zeros((SYNTH_H, SYNTH_W), dtype=np.uint8)
    pred_bin = np.zeros((SYNTH_H, SYNTH_W), dtype=np.uint8)
    prob = np.zeros((SYNTH_H, SYNTH_W), dtype=np.float64)
    return gt_bin, pred_bin, prob


# ---- multi-frame per-case CSV builder (aggregation fixtures) ----
# Canonical per-frame schema, canonical-first order. Matches what
# new_model/src/evaluate.py:110-118 emits: the canonical D-13 columns
# (frame_idx,precision,recall,f1,iou,bce) followed by the count + accuracy
# provenance columns the micro/aggregation layers read.
PER_FRAME_COLUMNS: List[str] = [
    "frame_idx", "precision", "recall", "f1", "iou", "bce",
    "tp", "tn", "fp", "fn", "accuracy",
]


def _frozen_seam_row(tp: float, fp: float, fn: float, tn: float) -> Dict[str, float]:
    """Per-frame metrics with the FROZEN seam convention (0/0 -> 0.0).

    Identical arithmetic to ``frac_metrics.per_frame_metrics`` (lines 82-86):
    the all-zero frame yields ``precision = recall = f1 = iou = 0.0``. This is
    intentionally the OPPOSITE of the D-03 aggregation convention (0/0 -> 1.0)
    so the stored ``f1`` column DIFFERS from the macro-F1 Plan 02 recomputes
    from the counts — that divergence is exactly what the Plan-02 tests pin.
    """
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0
    denom = tp + tn + fp + fn
    acc = (tp + tn) / denom if denom > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "iou": iou, "accuracy": acc}


@pytest.fixture
def per_frame_csv(tmp_path_factory) -> Callable[..., Path]:
    """FACTORY fixture: build a ``per_frame_metrics.csv`` from frame counts.

    Returns a callable ``build(counts, *, name="case")`` where ``counts`` is a
    sequence of per-frame ``(tp, fp, fn, tn)`` tuples. It writes one CSV row per
    frame into a fresh tmp ``<name>/per_frame_metrics.csv`` (mirroring the real
    per-case layout consumed by ``micro_metrics.collect``) and returns its Path.

    Columns are the canonical-first schema ``PER_FRAME_COLUMNS`` matching
    ``evaluate.py:110-118``. precision/recall/f1/iou are computed with the
    FROZEN seam convention (``0/0 -> 0.0``); ``bce`` is a ``0.0`` placeholder.
    Deterministic / no RNG (the underlying data is caller-supplied); any future
    randomness must be seeded with ``np.random.default_rng(SEED)`` (SEED=42).
    """
    def build(counts: Sequence[Tuple[float, float, float, float]], *, name: str = "case") -> Path:
        case_dir = tmp_path_factory.mktemp(name)
        path = case_dir / "per_frame_metrics.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=PER_FRAME_COLUMNS)
            w.writeheader()
            for frame_idx, (tp, fp, fn, tn) in enumerate(counts):
                m = _frozen_seam_row(float(tp), float(fp), float(fn), float(tn))
                w.writerow({
                    "frame_idx": frame_idx,
                    "precision": m["precision"], "recall": m["recall"],
                    "f1": m["f1"], "iou": m["iou"], "bce": 0.0,
                    "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
                    "accuracy": m["accuracy"],
                })
        return path
    return build


@pytest.fixture
def multi_frame_case(per_frame_csv) -> Path:
    """A default multi-frame case CSV: a growing crack of N=10 frames.

    Includes at least one all-zero frame (empty GT + empty pred -> the 0/0 seam
    case, frame 0) and one fully-overlapping frame (pred == GT, no fp/fn -> the
    last frame). Foreground area grows monotonically (no-healing crack), with
    the prediction lagging slightly so fp/fn are nonzero on the interior frames.
    Returns the written ``per_frame_metrics.csv`` Path.
    """
    total = SYNTH_H * SYNTH_W           # pixels per frame (561)
    n = 10
    counts: List[Tuple[int, int, int, int]] = []
    for t in range(n):
        if t == 0:                      # all-zero frame: empty GT + empty pred
            tp, fp, fn = 0, 0, 0
        elif t == n - 1:                # fully-overlapping frame: pred == GT
            gt = int(round(total * 0.5))
            tp, fp, fn = gt, 0, 0
        else:                           # interior: growing GT, lagging pred
            gt = int(round(total * (t / n) * 0.5))
            tp = int(round(gt * 0.85))
            fn = gt - tp
            fp = max(1, int(round(gt * 0.1)))
        tn = total - tp - fp - fn
        counts.append((tp, fp, fn, tn))
    return per_frame_csv(counts, name="multi_frame_case")


# ===================================================================== #
# Phase 5 (physical-metrics) fixtures — added in Plan 05-01 (Wave 0).    #
# These encode the golden invariants the RED PHYS tests assert and the   #
# Wave-2 phys_metrics modules must satisfy.                              #
# ===================================================================== #

# Crack-stack geometry (small stand-in; H/W chosen so a growing horizontal
# crack stays a single connected component under 4- and 8-connectivity).
CRACK_FRAMES: int = 8
CRACK_H: int = 32
CRACK_W: int = 48
CRACK_ONSET: int = 2          # first frame with >=1 positive pixel (PHYS-03 contract)


@pytest.fixture
def growing_crack_stack() -> np.ndarray:
    """Monotone non-healing crack stack; onset==2, length non-decreasing (PHYS-03).

    Returns a ``(CRACK_FRAMES=8, CRACK_H=32, CRACK_W=48)`` uint8 mask stack.
    Frames 0-1 are EMPTY (no positive pixels); from frame 2 onward a single
    connected horizontal crack on the centre row grows in length, each frame a
    strict superset of the previous one. Therefore: ``time_to_onset == 2`` (the
    first frame with >=1 positive pixel) and ``crack_length_curve`` is
    monotonically non-decreasing. Adapts the channel-0 monotone-growth idiom of
    ``synth_runs`` (conftest.py:81-83); fully deterministic (no RNG needed).
    """
    stack = np.zeros((CRACK_FRAMES, CRACK_H, CRACK_W), dtype=np.uint8)
    mid = CRACK_H // 2
    n_growing = CRACK_FRAMES - CRACK_ONSET            # frames 2..7 -> 6 steps
    for t in range(CRACK_ONSET, CRACK_FRAMES):
        step = t - CRACK_ONSET + 1                    # 1..n_growing
        length = int(round(step / n_growing * CRACK_W))   # 8,16,...,48 (<= W)
        length = max(1, min(CRACK_W, length))
        stack[t, mid, :length] = 1                    # contiguous => 1 component
    # invariants the PHYS-03 test relies on
    assert stack[:CRACK_ONSET].sum() == 0                       # empty before onset
    assert stack[CRACK_ONSET].sum() >= 1                        # onset frame nonempty
    areas = stack.reshape(CRACK_FRAMES, -1).sum(axis=1)
    assert np.all(np.diff(areas) >= 0)                         # non-decreasing
    return stack


@pytest.fixture
def perfectly_calibrated_probs() -> Tuple[np.ndarray, np.ndarray]:
    """Perfectly-calibrated (probs, gts) so reliability ECE == 0 exactly (PHYS-05).

    Returns ``(probs, gts)`` 1-D arrays of length ``15 * 270 = 4050``. Probs take
    15 distinct values at the centres of 15 equal-width bins, ``p_i=(i+0.5)/15``.
    For each value, EXACTLY ``round(p_i * 270)`` of the 270 group members have
    ``gt == 1`` (an exact integer count, since ``p_i*270=(2i+1)*9``), so within
    every bin the mean confidence equals the empirical positive fraction
    (conf == acc) and the expected-calibration-error is 0 to machine precision.
    Seeded with ``np.random.default_rng(42)`` only to shuffle row order (ECE is
    order-invariant) so the arrays are not trivially sorted.
    """
    rng = np.random.default_rng(SEED)
    n_bins = 15
    group_size = 270                                  # divisible by 30 -> exact counts
    p_values = (np.arange(n_bins) + 0.5) / n_bins     # bin centres
    probs_parts: List[np.ndarray] = []
    gts_parts: List[np.ndarray] = []
    for p in p_values:
        n_pos = int(round(float(p) * group_size))     # exact: (2i+1)*9
        g = np.zeros(group_size, dtype=np.uint8)
        g[:n_pos] = 1
        probs_parts.append(np.full(group_size, float(p), dtype=np.float64))
        gts_parts.append(g)
    probs = np.concatenate(probs_parts)
    gts = np.concatenate(gts_parts)
    perm = rng.permutation(probs.size)                # order-invariant shuffle
    return probs[perm], gts[perm]


@pytest.fixture
def tiny_torch_module() -> Tuple[object, int]:
    """Tiny torch Module with a KNOWN parameter count for efficiency probes (PHYS-06).

    Returns ``(module, expected_param_count)`` where ``module`` is a single
    ``nn.Conv2d(1, 2, 3)`` => ``1*2*3*3 + 2 = 20`` trainable parameters. ``torch``
    is imported LOCALLY so non-torch test paths never pay the import cost.
    """
    import torch.nn as nn                             # local import (PHYS-06 only)

    module = nn.Conv2d(1, 2, 3)                       # 1*2*3*3 weights + 2 biases
    expected = 1 * 2 * 3 * 3 + 2                       # == 20
    assert sum(p.numel() for p in module.parameters()) == expected
    return module, expected
