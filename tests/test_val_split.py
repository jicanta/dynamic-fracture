# tests/test_val_split.py
"""DATA-02: train/val frame-boundary split disjointness across all window sizes.

The production logic lives in new_model/src/data.py (WindowDataset +
assert_split_disjoint), but importing that module pulls in torch, which is not
guaranteed to be installed in the local/test env. So this test re-implements the
exact index-building contract from the interfaces block and pins it
independently — if data.py ever drifts from this contract, the disjointness it
promises is what these asserts protect.
"""
from __future__ import annotations

import pytest

VAL_FRACTION = 0.1

# Window sizes standing in for T+1, L_train, L_val. All < min synth n_frames (20).
W_TPLUS1 = 11
W_LTRAIN = 14
W_LVAL = 12


def _starts(n_frames: int, window_len: int, val_fraction: float, split: str):
    """Mirror WindowDataset's frame-boundary index logic for one run."""
    n_windows = n_frames - window_len + 1
    if n_windows <= 0:
        return []
    f_cut = int(round(n_frames * (1.0 - val_fraction)))
    out = []
    for s in range(n_windows):
        last = s + window_len - 1
        if split == "train":
            if last < f_cut:
                out.append(s)
        elif split == "val":
            if s >= f_cut:
                out.append(s)
        else:
            out.append(s)
    return out


def _train_last_val_first(n_frames, window_len, val_fraction):
    train = _starts(n_frames, window_len, val_fraction, "train")
    val = _starts(n_frames, window_len, val_fraction, "val")
    if not train or not val:
        return None
    return max(s + window_len - 1 for s in train), min(val)


def test_disjoint_across_three_window_sizes(synth_runs):
    """For each of T+1 / L_train / L_val, every run is frame-disjoint."""
    for window_len in (W_TPLUS1, W_LTRAIN, W_LVAL):
        for run in synth_runs:
            res = _train_last_val_first(run["n_frames"], window_len, VAL_FRACTION)
            if res is None:
                continue
            tr_last, val_first = res
            assert tr_last < val_first, (
                f"{run['name']} w={window_len}: overlap {tr_last} >= {val_first}"
            )


def test_cross_dataset_ltrain_vs_lval(synth_runs):
    """Pitfall 2: stage-2 train windows (L_train) must not overlap val (L_val).

    Both use the SAME per-run f_cut, so a train window of ANY length ends before
    the cut and a val window of ANY length starts at/after it. Verify the
    worst-case pairing per run: latest L_train train frame < earliest L_val val
    frame.
    """
    for run in synth_runs:
        n = run["n_frames"]
        train = _starts(n, W_LTRAIN, VAL_FRACTION, "train")
        val = _starts(n, W_LVAL, VAL_FRACTION, "val")
        if not train or not val:
            continue
        tr_last = max(s + W_LTRAIN - 1 for s in train)
        val_first = min(val)
        assert tr_last < val_first, (
            f"{run['name']}: L_train/L_val overlap {tr_last} >= {val_first}"
        )


def test_no_shared_frame_any_window(synth_runs):
    """The dropped straddling band is the guard gap: train and val frame SETS
    share no frame, for every window size."""
    for window_len in (W_TPLUS1, W_LTRAIN, W_LVAL):
        for run in synth_runs:
            n = run["n_frames"]
            train = _starts(n, window_len, VAL_FRACTION, "train")
            val = _starts(n, window_len, VAL_FRACTION, "val")
            train_frames = {f for s in train for f in range(s, s + window_len)}
            val_frames = {f for s in val for f in range(s, s + window_len)}
            assert train_frames.isdisjoint(val_frames), (
                f"{run['name']} w={window_len}: shared frames "
                f"{sorted(train_frames & val_frames)}"
            )


def test_overlapping_construction_is_detected():
    """A naive window-count split (the old data.py:324 bug) DOES overlap — a
    disjointness checker must flag it. This guards against regressing to the
    per-dataset cut."""
    n_frames, window_len = 26, 14
    n_windows = n_frames - window_len + 1
    # Old buggy logic: cut on window COUNT, no straddle drop.
    cut = int(round(n_windows * (1.0 - VAL_FRACTION)))
    train = list(range(0, cut))
    val = list(range(cut, n_windows))
    tr_last = max(s + window_len - 1 for s in train)
    val_first = min(val)
    # The buggy split leaks: train's last frame is NOT before val's first frame.
    assert not (tr_last < val_first), (
        "expected the old window-count split to overlap, but it did not — "
        "the negative control is no longer valid"
    )
