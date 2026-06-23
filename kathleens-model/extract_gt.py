#!/usr/bin/env python3
"""Model-free GT extractor = the y-target half of evaluate.py, no checkpoint loaded.

Closes the D-09/D-10 correctness gap: every demo image (D-07/D-08) must trace to
what was actually measured, so the slide GT must be PROVABLY the exact GT her
metrics were scored against -- never the new-model's gt.npz.

This is a stripped-down fork of evaluate.py: SAME env preamble, SAME repo-root
shim, SAME `source/` imports, SAME dataset-rebuild block (the D-09 fidelity
guarantee), but it loads no model. It reads only `y[:, -1, :, :, 0]`, flips
(flipud, matching her flipped prediction PNGs), thresholds at config.GT_THRESHOLD,
reconstructs the exact gt_bin frames the continuous-AR rollout scored, and
ASSERTS the per-frame GT positive-pixel count equals `tp+fn` from the eval
`per_frame_metrics.csv` (D-10 parity). It then writes `gt.npz` (key 'gt',
already-flipped, so make_repro_slides.py consumes it WITHOUT --gt-flip).

`source/` stays SHA256-clean: reuse by import only, never edited.

NOTE: rebuilds tf datasets -> requires the cluster TF env to RUN. This file is
static-checkable locally; the live parity assertion runs on the cluster (Plan 06).

Usage (from kathleens-model/, on the Gilbreth TF env):
    python extract_gt.py --mode 4 --run-dir OUTPUTS/sed_repro
    python extract_gt.py --mode 1 --run-dir OUTPUTS/base_repro \
        --cases test_MS206_V1000 test_inclusions_1_2
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

tf.config.optimizer.set_jit(False)
for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except Exception as e:  # noqa: BLE001
        print(f"Could not set memory growth for {_gpu}: {e}")

import sys
from pathlib import Path as _Path

# Repo-root shim so the shared, framework-free modules (Plan 02/03) import here.
_REPO_ROOT = _Path(__file__).resolve().parents[1]   # kathleens-model -> dynamic-fracture
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from case_registry import assert_manifest  # noqa: E402

import config
from source import build_datasets as bd
from source import mapping as ma
from source import run_config as rc


def _discover_cases(run_dir: Path) -> list[str]:
    """Case keys = subdirs of run_dir that carry a per_frame_metrics.csv (eval output)."""
    return sorted(
        p.name for p in run_dir.iterdir()
        if p.is_dir() and (p / "per_frame_metrics.csv").exists()
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", type=int, required=True, choices=[1, 2, 3, 4])
    ap.add_argument("--run-dir", required=True,
                    help="Eval output dir holding <case>/per_frame_metrics.csv.")
    ap.add_argument("--data-root", default=str(config.DATA_ROOT))
    ap.add_argument("--cases", nargs="*", default=None,
                    help="Subset of build_datasets test keys. Default: cases under --run-dir.")
    args = ap.parse_args()

    info = config.mode_info(args.mode)

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"--run-dir not found: {run_dir}")

    # Fail loud (D-02) if any registered GT case folder is missing under data_root.
    assert_manifest(Path(args.data_root))

    # --- dataset rebuild: copied EXACTLY from evaluate.py:153-201 (D-09 fidelity) ---
    cfg = rc.build_cfg(args.mode, include_extra_in_feature_cols=True)
    cfg["add_velocity"] = config.ADD_VELOCITY
    cfg["velocity_scale"] = config.VELOCITY_SCALE

    print(f"=== EXTRACT-GT  mode={args.mode} ({info['name']})  run_dir={run_dir} ===")
    print(f"extra={cfg['extra']}  data_root={args.data_root}")

    # 1) raw datasets (train needed only to infer the shared grid, exactly as the
    #    notebook does: grid_source_key='train' for all mapped sets).
    datasets_raw = bd.make_datasets(
        data_root=args.data_root,
        feature_cols=cfg["feature_cols"],
        batch_size=config.BATCH_SIZE,
        sequence_length=config.SEQ_LEN,
        drop_first_csv=config.DROP_FIRST_CSV,
        print_run_stats=config.PRINT_RUN_STATS,
        stats_max_files_train=config.STATS_MAX_FILES_TRAIN,
        stats_max_files_test=None,
        add_velocity=cfg["add_velocity"],
        velocity_scale=cfg["velocity_scale"],
    )

    all_test_keys = [k for k in datasets_raw if k != "train"]
    cases = args.cases or _discover_cases(run_dir)
    if not cases:
        raise SystemExit(f"No cases to extract: pass --cases or populate {run_dir}/<case>/")
    unknown = [c for c in cases if c not in datasets_raw]
    if unknown:
        raise SystemExit(f"Unknown cases {unknown}. Available: {sorted(all_test_keys)}")

    # 2) map train (grid source) + requested test cases with the shared train grid
    to_map = {"train": datasets_raw["train"]}
    for c in cases:
        to_map[c] = datasets_raw[c]

    mapping_cfg = ma.MappingConfig(
        sequence_length=config.SEQ_LEN,
        uxuy_scale=config.UXUY_SCALE,
        binarize_masks=config.BINARIZE_MASKS,
        mask_thresh=config.MASK_THRESH,
        binarize_after_avg=config.BINARIZE_AFTER_AVG,
    )
    datasets_img, grid, Cin = ma.map_existing_datasets(
        datasets_raw=to_map,
        mapping_cfg=mapping_cfg,
        feature_cols=cfg["feature_cols"],
        extra_var=cfg["extra"],
        grid_source_key="train",
        add_velocity=cfg["add_velocity"],
    )
    print(f"Mapped grid: H={grid.H}, W={grid.W} | data Cin={Cin}")

    # 3) per case: reconstruct the GT frames the continuous-AR rollout scored.
    #    Full-size batches only, dataset order, flipud orientation -- the undersized
    #    final batch BREAKS before being emitted (mirrors psm rollout :453-460), so
    #    the emitted frame count matches per_frame_metrics.csv exactly.
    for case in cases:
        it = iter(datasets_img[case])
        X0, y0 = next(it)
        cur = y0.numpy().astype(np.float32)[:, -1, :, :, 0]   # (B,H,W)
        B = cur.shape[0]
        gt_frames: list[np.ndarray] = []
        while True:
            for i in range(B):
                gt_frames.append((np.flipud(cur[i]) >= config.GT_THRESHOLD).astype(np.uint8))
            try:
                Xn, yn = next(it)
            except StopIteration:
                break
            yn_np = yn.numpy().astype(np.float32)
            if yn_np.shape[0] != B:
                break   # undersized final batch -> rollout breaks BEFORE emitting it
            cur = yn_np[:, -1, :, :, 0]
        gt = np.stack(gt_frames, axis=0)     # (T,H,W) uint8

        # --- D-10 parity assertion: extractor GT must equal the eval-scored GT ---
        out_dir = run_dir / case
        df = (
            pd.read_csv(out_dir / "per_frame_metrics.csv")
            .sort_values("frame_id")
            .reset_index(drop=True)
        )
        assert len(df) == gt.shape[0], (
            f"GT frame count {gt.shape[0]} != eval frames {len(df)} for case {case}"
        )
        gt_pos = gt.reshape(gt.shape[0], -1).sum(axis=1)
        csv_pos = (df["tp"].to_numpy() + df["fn"].to_numpy())
        assert np.array_equal(gt_pos, csv_pos), (
            f"GT parity FAILED for case {case}: extractor GT != eval-scored GT (tp+fn)"
        )

        np.savez_compressed(out_dir / "gt.npz", gt=gt)
        print(f"--- case {case} -> {out_dir} (T={gt.shape[0]}) parity OK ---")

    print(f"\n=== EXTRACT-GT DONE  {len(cases)} cases, all parity OK ===")


if __name__ == "__main__":
    main()
