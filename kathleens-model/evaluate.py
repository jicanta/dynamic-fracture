#!/usr/bin/env python3
"""Headless evaluation entrypoint = notebook cells 35-45, made job-submittable.

Loads a checkpoint, runs a single continuous autoregressive rollout per test
case (her exact predict_full_simulation_continuous_ar_with_metrics), writes
per_frame_metrics.csv per case, and aggregates a run-level total_metrics.csv
(micro precision/recall/F1 per case) in the same shape as new_model's outputs.

Two checkpoint sources:
  --use-ref-ckpt        load HER reference checkpoint for this mode (default).
                        This is the deterministic reproduction path: her weights
                        through this pipeline must match her reference numbers.
  --model-path PATH     load a checkpoint you trained (e.g. OUTPUTS/<run>/checkpoints/...).

Usage (from kathleens-model/):
    python evaluate.py --mode 4 --run-name sed_ref_eval --use-ref-ckpt
    python evaluate.py --mode 1 --run-name base_repro_eval \
        --model-path OUTPUTS/base_repro/checkpoints/ep130-val0.0001.keras
    # restrict cases:
    python evaluate.py --mode 4 --use-ref-ckpt --cases test_MS206_V400 test_MS206_V200
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")

import csv as _csv
import json as _json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

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
from source import predict_full_simulation_metrics as psm
from source import run_config as rc
from source import validation_util as vu
from source.calibrate import calibrate  # noqa: E402


def _collect_train_val_probs(model, train_ds_img, *, val_fraction, gt_threshold,
                             fracture_channel_idx, enforce_no_healing=True):
    """Run the FROZEN model over the TRAINING data and return (val_probs, val_gt_bin)
    for the clean frame-boundary val tail (CMP-04 / D-07).

    The val frames are the temporal tail of the TRAINING path -- NOT the 16
    held-out test cases. The boundary is the SAME formula as Plan 01-04:
        f_cut = int(round(n_frames * (1.0 - 0.1)))     # val_fraction=0.1 (config.py:55)
    computed on the post-frame-0-drop frame count (drop_first_csv=True is applied
    upstream in build_datasets, so the frame indices here are already post-drop).

    Probabilities are collected TEACHER-FORCED (one prediction per dataset window
    from the true input frames), not via cross-batch AR feedback -- the training
    dataset has variable batch sizes, so injecting a previous batch's prediction
    into the next batch's window is ill-defined. For THRESHOLD calibration this is
    the right basis: the val FRAMES and the objective/grid match the new side; the
    prob-production protocol (teacher-forced vs AR) is the one documented
    difference (``enforce_no_healing`` is accepted for signature compatibility but
    unused in teacher-forced collection). Inference only -- frozen .keras untouched.

    NOTE (autonomous:false): executes on Gilbreth (TF GPU + frozen .keras); the
    chronological ordering / per-run alignment of the val tail is confirmed there.
    """
    import numpy as np

    model_expected_channels = psm._get_model_expected_channels(model)
    probs, gts = [], []

    for X, y in train_ds_img:
        X_ar, _ = psm._prepare_x_window(
            X, model_expected_channels=model_expected_channels,
            fracture_channel_idx=fracture_channel_idx)
        y_prob = model.predict(X_ar, verbose=0)
        if y_prob.ndim == 5:
            pred_last_raw = y_prob[:, -1, :, :, 0]      # (B,H,W)
        elif y_prob.ndim == 4:
            pred_last_raw = y_prob[:, :, :, 0]
        else:
            raise ValueError(f"Unexpected model output shape {y_prob.shape}")

        gt_last = y.numpy().astype(np.float32)[:, -1, :, :, 0]   # (B,H,W)
        for i in range(pred_last_raw.shape[0]):                  # per-batch B (variable)
            probs.append(pred_last_raw[i].astype(np.float32))
            gts.append((gt_last[i] >= gt_threshold).astype(np.uint8))

    # Clean frame-boundary val split (Plan 01-04 / D-07): tail at/after f_cut.
    n_frames = len(probs)
    f_cut = int(round(n_frames * (1.0 - val_fraction)))
    return probs[f_cut:], gts[f_cut:]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", type=int, required=True, choices=[1, 2, 3, 4])
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--data-root", default=str(config.DATA_ROOT))
    ap.add_argument("--out-root", default=str(config.OUT_ROOT))
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--use-ref-ckpt", action="store_true",
                     help="Load HER reference checkpoint for this mode (default).")
    src.add_argument("--model-path", default=None,
                     help="Load a checkpoint you trained instead.")
    ap.add_argument("--cases", nargs="*", default=None,
                    help="Subset of build_datasets test keys. Default: all.")
    ap.add_argument("--threshold", type=float, default=config.PRED_THRESHOLD)
    ap.add_argument("--max-ar-steps", type=int, default=None)
    # CMP-04: val-calibrate the decision threshold (default) instead of the
    # hardcoded config.PRED_THRESHOLD=0.5, using the SAME objective + grid and the
    # SAME clean frame-boundary val frames as the new pipeline. --no-calibrate
    # keeps the fixed --threshold (e.g. for ablations / quick checks).
    ap.add_argument("--calibrate", dest="calibrate", action="store_true", default=True,
                    help="Val-calibrate the threshold on the clean trainDS val frames (default).")
    ap.add_argument("--no-calibrate", dest="calibrate", action="store_false",
                    help="Skip calibration; use the fixed --threshold.")
    args = ap.parse_args()

    info = config.mode_info(args.mode)
    run_name = args.run_name or f"{info['name'].lower()}_eval"
    run_dir = Path(args.out_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Fail loud (D-02) if any registered GT case folder is missing under data_root.
    assert_manifest(Path(args.data_root))

    if args.model_path:
        ckpt = Path(args.model_path)
    else:
        ckpt = Path(info["ref_ckpt"])  # default = her reference checkpoint
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    cfg = rc.build_cfg(args.mode, include_extra_in_feature_cols=True)
    cfg["add_velocity"] = config.ADD_VELOCITY
    cfg["velocity_scale"] = config.VELOCITY_SCALE

    print(f"=== EVAL  mode={args.mode} ({info['name']})  run={run_name} ===")
    print(f"checkpoint={ckpt}")
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
    cases = args.cases or all_test_keys
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

    # 3) load model + compile (cells 38/39). predict adapts trailing channels if
    #    the data has more channels than the model expects.
    model = keras.models.load_model(str(ckpt))
    opt = keras.optimizers.Adam(config.LR, clipnorm=config.CLIPNORM)
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(threshold=0.5),
            tf.keras.metrics.Precision(thresholds=0.5),
            tf.keras.metrics.Recall(thresholds=0.5),
        ],
    )
    print(f"Model input shape: {model.input_shape}")

    # 3.5) CMP-04: val-calibrate the threshold on the SAME clean frame-boundary
    #      val frames as the new model (sourced from trainDS, NOT the test cases),
    #      replacing the hardcoded 0.5. Frozen weights are used inference-only.
    eval_threshold = args.threshold
    if args.calibrate:
        val_probs, val_gt = _collect_train_val_probs(
            model, datasets_img["train"],
            val_fraction=config.VAL_FRACTION if hasattr(config, "VAL_FRACTION") else 0.1,
            gt_threshold=config.GT_THRESHOLD,
            fracture_channel_idx=config.FRACTURE_CHANNEL_IDX,
            enforce_no_healing=config.ENFORCE_NO_HEALING,
        )
        if val_probs:
            calib = calibrate(val_probs, val_gt)          # shared objective + np.linspace(0.2,0.8,25)
            eval_threshold = calib["threshold"]
            (run_dir / "calibration.json").write_text(
                _json.dumps({"threshold": calib["threshold"], "val_f1": calib["val_f1"],
                             "n_val_frames": len(val_probs)}, indent=2)
            )
            print(f"[calibrate] threshold={eval_threshold:.3f} "
                  f"val_f1={calib['val_f1']:.4f} on {len(val_probs)} clean val frames")
        else:
            print("[calibrate] no val frames produced; falling back to "
                  f"--threshold {eval_threshold}")

    # 4) AR rollout + per-frame metrics per case (cell 45).
    #    out_dir = run_dir / <case>, where <case> is a CANONICAL registry key
    #    (build_datasets keys are wired to case_registry in Plan 02), so
    #    regenerated baselines are written under canonical keys (enables the 1:1
    #    CASE_MAP in compare_runs.py).
    totals = []
    for case in cases:
        out_dir = run_dir / case
        print(f"\n--- case {case} -> {out_dir} ---")
        df = psm.predict_full_simulation_continuous_ar_with_metrics(
            ds_img=datasets_img[case],
            model=model,
            out_dir=str(out_dir),
            threshold=eval_threshold,
            csv_path=str(out_dir / "per_frame_metrics.csv"),
            gt_threshold=config.GT_THRESHOLD,
            frames_per_ns=config.FRAMES_PER_NS,
            fracture_channel_idx=config.FRACTURE_CHANNEL_IDX,
            max_ar_steps=args.max_ar_steps,
            enforce_no_healing=config.ENFORCE_NO_HEALING,
            make_bce_plot=False,
        )
        tp = int(df["tp"].sum()); tn = int(df["tn"].sum())
        fp = int(df["fp"].sum()); fn = int(df["fn"].sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
        totals.append({
            "case": case, "n_frames_evaluated": int(len(df)),
            "precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        })

    # 5) run-level micro summary
    total_csv = run_dir / "total_metrics.csv"
    with open(total_csv, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(totals[0].keys()))
        w.writeheader()
        w.writerows(totals)
    print(f"\n=== EVAL DONE  wrote {total_csv} ({len(totals)} cases) ===")
    for t in totals:
        print(f"  {t['case']:<28} F1={t['f1']:.4f}  P={t['precision']:.4f}  R={t['recall']:.4f}")


if __name__ == "__main__":
    main()
