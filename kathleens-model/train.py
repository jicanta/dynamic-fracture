#!/usr/bin/env python3
"""Headless training entrypoint = notebook cells 16-33, made job-submittable.

Reproduces Kathleen's training exactly: builds the point-wise windowed datasets,
carves a validation split by batches, maps points->images, builds her ConvLSTM,
and fits it with her callbacks. All model/data logic is her source/ modules
verbatim; this script only wires them together the way the notebook did.

Usage (from kathleens-model/):
    python train.py --mode 1 --run-name base_repro --epochs 130
    # resume:
    python train.py --mode 1 --run-name base_repro --epochs 130 \
        --initial-epoch 70 --model-path OUTPUTS/base_repro/checkpoints/ep070-*.keras

MODE: 1=base 2=+pressure 3=+vonmises 4=+SED
Outputs land in OUTPUTS/<run-name>/ (checkpoints/, training_log.csv).
"""
from __future__ import annotations

import argparse
import os

# Must be set BEFORE importing tensorflow (notebook cell 3).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")

import tensorflow as tf

tf.config.optimizer.set_jit(False)
for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except Exception as e:  # noqa: BLE001
        print(f"Could not set memory growth for {_gpu}: {e}")

from pathlib import Path

import config
from source import build_datasets as bd
from source import mapping as ma
from source import run_config as rc
from source import train_utils as tu
from source import validation_util as vu


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", type=int, required=True, choices=[1, 2, 3, 4],
                    help="1=base 2=pressure 3=vonmises 4=SED")
    ap.add_argument("--run-name", default=None,
                    help="Output subfolder under OUTPUTS/. Default: <modename>_repro.")
    ap.add_argument("--data-root", default=str(config.DATA_ROOT))
    ap.add_argument("--out-root", default=str(config.OUT_ROOT))
    ap.add_argument("--epochs", type=int, default=config.EPOCHS_TOTAL)
    ap.add_argument("--initial-epoch", type=int, default=0)
    ap.add_argument("--model-path", default=None,
                    help="Checkpoint to resume from (sets initial-epoch accordingly).")
    ap.add_argument("--seed", type=int, default=config.SEED,
                    help="Global seed for model init + shuffle ordering (D-05).")
    args = ap.parse_args()

    info = config.mode_info(args.mode)
    run_name = args.run_name or f"{info['name'].lower()}_repro"
    run_dir = Path(args.out_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = rc.build_cfg(args.mode, include_extra_in_feature_cols=True)
    cfg["add_velocity"] = config.ADD_VELOCITY
    cfg["velocity_scale"] = config.VELOCITY_SCALE

    print(f"=== TRAIN  mode={args.mode} ({info['name']})  run={run_name} ===")
    print(f"extra={cfg['extra']}  feature_cols={cfg['feature_cols']}")
    print(f"data_root={args.data_root}")
    print(f"run_dir={run_dir}")
    print(f"epochs={args.epochs}  initial_epoch={args.initial_epoch}")
    print(f"seed={args.seed}")

    _preflight(Path(args.data_root))

    # 1) raw point-wise windowed datasets (cell 18)
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

    # 1b) validation split from TRAIN (cell 18)
    train_raw, val_raw, total_batches, val_batches = vu.split_train_val_by_batches(
        datasets_raw["train"], validation_fraction=config.VAL_FRAC, verbose=True,
    )

    # Only train + val need mapping for fitting; skip mapping the test sets here
    # to save graph-build time (they are evaluated separately in evaluate.py).
    to_map = {"train": train_raw, "val": val_raw}

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
    print(f"Mapped grid: H={grid.H}, W={grid.W} | Model Cin={Cin}")

    # 3) model (cell 21)
    model_cfg = ma.ModelConfig(
        sequence_length=config.SEQ_LEN,
        dropout=config.DROPOUT,
        lr=config.LR,
        clipnorm=config.CLIPNORM,
        Cin=Cin,
        seed=args.seed,
    )
    model = ma.build_convlstm_model(model_cfg, grid)
    model.summary()

    # 4) train (cells 28/33)
    model, history, train_info = tu.train_model(
        model=model,
        train_dataset_img=datasets_img["train"],
        val_dataset_img=datasets_img["val"],
        run_dir=run_dir,
        epochs_total=args.epochs,
        initial_epoch=args.initial_epoch,
        model_path=args.model_path,
    )
    print(train_info)
    print(f"=== TRAIN DONE  checkpoints in {run_dir/'checkpoints'} ===")


def _preflight(data_root: Path) -> None:
    """Fail fast with a clear message if the dataset layout is wrong."""
    if not data_root.exists():
        raise SystemExit(
            f"DATA_ROOT does not exist: {data_root}\n"
            f"Set KM_DATA_ROOT (or --data-root) to the folder that directly "
            f"contains trainDS/ and the F_* test-case folders of CSVs."
        )
    train = data_root / "trainDS"
    if not train.exists():
        present = sorted(p.name for p in data_root.iterdir() if p.is_dir())[:10]
        raise SystemExit(
            f"No trainDS/ under {data_root}. Found dirs: {present}\n"
            f"Point --data-root at the directory containing trainDS/."
        )
    print(f"[preflight] OK: {train} exists.")


if __name__ == "__main__":
    main()
