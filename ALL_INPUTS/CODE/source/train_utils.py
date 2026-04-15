from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

from source.callbacks import EpochTimeCallback


def train_model(
    *,
    model: keras.Model,
    train_dataset_img,
    val_dataset_img,
    run_dir: Path,
    epochs_total: int,
    initial_epoch: int = 0,
    model_path: Optional[str] = None,
    learning_rate: float = 1e-4,
    clipnorm: float = 1.0,
    early_stop_patience: int = 5,
    lr_patience: int = 2,
    lr_factor: float = 0.5,
    min_lr: float = 1e-7,
    csv_name: str = "training_log.csv",
    ckpt_subdir: str = "checkpoints",
) -> Tuple[keras.Model, Any, Dict[str, Any]]:
    """
    Train from scratch or resume from a checkpoint.

    User controls:
      - epochs_total
      - model_path (optional)
      - initial_epoch (0 for scratch; set to completed epoch for resume)

    Creates:
      run_dir/<ckpt_subdir>/ep{epoch:03d}-val{val_loss:.4f}.keras
      run_dir/<csv_name>

    Returns:
      model, history, info_dict
    """
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / ckpt_subdir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / csv_name

    # If resuming, load model (and append CSV)
    if model_path is not None:
        model = keras.models.load_model(model_path)
        csv_append = True
    else:
        csv_append = False if initial_epoch == 0 else True

    # Compile (explicit)
    opt = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm)
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(threshold=0.5),
            tf.keras.metrics.Precision(thresholds=0.5),
            tf.keras.metrics.Recall(thresholds=0.5),
        ],
    )

    # Callbacks
    csv_logger = CSVLogger(str(csv_path), append=csv_append)
    epoch_time_cb = EpochTimeCallback()

    ckpt_cb = ModelCheckpoint(
        filepath=str(ckpt_dir / "ep{epoch:03d}-val{val_loss:.4f}.keras"),
        save_weights_only=False,
        save_best_only=False,
        monitor="val_loss",
        mode="min",
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=early_stop_patience,
        restore_best_weights=True,
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=lr_factor,
        patience=lr_patience,
        min_lr=min_lr,
        verbose=1,
    )

    # Train
    start_time = time.time()
    history = model.fit(
        train_dataset_img,
        validation_data=val_dataset_img,
        epochs=epochs_total,
        initial_epoch=initial_epoch,
        callbacks=[early_stopping, reduce_lr, ckpt_cb, csv_logger, epoch_time_cb],
    )
    elapsed = time.time() - start_time

    info = {
        "run_dir": str(run_dir),
        "ckpt_dir": str(ckpt_dir),
        "csv_log": str(csv_path),
        "elapsed_sec": elapsed,
        "elapsed_min": elapsed / 60.0,
        "model_loaded_from": model_path,
        "epochs_total": epochs_total,
        "initial_epoch": initial_epoch,
    }
    return model, history, info
