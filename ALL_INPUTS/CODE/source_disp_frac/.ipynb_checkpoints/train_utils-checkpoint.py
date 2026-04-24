from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

from source_disp_frac.callbacks import EpochTimeCallback


# -----------------------------
# Loss + metrics for 3-channel shifted output
#
# y_true:
#   y[..., 0] = shifted/future fracture_mask
#   y[..., 1] = shifted/future ux_scaled
#   y[..., 2] = shifted/future uy_scaled
#
# y_pred:
#   pred[..., 0] = fracture logits
#   pred[..., 1] = ux_scaled prediction
#   pred[..., 2] = uy_scaled prediction
# -----------------------------
def make_combined_loss(
    ux_loss_weight: float = 1.0,
    uy_loss_weight: float = 1.0,
):
    def combined_loss(y_true, y_pred):
        frac_true = tf.cast(y_true[..., 0:1], tf.float32)
        ux_true = tf.cast(y_true[..., 1:2], tf.float32)
        uy_true = tf.cast(y_true[..., 2:3], tf.float32)

        frac_logits = tf.cast(y_pred[..., 0:1], tf.float32)
        ux_pred = tf.cast(y_pred[..., 1:2], tf.float32)
        uy_pred = tf.cast(y_pred[..., 2:3], tf.float32)

        frac_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=frac_true,
            logits=frac_logits,
        )
        frac_loss = tf.reduce_mean(frac_loss)

        ux_loss = tf.reduce_mean(tf.square(ux_true - ux_pred))
        uy_loss = tf.reduce_mean(tf.square(uy_true - uy_pred))

        return (
            frac_loss
            + ux_loss_weight * ux_loss
            + uy_loss_weight * uy_loss
        )

    combined_loss.__name__ = "combined_loss"
    return combined_loss


# -----------------------------
# Fracture metrics
# -----------------------------
def fracture_bce_metric(y_true, y_pred):
    frac_true = tf.cast(y_true[..., 0:1], tf.float32)
    frac_logits = tf.cast(y_pred[..., 0:1], tf.float32)

    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=frac_true,
        logits=frac_logits,
    )
    return tf.reduce_mean(loss)


def fracture_accuracy_metric(y_true, y_pred):
    frac_true = tf.cast(y_true[..., 0:1], tf.float32)
    frac_prob = tf.math.sigmoid(tf.cast(y_pred[..., 0:1], tf.float32))
    frac_pred = tf.cast(frac_prob >= 0.5, tf.float32)

    return tf.reduce_mean(tf.cast(tf.equal(frac_true, frac_pred), tf.float32))


def fracture_precision_metric(y_true, y_pred):
    frac_true = tf.cast(y_true[..., 0:1], tf.float32)
    frac_prob = tf.math.sigmoid(tf.cast(y_pred[..., 0:1], tf.float32))
    frac_pred = tf.cast(frac_prob >= 0.5, tf.float32)

    tp = tf.reduce_sum(frac_pred * frac_true)
    fp = tf.reduce_sum(frac_pred * (1.0 - frac_true))

    return tp / (tp + fp + 1e-8)


def fracture_recall_metric(y_true, y_pred):
    frac_true = tf.cast(y_true[..., 0:1], tf.float32)
    frac_prob = tf.math.sigmoid(tf.cast(y_pred[..., 0:1], tf.float32))
    frac_pred = tf.cast(frac_prob >= 0.5, tf.float32)

    tp = tf.reduce_sum(frac_pred * frac_true)
    fn = tf.reduce_sum((1.0 - frac_pred) * frac_true)

    return tp / (tp + fn + 1e-8)


# -----------------------------
# Displacement metrics
# -----------------------------
def ux_mse_metric(y_true, y_pred):
    ux_true = tf.cast(y_true[..., 1:2], tf.float32)
    ux_pred = tf.cast(y_pred[..., 1:2], tf.float32)
    return tf.reduce_mean(tf.square(ux_true - ux_pred))


def ux_mae_metric(y_true, y_pred):
    ux_true = tf.cast(y_true[..., 1:2], tf.float32)
    ux_pred = tf.cast(y_pred[..., 1:2], tf.float32)
    return tf.reduce_mean(tf.abs(ux_true - ux_pred))


def uy_mse_metric(y_true, y_pred):
    uy_true = tf.cast(y_true[..., 2:3], tf.float32)
    uy_pred = tf.cast(y_pred[..., 2:3], tf.float32)
    return tf.reduce_mean(tf.square(uy_true - uy_pred))


def uy_mae_metric(y_true, y_pred):
    uy_true = tf.cast(y_true[..., 2:3], tf.float32)
    uy_pred = tf.cast(y_pred[..., 2:3], tf.float32)
    return tf.reduce_mean(tf.abs(uy_true - uy_pred))


def disp_mse_metric(y_true, y_pred):
    ux_true = tf.cast(y_true[..., 1:2], tf.float32)
    uy_true = tf.cast(y_true[..., 2:3], tf.float32)

    ux_pred = tf.cast(y_pred[..., 1:2], tf.float32)
    uy_pred = tf.cast(y_pred[..., 2:3], tf.float32)

    return tf.reduce_mean(
        tf.square(ux_true - ux_pred)
        + tf.square(uy_true - uy_pred)
    )


def disp_mae_metric(y_true, y_pred):
    ux_true = tf.cast(y_true[..., 1:2], tf.float32)
    uy_true = tf.cast(y_true[..., 2:3], tf.float32)

    ux_pred = tf.cast(y_pred[..., 1:2], tf.float32)
    uy_pred = tf.cast(y_pred[..., 2:3], tf.float32)

    disp_err = tf.sqrt(
        tf.square(ux_true - ux_pred)
        + tf.square(uy_true - uy_pred)
        + 1e-12
    )

    return tf.reduce_mean(disp_err)


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
    ux_loss_weight: float = 1.0,
    uy_loss_weight: float = 1.0,
    early_stop_patience: int = 10,
    lr_patience: int = 2,
    lr_factor: float = 0.5,
    min_lr: float = 1e-7,
    csv_name: str = "training_log.csv",
    ckpt_subdir: str = "checkpoints",
) -> Tuple[keras.Model, Any, Dict[str, Any]]:
    """
    Train from scratch or resume from a checkpoint.

    Assumes:
      y_true[..., 0] = fracture_mask
      y_true[..., 1] = ux_scaled
      y_true[..., 2] = uy_scaled

      y_pred[..., 0] = fracture logits
      y_pred[..., 1] = ux_scaled prediction
      y_pred[..., 2] = uy_scaled prediction

    Loss:
      combined_loss =
          BCE_with_logits(fracture)
        + ux_loss_weight * MSE(ux_scaled)
        + uy_loss_weight * MSE(uy_scaled)

    Returns:
      model, history, info_dict
    """
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / ckpt_subdir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / csv_name

    # If resuming, load model architecture/weights only, then recompile explicitly.
    if model_path is not None:
        model = keras.models.load_model(model_path, compile=False)
        csv_append = True
    else:
        csv_append = False if initial_epoch == 0 else True

    opt = keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=clipnorm,
    )

    model.compile(
        optimizer=opt,
        loss=make_combined_loss(
            ux_loss_weight=ux_loss_weight,
            uy_loss_weight=uy_loss_weight,
        ),
        metrics=[
            fracture_bce_metric,
            fracture_accuracy_metric,
            fracture_precision_metric,
            fracture_recall_metric,
            ux_mse_metric,
            ux_mae_metric,
            uy_mse_metric,
            uy_mae_metric,
            disp_mse_metric,
            disp_mae_metric,
        ],
    )

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

    start_time = time.time()

    history = model.fit(
        train_dataset_img,
        validation_data=val_dataset_img,
        epochs=epochs_total,
        initial_epoch=initial_epoch,
        callbacks=[
            early_stopping,
            reduce_lr,
            ckpt_cb,
            csv_logger,
            epoch_time_cb,
        ],
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
        "ux_loss_weight": ux_loss_weight,
        "uy_loss_weight": uy_loss_weight,
    }

    return model, history, info