from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Sequence

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# -----------------------------
# Config dataclasses
# -----------------------------
@dataclass
class MappingConfig:
    sequence_length: int = 10
    uxuy_scale: float = 1e4
    binarize_masks: bool = True
    mask_thresh: float = 0.5
    binarize_after_avg: bool = False


@dataclass
class ModelConfig:
    sequence_length: int = 10
    dropout: float = 0.1
    lr: float = 1e-4
    clipnorm: float = 1.0
    Cin: int = 8
    seed: int = 42


@dataclass
class GridSpec:
    H: int
    W: int
    x_vals: np.ndarray
    y_vals: np.ndarray
    x_grid: np.ndarray
    y_grid: np.ndarray


# -----------------------------
# Grid inference from a dataset batch
# -----------------------------
def _infer_grid_from_dataset_with_feature_cols(
    ds_raw: tf.data.Dataset,
    feature_cols: Sequence[str],
) -> GridSpec:
    """
    Infers x/y unique sorted values from ONE batch.
    Expects Xb channels include x and y, located by feature_cols.

    Xb: (B,T,N,C)
    """
    name_to_idx = {n: i for i, n in enumerate(feature_cols)}

    if "x" not in name_to_idx or "y" not in name_to_idx:
        raise ValueError(f"feature_cols must include 'x' and 'y'. Got: {feature_cols}")

    ix = name_to_idx["x"]
    iy = name_to_idx["y"]

    Xb = None
    for Xb0, _ in ds_raw.take(1):
        Xb = Xb0
        break

    if Xb is None:
        raise ValueError("Could not take(1) from ds_raw to infer grid.")

    x = tf.cast(Xb[0, 0, :, ix], tf.float32).numpy()
    y = tf.cast(Xb[0, 0, :, iy], tf.float32).numpy()

    x_vals = np.sort(np.unique(x)).astype(np.float32)
    y_vals = np.sort(np.unique(y)).astype(np.float32)

    W = int(len(x_vals))
    H = int(len(y_vals))

    if W <= 1 or H <= 1:
        raise ValueError(f"Inferred degenerate grid: H={H}, W={W}. Check x/y values in your CSVs.")

    x_min, x_max = float(x_vals[0]), float(x_vals[-1])
    y_min, y_max = float(y_vals[0]), float(y_vals[-1])

    x_norm = (x_vals - x_min) / (x_max - x_min + 1e-12)
    y_norm = (y_vals - y_min) / (y_max - y_min + 1e-12)

    x_grid = np.tile(x_norm[None, :], (H, 1)).astype(np.float32)[:, :, None]
    y_grid = np.tile(y_norm[:, None], (1, W)).astype(np.float32)[:, :, None]

    return GridSpec(
        H=H,
        W=W,
        x_vals=x_vals,
        y_vals=y_vals,
        x_grid=x_grid,
        y_grid=y_grid,
    )


# -----------------------------
# Mapper builder: points -> images
# -----------------------------
def build_points_to_images_mapper(
    *,
    grid: GridSpec,
    mapping_cfg: MappingConfig,
    feature_cols: Sequence[str],
    extra_var: Optional[str] = None,
    add_velocity: bool = False,
):
    """
    Builds a mapper (Xb,yb)->(X_img,y_img).

    IMPORTANT:
      Xb columns are in the order given by feature_cols, with optional velocity appended after.
      yb must contain shifted targets from import_datasets.py.

    For fracture + displacement prediction:

      Xb:
        frames [t ... t+T-1]

      yb:
        frames [t+shift ... t+shift+T-1]
        yb[..., 0] = fracture_mask
        yb[..., 1] = ux
        yb[..., 2] = uy

    Channel order (X_img):
      [0] fracture_mask
      [1] ux_scaled
      [2] uy_scaled
      [3] Gc
      [4] extra_var, optional, e.g. SED for MODE 6
      velocity, optional
      [last-2] x_norm
      [last-1] y_norm

    Channel order (y_img):
      [0] shifted fracture_mask
      [1] shifted ux_scaled
      [2] shifted uy_scaled
    """
    H, W = grid.H, grid.W

    name_to_idx = {n: i for i, n in enumerate(feature_cols)}

    required = ["fracture_mask", "x", "y", "ux", "uy", "Gc"]
    missing = [k for k in required if k not in name_to_idx]
    if missing:
        raise ValueError(f"feature_cols missing required columns: {missing}. Got: {feature_cols}")

    if extra_var is not None and extra_var not in name_to_idx:
        raise ValueError(f"extra_var='{extra_var}' not in feature_cols. Got feature_cols={feature_cols}")

    imask = name_to_idx["fracture_mask"]
    ix = name_to_idx["x"]
    iy = name_to_idx["y"]
    iux = name_to_idx["ux"]
    iuy = name_to_idx["uy"]
    igc = name_to_idx["Gc"]
    iextra = name_to_idx[extra_var] if extra_var is not None else None

    # Velocity is appended AFTER feature_cols in raw Xb, if present.
    ivel = len(feature_cols)

    # Scatter channels before x_norm/y_norm:
    # fracture, ux, uy, Gc, optional extra, optional velocity
    C_scatter = 4 + (1 if extra_var is not None else 0) + (1 if add_velocity else 0)
    C_img = C_scatter + 2

    # Target channels: fracture, ux, uy
    C_target = 3

    x_vals_tf = tf.constant(grid.x_vals, dtype=tf.float32)
    y_vals_tf = tf.constant(grid.y_vals, dtype=tf.float32)
    x_grid_tf = tf.constant(grid.x_grid, dtype=tf.float32)
    y_grid_tf = tf.constant(grid.y_grid, dtype=tf.float32)

    seq_len = int(mapping_cfg.sequence_length)
    uxuy_scale = float(mapping_cfg.uxuy_scale)
    binarize_masks = bool(mapping_cfg.binarize_masks)
    mask_thresh = float(mapping_cfg.mask_thresh)
    binarize_after_avg = bool(mapping_cfg.binarize_after_avg)

    @tf.function
    def mapper(Xb, yb):
        """
        Xb: (B,T,N,Craw)
        yb: (B,T,N,3)
            yb[..., 0] = shifted fracture_mask
            yb[..., 1] = shifted ux
            yb[..., 2] = shifted uy
        """
        B = tf.shape(Xb)[0]
        T = tf.shape(Xb)[1]
        N = tf.shape(Xb)[2]
        Craw = tf.shape(Xb)[-1]

        # --- coords -> indices ---
        x_f = tf.cast(Xb[..., ix], tf.float32)
        y_f = tf.cast(Xb[..., iy], tf.float32)

        x_sorted = tf.broadcast_to(x_vals_tf[None, None, :], [B, T, W])
        y_sorted = tf.broadcast_to(y_vals_tf[None, None, :], [B, T, H])

        x_idx = tf.searchsorted(x_sorted, x_f, side="left")
        y_idx = tf.searchsorted(y_sorted, y_f, side="left")

        x_idx = tf.clip_by_value(x_idx, 0, W - 1)
        y_idx = tf.clip_by_value(y_idx, 0, H - 1)

        # --- input fracture channel cleaning ---
        mask = tf.cast(Xb[..., imask:imask + 1], tf.float32)
        mask = tf.clip_by_value(mask, 0.0, 1.0)

        if binarize_masks:
            mask = tf.cast(mask > mask_thresh, tf.float32)

        # --- input continuous channels ---
        ux = tf.cast(Xb[..., iux:iux + 1], tf.float32) / uxuy_scale
        uy = tf.cast(Xb[..., iuy:iuy + 1], tf.float32) / uxuy_scale
        gc = tf.cast(Xb[..., igc:igc + 1], tf.float32)

        # --- input feature channels ---
        parts = [mask, ux, uy, gc]

        # Optional extra input channel, e.g. SED for MODE 6.
        # This is an input feature, not a predicted target in this model.
        if iextra is not None:
            extra = tf.cast(Xb[..., iextra:iextra + 1], tf.float32)
            parts.append(extra)

        # Optional velocity channel.
        if add_velocity:
            vel = tf.cond(
                Craw > ivel,
                lambda: tf.cast(Xb[..., ivel:ivel + 1], tf.float32),
                lambda: tf.zeros_like(mask),
            )
            parts.append(vel)

        feats = tf.concat(parts, axis=-1)  # (B,T,N,C_scatter)

        # --- shifted target channels from yb ---
        # These come from out_paths in import_datasets.py.
        frac_target = tf.cast(yb[..., 0:1], tf.float32)
        ux_target = tf.cast(yb[..., 1:2], tf.float32) / uxuy_scale
        uy_target = tf.cast(yb[..., 2:3], tf.float32) / uxuy_scale

        frac_target = tf.clip_by_value(frac_target, 0.0, 1.0)

        if binarize_masks:
            frac_target = tf.cast(frac_target > mask_thresh, tf.float32)

        y_feats = tf.concat(
            [frac_target, ux_target, uy_target],
            axis=-1,
        )  # (B,T,N,3)

        # --- scatter indices: [b, t, y, x] ---
        b_ids = tf.repeat(tf.range(B), T * N)
        t_ids = tf.tile(tf.repeat(tf.range(T), N), [B])

        idx = tf.stack(
            [
                b_ids,
                t_ids,
                tf.reshape(y_idx, [-1]),
                tf.reshape(x_idx, [-1]),
            ],
            axis=1,
        )

        X_sum = tf.scatter_nd(
            idx,
            tf.reshape(feats, [-1, C_scatter]),
            [B, T, H, W, C_scatter],
        )

        y_sum = tf.scatter_nd(
            idx,
            tf.reshape(y_feats, [-1, C_target]),
            [B, T, H, W, C_target],
        )

        # counts for averaging
        ones = tf.ones([B, T, N, 1], dtype=tf.float32)
        count = tf.scatter_nd(
            idx,
            tf.reshape(ones, [-1, 1]),
            [B, T, H, W, 1],
        )
        count_safe = tf.maximum(count, 1.0)

        X_img_sc = X_sum / count_safe
        y_img = y_sum / count_safe

        # --- clean target channels after averaging ---
        frac_y = tf.clip_by_value(y_img[..., 0:1], 0.0, 1.0)

        if binarize_after_avg:
            frac_y = tf.cast(frac_y > 0.5, tf.float32)

        ux_y = y_img[..., 1:2]
        uy_y = y_img[..., 2:3]

        y_img = tf.concat([frac_y, ux_y, uy_y], axis=-1)

        # --- keep input fracture channel sane after averaging ---
        mask_img = tf.clip_by_value(X_img_sc[..., 0:1], 0.0, 1.0)

        if binarize_masks:
            mask_img = tf.cast(mask_img > 0.5, tf.float32)

        X_img_sc = tf.concat([mask_img, X_img_sc[..., 1:]], axis=-1)

        # --- add coordinate channels ---
        x_img = tf.broadcast_to(x_grid_tf[None, None, ...], [B, T, H, W, 1])
        y_imgc = tf.broadcast_to(y_grid_tf[None, None, ...], [B, T, H, W, 1])

        X_img = tf.concat([X_img_sc, x_img, y_imgc], axis=-1)

        # finite safety
        X_img = tf.where(tf.math.is_finite(X_img), X_img, tf.zeros_like(X_img))
        y_img = tf.where(tf.math.is_finite(y_img), y_img, tf.zeros_like(y_img))

        # static shapes for graph stability
        X_img.set_shape([None, seq_len, H, W, C_img])
        y_img.set_shape([None, seq_len, H, W, C_target])

        return X_img, y_img

    return mapper, C_img


# -----------------------------
# Public mapping entrypoint
# -----------------------------
def map_existing_datasets(
    *,
    datasets_raw: Dict[str, tf.data.Dataset],
    mapping_cfg: MappingConfig,
    feature_cols: Sequence[str],
    extra_var: Optional[str] = None,
    grid_source_key: Optional[str] = None,
    add_velocity: bool = True,
) -> Tuple[Dict[str, tf.data.Dataset], GridSpec, int]:
    """
    Maps ONLY the datasets you pass in.

    Returns:
      datasets_img: dict name -> mapped dataset (X_img, y_img)
      grid: GridSpec
      Cin: number of image channels fed to the model
    """
    if not datasets_raw:
        raise ValueError("datasets_raw is empty. Pass in the datasets you created in workflow.")

    if grid_source_key is None:
        grid_source_key = next(iter(datasets_raw.keys()))

    if grid_source_key not in datasets_raw:
        raise ValueError(
            f"grid_source_key='{grid_source_key}' not found. Keys: {list(datasets_raw.keys())}"
        )

    grid = _infer_grid_from_dataset_with_feature_cols(
        datasets_raw[grid_source_key],
        feature_cols,
    )

    mapper, Cin = build_points_to_images_mapper(
        grid=grid,
        mapping_cfg=mapping_cfg,
        feature_cols=feature_cols,
        extra_var=extra_var,
        add_velocity=add_velocity,
    )

    mapped: Dict[str, tf.data.Dataset] = {}

    for name, ds in datasets_raw.items():
        mapped[name] = ds.map(
            mapper,
            num_parallel_calls=tf.data.AUTOTUNE,
        ).prefetch(tf.data.AUTOTUNE)

    return mapped, grid, Cin


# -----------------------------
# Model builder
# -----------------------------
def build_convlstm_model(cfg: ModelConfig, grid: GridSpec) -> keras.Model:
    tf.random.set_seed(cfg.seed)

    he_init = keras.initializers.HeNormal()

    H, W = grid.H, grid.W
    Cin = int(cfg.Cin)

    seq_in = layers.Input(shape=(cfg.sequence_length, H, W, Cin), name="seq_in")

    def convlstm_block(x, filters=128, name=None):
        x = layers.ConvLSTM2D(
            filters,
            (3, 3),
            padding="same",
            return_sequences=True,
            activation="relu",
            kernel_initializer=he_init,
            dtype="float32",
            unroll=False,
            name=name,
        )(x)

        x = layers.BatchNormalization(dtype="float32")(x)
        return x

    x = convlstm_block(seq_in, 128, name="clstm1")
    x = convlstm_block(x, 128, name="clstm2")

    # 321x161 -> 322x162
    x = layers.TimeDistributed(
        layers.ZeroPadding2D(padding=((0, 1), (0, 1)))
    )(x)

    x = layers.TimeDistributed(
        layers.AveragePooling2D((2, 2))
    )(x)

    x = convlstm_block(x, 128, name="clstm3")
    x = convlstm_block(x, 128, name="clstm4")
    x = convlstm_block(x, 128, name="clstm5")
    x = convlstm_block(x, 128, name="clstm6")
    x = convlstm_block(x, 128, name="clstm7")

    x = layers.Dropout(cfg.dropout, seed=cfg.seed)(x)

    x = layers.TimeDistributed(
        layers.UpSampling2D((2, 2))
    )(x)

    x = layers.TimeDistributed(
        layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same",
            kernel_initializer=he_init,
        )
    )(x)

    # Output channels:
    # [0] fracture logit
    # [1] ux_scaled prediction
    # [2] uy_scaled prediction
    out = layers.TimeDistributed(
        layers.Conv2D(
            3,
            (3, 3),
            activation=None,
            padding="same",
            dtype="float32",
        )
    )(x)

    # 322x162 -> 321x161
    out = layers.TimeDistributed(
        layers.Cropping2D(cropping=((0, 1), (0, 1)))
    )(out)

    model = keras.Model(seq_in, out, name="microML_convlstm_frac_disp")

    return model