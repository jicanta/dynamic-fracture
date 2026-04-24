# source/predict_full_simulation_metrics_fullrollout.py

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix


def metrics_from_binary(y_true_1d: np.ndarray, y_pred_1d: np.ndarray):
    """
    Compute binary classification metrics + confusion counts from flattened 0/1 arrays.
    Returns: (precision, recall, f1, accuracy, tp, tn, fp, fn)
    """
    cm = confusion_matrix(y_true_1d, y_pred_1d, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    return prec, rec, f1, acc, tp, tn, fp, fn


def _get_model_expected_channels(model) -> int:
    """
    Infer the model's expected input channel count from model.input_shape.

    Supports:
      - single-input models with shape (None, T, H, W, C)
      - list-valued input_shape for multi-input models (uses the first input)

    Returns
    -------
    int
        Expected channel count C.
    """
    input_shape = getattr(model, "input_shape", None)

    if input_shape is None:
        raise ValueError("Could not read model.input_shape from the model.")

    if isinstance(input_shape, tuple):
        if len(input_shape) < 5:
            raise ValueError(f"Unexpected model.input_shape={input_shape}")
        c = input_shape[-1]
        if c is None:
            raise ValueError(f"Could not infer channel count from model.input_shape={input_shape}")
        return int(c)

    if isinstance(input_shape, list) and len(input_shape) > 0:
        first_shape = input_shape[0]
        if not isinstance(first_shape, tuple) or len(first_shape) < 5:
            raise ValueError(f"Unexpected first model input shape={first_shape}")
        c = first_shape[-1]
        if c is None:
            raise ValueError(f"Could not infer channel count from first model input shape={first_shape}")
        return int(c)

    raise ValueError(f"Unsupported model.input_shape format: {input_shape}")


def _adapt_channels_to_model(
    X: np.ndarray,
    model_expected_channels: int,
    fracture_channel_idx: int,
):
    """
    Adapt dataset channels to what the model expects.

    Rules
    -----
    - If dataset has exactly the same number of channels as the model expects:
        keep as-is
    - If dataset has MORE channels than the model expects:
        drop the highest-index channels until the counts match
        (assumes most recently added features are appended at the end)
    - If dataset has FEWER channels than the model expects:
        raise an error
    """
    if X.ndim != 5:
        raise ValueError(f"Expected X to have shape (B,T,H,W,C), got {X.shape}")

    cin = X.shape[-1]

    if cin == model_expected_channels:
        return X, fracture_channel_idx

    if cin < model_expected_channels:
        raise ValueError(
            f"Dataset has fewer channels than the model expects: "
            f"Cin={cin}, model_expected_channels={model_expected_channels}. "
            f"This code only auto-drops extra trailing channels; it does not invent missing ones."
        )

    keep_channels = list(range(model_expected_channels))
    dropped_channels = list(range(model_expected_channels, cin))

    X = X[..., keep_channels]

    if fracture_channel_idx in dropped_channels:
        raise ValueError(
            f"fracture_channel_idx={fracture_channel_idx} would be dropped when adapting "
            f"Cin={cin} to model_expected_channels={model_expected_channels}. "
            f"Check your fracture_channel_idx and channel ordering."
        )

    return X, fracture_channel_idx


def _prepare_x_window(
    X_tensor,
    model_expected_channels: int,
    fracture_channel_idx: int,
):
    """
    Convert tensor to numpy float32 and adapt channels to the model.
    """
    X_np = X_tensor.numpy().astype(np.float32)
    X_np, fracture_channel_idx_eff = _adapt_channels_to_model(
        X_np,
        model_expected_channels=model_expected_channels,
        fracture_channel_idx=fracture_channel_idx,
    )
    return X_np, fracture_channel_idx_eff


def _make_single_window_iterator(ds_img, max_batches: Optional[int] = None):
    """
    Convert a batched dataset of windows into a sample-by-sample iterator.

    Input ds_img is expected to yield:
        X: (B,T,H,W,C)
        y: (B,T,H,W,1)

    This function returns an iterator over:
        X: (1,T,H,W,C)
        y: (1,T,H,W,1)

    so rollout can proceed continuously for the full available sequence length
    without being cut off by a final smaller batch.
    """
    ds_use = ds_img.take(max_batches) if max_batches is not None else ds_img

    # Unbatch individual windows, then re-add a batch dimension of 1 in numpy space.
    for X_win, y_win in ds_use.unbatch():
        X_win = np.expand_dims(X_win.numpy().astype(np.float32), axis=0)  # (1,T,H,W,C)
        y_win = np.expand_dims(y_win.numpy().astype(np.float32), axis=0)  # (1,T,H,W,1)
        yield X_win, y_win


def predict_full_simulation_continuous_ar_fullrollout_with_metrics(
    ds_img,
    model,
    out_dir: str,
    *,
    threshold: float | None = 0.5,
    max_batches: Optional[int] = None,
    prefix: str = "mask",
    csv_path: Optional[str] = None,
    gt_threshold: float = 0.5,
    frames_per_ns: float | None = None,
    fracture_channel_idx: int = 0,
    max_ar_steps: Optional[int] = None,
    enforce_no_healing: bool = True,
):
    """
    Run a SINGLE continuous autoregressive rollout through the full simulation.

    Key fix compared with the old version
    -------------------------------------
    This function internally converts the dataset into B=1 chronological windows,
    so rollout does NOT stop early just because the final original dataset batch
    is smaller than previous batches.

    Behavior
    --------
    - Uses the first window as the initial true input state.
    - For each subsequent chronological window:
        * model predicts fracture from the current AR window
        * compare predicted last frame against GT last frame of the matching window
        * shift the input window causally (no wraparound)
        * replace only the fracture channel in the new last frame with the prediction
        * use the true future values of all non-fracture channels from the next window

    Assumptions
    -----------
    - ds_img is chronological and shuffle=False
    - ds_img corresponds to ONE simulation stream
    - model predicts fracture only
    - all non-fracture channels are known exogenous inputs

    Notes
    -----
    - This is the physically meaningful continuous AR mode because it uses B=1.
    - If the source dataset was originally batched, that batching is ignored for
      rollout continuity.
    """
    os.makedirs(out_dir, exist_ok=True)
    if csv_path is None:
        csv_path = os.path.join(out_dir, "per_frame_metrics.csv")

    pred_threshold = 0.5 if threshold is None else float(threshold)

    t0 = time.time()
    rows = []
    frame_counter = 0
    TP = TN = FP = FN = 0

    model_expected_channels = _get_model_expected_channels(model)
    print(f"Model expected channels: {model_expected_channels}")

    iterator = iter(_make_single_window_iterator(ds_img, max_batches=max_batches))

    # ---------------------------
    # Initialize from first window
    # ---------------------------
    try:
        X0_raw, y0_np = next(iterator)
    except StopIteration:
        raise ValueError("ds_img is empty; no windows available for continuous AR rollout.")

    # Adapt channels to model
    X_ar, fracture_channel_idx_eff = _adapt_channels_to_model(
        X0_raw,
        model_expected_channels=model_expected_channels,
        fracture_channel_idx=fracture_channel_idx,
    )

    if X_ar.ndim != 5:
        raise ValueError(f"Expected X_ar shape (B,T,H,W,C), got {X_ar.shape}")
    if y0_np.ndim != 5:
        raise ValueError(f"Expected y0 shape (B,T,H,W,1), got {y0_np.shape}")

    B, T, H, W, Cin = X_ar.shape
    if B != 1:
        raise ValueError(
            f"Internal rollout expects B=1 after unbatching, but got X_ar.shape={X_ar.shape}"
        )

    if fracture_channel_idx_eff >= Cin:
        raise ValueError(
            f"fracture_channel_idx={fracture_channel_idx_eff} out of range for Cin={Cin}"
        )

    print(f"Initial input shape before channel adaptation: {X0_raw.shape}")
    print(f"Using input shape for rollout after channel adaptation: {X_ar.shape}")
    print(f"fracture_channel_idx_eff={fracture_channel_idx_eff}")
    print(f"enforce_no_healing={enforce_no_healing}")
    print("Using full-length sample-by-sample AR rollout (internal batch size = 1).")

    # Step 0 GT = last frame of initial target window
    current_y_gt_last = y0_np[:, -1, :, :, 0]  # (1,H,W)

    # Start no-heal state from the last fracture frame already in the input window
    running_fracture_state = X_ar[:, -1, :, :, fracture_channel_idx_eff].copy()
    running_fracture_state = (running_fracture_state >= pred_threshold).astype(np.float32)

    pbar = tqdm(desc=f"Continuous AR full rollout {out_dir}", total=None)

    step_idx = 0

    while True:
        y_prob = model.predict(X_ar, verbose=0)

        if y_prob.ndim == 5:
            pred_last_raw = y_prob[:, -1, :, :, 0]  # (1,H,W)
        elif y_prob.ndim == 4:
            pred_last_raw = y_prob[:, :, :, 0]      # (1,H,W)
        else:
            raise ValueError(
                f"Unexpected model output shape {y_prob.shape}. "
                f"Expected 5D (B,T,H,W,1) or 4D (B,H,W,1)."
            )

        pred_last_bin = (pred_last_raw >= pred_threshold).astype(np.float32)

        if enforce_no_healing:
            pred_last_eval = np.maximum(running_fracture_state, pred_last_bin).astype(np.float32)
        else:
            pred_last_eval = pred_last_bin.astype(np.float32)

        # Save/evaluate current prediction against current GT
        pred_mask = pred_last_eval[0]
        gt = current_y_gt_last[0]

        pred_flip = np.flipud(pred_mask)
        gt_flip = np.flipud(gt)

        img_u8 = (np.clip(pred_flip, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img_u8).save(
            os.path.join(out_dir, f"{prefix}_{frame_counter:06d}.png")
        )

        pred_bin = (pred_flip >= 0.5).astype(np.uint8)
        gt_bin = (gt_flip >= gt_threshold).astype(np.uint8)

        yt = gt_bin.reshape(-1)
        yp = pred_bin.reshape(-1)

        prec, rec, f1, acc, tp, tn, fp, fn = metrics_from_binary(yt, yp)
        TP += tp
        TN += tn
        FP += fp
        FN += fn

        row = {
            "frame_id": frame_counter,
            "rollout_step": step_idx,
            "batch_source": "continuous_ar_fullrollout",
            "i_in_batch": 0,
            "H": H,
            "W": W,
            "n_pixels": int(yt.size),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "accuracy": float(acc),
        }
        if frames_per_ns is not None:
            row["time_ns"] = frame_counter / float(frames_per_ns)

        rows.append(row)
        frame_counter += 1
        pbar.update(1)

        # Update no-heal state after evaluation
        running_fracture_state = pred_last_eval.copy()

        # Optional hard stop
        if max_ar_steps is not None and step_idx >= max_ar_steps:
            break

        # Get next chronological true window
        try:
            X_next_true_raw, y_next_true = next(iterator)
        except StopIteration:
            break

        X_next_true, fracture_channel_idx_eff_next = _adapt_channels_to_model(
            X_next_true_raw,
            model_expected_channels=model_expected_channels,
            fracture_channel_idx=fracture_channel_idx,
        )

        if fracture_channel_idx_eff_next != fracture_channel_idx_eff:
            raise ValueError(
                "Fracture channel index changed unexpectedly after channel processing."
            )

        if X_next_true.shape != X_ar.shape:
            raise ValueError(
                f"Shape mismatch between current AR state {X_ar.shape} "
                f"and next true window {X_next_true.shape}"
            )

        # Feed back predicted/no-heal fracture mask into the last frame
        next_mask = pred_last_eval[..., None].astype(np.float32)  # (1,H,W,1)

        # Build the next AR state
        X_ar_new = np.empty_like(X_ar)
        X_ar_new[:, :-1, :, :, :] = X_ar[:, 1:, :, :, :]
        X_ar_new[:, -1, :, :, :] = X_next_true[:, -1, :, :, :]
        X_ar_new[:, -1, :, :, fracture_channel_idx_eff:fracture_channel_idx_eff + 1] = next_mask

        X_ar = X_ar_new
        current_y_gt_last = y_next_true[:, -1, :, :, 0]
        step_idx += 1

    pbar.close()

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    total_prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    total_rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    total_f1 = (2 * total_prec * total_rec / (total_prec + total_rec)) if (total_prec + total_rec) > 0 else 0.0
    total_acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

    t1 = time.time()
    print(f"\nSaved {frame_counter} frames to: {out_dir}")
    print(f"Saved per-frame metrics CSV to: {csv_path}")
    print("\nTOTAL METRICS")
    print(f"TP: {TP}  TN: {TN}  FP: {FP}  FN: {FN}")
    print(f"Precision: {total_prec:.6f}")
    print(f"Recall:    {total_rec:.6f}")
    print(f"F1:        {total_f1:.6f}")
    print(f"Accuracy:  {total_acc:.6f}")
    print(f"Elapsed time: {t1 - t0:.3f} s")

    return df