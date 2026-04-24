from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from tqdm import tqdm


def predict_full_simulation_last_frames(
    ds_img,
    model,
    out_dir,
    *,
    threshold: Optional[float] = 0.5,  # None => save probability maps
    max_batches: Optional[int] = None, # None => all
    prefix: str = "mask",
    flipud: bool = True,               # keep your ParaView orientation fix
):
    """
    Saves exactly ONE PNG per dataset window: the last predicted frame in y_pred.

    Args:
      ds_img: tf.data.Dataset yielding (X_img, y_img)
      model:  tf.keras model
      out_dir: output directory
      threshold: float for binarization; None saves probabilities
      max_batches: limit dataset batches
      prefix: output file prefix
      flipud: whether to flip vertically for ParaView orientation
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_counter = 0
    iterator = ds_img.take(max_batches) if max_batches is not None else ds_img

    for batch_idx, (X_img, y_img) in enumerate(tqdm(iterator, desc=f"Predicting {out_dir}")):
        # y_pred: (B, T, H, W, 1)
        y_pred = model.predict(X_img, verbose=0)

        # last frame from the sequence
        last_frames = y_pred[:, -1, :, :, 0]  # (B, H, W)

        B, H, W = last_frames.shape
        for i in range(B):
            arr = last_frames[i]  # (H, W) in [0,1]

            if flipud:
                arr = np.flipud(arr)

            if threshold is not None:
                arr = (arr > threshold).astype(np.float32)

            img_u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            img = Image.fromarray(img_u8)

            fname = out_dir / f"{prefix}_{frame_counter:06d}.png"
            img.save(fname)
            frame_counter += 1

    print(f"Saved {frame_counter} frames to: {out_dir}")
    return out_dir
