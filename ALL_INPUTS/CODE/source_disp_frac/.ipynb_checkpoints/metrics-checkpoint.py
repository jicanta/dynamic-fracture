# metrics.py
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix


# ------------------------
# Core metric helper
# ------------------------

def metrics_from_binary(y_true_1d: np.ndarray, y_pred_1d: np.ndarray):
    """
    Compute precision, recall, F1, accuracy and confusion counts for
    binary masks flattened to 1D (values 0 or 1).
    """
    cm = confusion_matrix(y_true_1d, y_pred_1d, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc  = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return prec, rec, f1, acc, tp, tn, fp, fn


# ------------------------
# Image loading helpers
# ------------------------

def crop_to_first_black(arr: np.ndarray, black_threshold: int = 10) -> np.ndarray:
    """
    Crop a grayscale image (2D array) to the bounding box where pixels
    are 'black' (value < black_threshold) on all sides.
    This matches the debugging process we used.
    """
    mask = arr < black_threshold
    if not mask.any():
        # No black pixels found; return original
        return arr

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0, r1 = rows[0], rows[-1]
    c0, c1 = cols[0], cols[-1]
    return arr[r0 : r1 + 1, c0 : c1 + 1]


def load_pred_mask(path: str,
                   *,
                   threshold: int = 127,
                   fracture_is_white: bool = True,
                   flipud: bool = False) -> np.ndarray:
    """
    Load a prediction PNG and convert to a binary mask.
    Assumes fracture pixels are white on black by default.
    """
    img = Image.open(path).convert("L")
    arr = np.array(img).astype(np.uint8)

    if flipud:
        arr = np.flipud(arr)

    if fracture_is_white:
        mask = arr > threshold
    else:
        mask = arr < threshold

    return mask.astype(np.uint8)


def load_gt_mask(path: str,
                 target_shape: Tuple[int, int],
                 *,
                 threshold: int = 127,
                 fracture_is_white: bool = True,
                 flipud: bool = False,
                 black_threshold: int = 10) -> np.ndarray:
    """
    Load a GT PNG, crop to first-black on all sides, resize to target_shape,
    and convert to a binary mask. This matches the debugging procedure.

    target_shape: (H, W) of the prediction mask.
    """
    img = Image.open(path).convert("L")
    arr = np.array(img).astype(np.uint8)

    # 1) crop to first black pixels on all sides
    arr_crop = crop_to_first_black(arr, black_threshold=black_threshold)

    # 2) resize cropped region to match prediction shape (nearest neighbor)
    Ht, Wt = target_shape
    img_crop = Image.fromarray(arr_crop)
    img_rs = img_crop.resize((Wt, Ht), resample=Image.NEAREST)
    arr_rs = np.array(img_rs).astype(np.uint8)

    if flipud:
        arr_rs = np.flipud(arr_rs)

    if fracture_is_white:
        mask = arr_rs > threshold
    else:
        mask = arr_rs < threshold

    return mask.astype(np.uint8)


# ------------------------
# Main driver
# ------------------------

def compute_metrics_from_pngs(
    pred_dir: str,
    gt_dir: str,
    out_csv: Optional[str] = None,
    *,
    pred_prefix: str = "mask_",
    pred_ext: str = ".png",
    gt_ext: str = ".png",
    threshold_pred: int = 127,
    threshold_gt: int = 127,
    fracture_is_white_pred: bool = True,
    fracture_is_white_gt: bool = True,
    flipud_pred: bool = False,
    flipud_gt: bool = False,
    black_threshold: int = 10,
) -> pd.DataFrame:
    """
    Compute per-frame metrics by comparing predicted PNG masks in `pred_dir`
    with GT PNGs in `gt_dir`.

    - Prediction files are expected to look like:  mask_000385.png, etc.
    - GT files are taken in lexicographic order from `gt_dir` (all *.png)
      and zipped with the prediction list. So the first pred matches the
      first GT, second pred -> second GT, etc.

    The GT images are cropped to the first black pixels on all sides, then
    resized to the prediction resolution (nearest neighbor), then both
    masks are thresholded and compared.

    Returns a DataFrame with per-frame metrics and optionally writes CSV.
    """
    pred_dir = str(pred_dir)
    gt_dir = str(gt_dir)

    # Collect prediction files
    pred_files: List[str] = sorted(
        f for f in os.listdir(pred_dir)
        if f.startswith(pred_prefix) and f.endswith(pred_ext)
    )
    if not pred_files:
        raise RuntimeError(f"No prediction files found in {pred_dir} with prefix '{pred_prefix}'.")

    # Collect GT files
    gt_files: List[str] = sorted(
        f for f in os.listdir(gt_dir)
        if f.endswith(gt_ext)
    )
    if not gt_files:
        raise RuntimeError(f"No GT files found in {gt_dir} with extension '{gt_ext}'.")

    if len(gt_files) < len(pred_files):
        print(
            f"[compute_metrics_from_pngs] WARNING: only {len(gt_files)} GT files for "
            f"{len(pred_files)} prediction files. Will only evaluate the first {len(gt_files)} frames."
        )

    n_frames = min(len(pred_files), len(gt_files))

    rows = []
    TP = TN = FP = FN = 0

    for idx in tqdm(range(n_frames), desc="Computing metrics from PNGs"):
        pred_name = pred_files[idx]
        gt_name   = gt_files[idx]

        pred_path = os.path.join(pred_dir, pred_name)
        gt_path   = os.path.join(gt_dir, gt_name)

        # Try to extract frame_id from prediction filename: mask_000385.png -> 385
        stem = pred_name[len(pred_prefix) : -len(pred_ext)]
        if stem.isdigit():
            frame_id = int(stem)
        else:
            frame_id = idx  # fallback

        # Load masks
        pred_bin = load_pred_mask(
            pred_path,
            threshold=threshold_pred,
            fracture_is_white=fracture_is_white_pred,
            flipud=flipud_pred,
        )
        H, W = pred_bin.shape

        gt_bin = load_gt_mask(
            gt_path,
            target_shape=(H, W),
            threshold=threshold_gt,
            fracture_is_white=fracture_is_white_gt,
            flipud=flipud_gt,
            black_threshold=black_threshold,
        )

        # Compute metrics
        yt = gt_bin.reshape(-1)
        yp = pred_bin.reshape(-1)

        prec, rec, f1, acc, tp, tn, fp, fn = metrics_from_binary(yt, yp)

        TP += tp
        TN += tn
        FP += fp
        FN += fn

        rows.append({
            "frame_id": frame_id,
            "frame_index": idx,
            "pred_file": pred_name,
            "gt_file": gt_name,
            "H": H,
            "W": W,
            "Cin": 1,
            "n_pixels": int(yt.size),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "accuracy": float(acc),
        })

    df = pd.DataFrame(rows)

    # Aggregate metrics over all frames
    total_prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    total_rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    total_f1   = 2 * total_prec * total_rec / (total_prec + total_rec) if (total_prec + total_rec) > 0 else 0.0
    total_acc  = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

    print("\nTOTAL METRICS (over evaluated frames)")
    print(f"Frames:    {n_frames}")
    print(f"TP: {TP}  TN: {TN}  FP: {FP}  FN: {FN}")
    print(f"Precision: {total_prec:.6f}")
    print(f"Recall:    {total_rec:.6f}")
    print(f"F1:        {total_f1:.6f}")
    print(f"Accuracy:  {total_acc:.6f}")

    # Write CSV if requested
    if out_csv is not None:
        out_csv = str(out_csv)
        out_dir = os.path.dirname(out_csv)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"Saved per-frame metrics CSV to: {out_csv}")

    return df


# Allow CLI usage if you want (optional)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute per-frame metrics from prediction & GT PNGs.")
    parser.add_argument("--pred-dir", required=True, help="Directory with prediction PNG masks.")
    parser.add_argument("--gt-dir", required=True, help="Directory with GT PNGs.")
    parser.add_argument("--out-csv", default="per_frame_metrics.csv", help="Path to output CSV.")
    args = parser.parse_args()

    compute_metrics_from_pngs(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        out_csv=args.out_csv,
    )
