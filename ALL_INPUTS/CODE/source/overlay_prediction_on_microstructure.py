# source/overlay_prediction_on_microstructure.py

from __future__ import annotations
import os
import re
from collections import Counter
from typing import Iterable, Optional

import numpy as np
from PIL import Image


DEFAULT_VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

IGNORE_IMAGE_NAMES = {
    "bce_over_time.png",
}


def is_image_file(name: str, valid_exts: Optional[Iterable[str]] = None) -> bool:
    if valid_exts is None:
        valid_exts = DEFAULT_VALID_EXTS
    valid_exts = {ext.lower() for ext in valid_exts}
    return os.path.splitext(name.lower())[1] in valid_exts


def _natural_key(s: str):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def _get_image_size(path: str) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def _list_image_files(folder: str, valid_exts: Optional[Iterable[str]] = None) -> list[str]:
    return sorted(
        [
            f for f in os.listdir(folder)
            if is_image_file(f, valid_exts)
            and f.lower() not in IGNORE_IMAGE_NAMES
        ],
        key=_natural_key,
    )


def _choose_target_prediction_size(pred_dir: str, pred_files: list[str]) -> tuple[int, int]:
    sizes = []
    for fname in pred_files:
        fpath = os.path.join(pred_dir, fname)
        try:
            sizes.append(_get_image_size(fpath))
        except Exception:
            continue

    if not sizes:
        raise RuntimeError("Could not read any prediction image sizes.")

    size_counts = Counter(sizes)
    target_size, count = size_counts.most_common(1)[0]

    print("Prediction image size counts:")
    for size, n in size_counts.most_common():
        print(f"  {size}: {n}")
    print(f"Chosen target prediction size: {target_size} (count={count})")

    return target_size


def crop_white_borders_rgb(
    img_rgb: Image.Image,
    *,
    white_thresh: int = 250,
    crop_tb: bool = False,
) -> Image.Image:
    arr = np.array(img_rgb, dtype=np.uint8)

    near_white = (
        (arr[..., 0] >= white_thresh)
        & (arr[..., 1] >= white_thresh)
        & (arr[..., 2] >= white_thresh)
    )

    h, w = near_white.shape

    col_all_white = near_white.all(axis=0)
    nonwhite_cols = np.where(~col_all_white)[0]
    if nonwhite_cols.size == 0:
        return img_rgb

    left = int(nonwhite_cols[0])
    right = int(nonwhite_cols[-1]) + 1

    top, bottom = 0, h
    if crop_tb:
        row_all_white = near_white.all(axis=1)
        nonwhite_rows = np.where(~row_all_white)[0]
        if nonwhite_rows.size == 0:
            return img_rgb
        top = int(nonwhite_rows[0])
        bottom = int(nonwhite_rows[-1]) + 1

    return img_rgb.crop((left, top, right, bottom))


def crop_uniform_border_rgb(
    img_rgb: Image.Image,
    *,
    tol: int = 12,
    crop_tb: bool = True,
    pad_px: int = 2,
) -> Image.Image:
    """
    Crop away a near-uniform border by estimating the border color from the corners.
    Good for pressure images with light gray / constant-color backgrounds.
    """
    arr = np.array(img_rgb, dtype=np.uint8)
    h, w, _ = arr.shape

    corners = np.array([
        arr[0, 0],
        arr[0, w - 1],
        arr[h - 1, 0],
        arr[h - 1, w - 1],
    ], dtype=np.int16)

    border_rgb = np.median(corners, axis=0)

    diff = np.abs(arr.astype(np.int16) - border_rgb[None, None, :])
    close_to_border = np.all(diff <= tol, axis=2)

    col_all_border = close_to_border.all(axis=0)
    nonborder_cols = np.where(~col_all_border)[0]
    if nonborder_cols.size == 0:
        return img_rgb

    left = max(0, int(nonborder_cols[0]) - pad_px)
    right = min(w, int(nonborder_cols[-1]) + 1 + pad_px)

    top, bottom = 0, h
    if crop_tb:
        row_all_border = close_to_border.all(axis=1)
        nonborder_rows = np.where(~row_all_border)[0]
        if nonborder_rows.size > 0:
            top = max(0, int(nonborder_rows[0]) - pad_px)
            bottom = min(h, int(nonborder_rows[-1]) + 1 + pad_px)

    return img_rgb.crop((left, top, right, bottom))


def _load_prediction_as_mask(
    pred_path: str,
    target_size: tuple[int, int],
    *,
    resize_predictions: bool,
) -> Image.Image:
    pred_img = Image.open(pred_path).convert("L")

    if pred_img.size != target_size:
        if resize_predictions:
            pred_img = pred_img.resize(target_size, resample=Image.NEAREST)
        else:
            raise ValueError(
                "Prediction image size mismatch inside pred_dir.\n"
                f"  file size   = {pred_img.size}\n"
                f"  target size = {target_size}\n"
                "Set resize_predictions=True, or clean the directory."
            )

    return pred_img


def _crop_background(
    bg: Image.Image,
    *,
    crop_mode: str,
    crop_white_thresh: int,
    crop_tb: bool,
    border_tol: int,
    crop_pad_px: int,
) -> Image.Image:
    """
    crop_mode:
      - 'none'
      - 'white'
      - 'border'
    """
    if crop_mode == "none":
        return bg
    if crop_mode == "white":
        return crop_white_borders_rgb(
            bg,
            white_thresh=crop_white_thresh,
            crop_tb=crop_tb,
        )
    if crop_mode == "border":
        return crop_uniform_border_rgb(
            bg,
            tol=border_tol,
            crop_tb=crop_tb,
            pad_px=crop_pad_px,
        )
    raise ValueError(f"Unknown crop_mode: {crop_mode}")


def _load_and_prepare_background(
    bg_path: str,
    target_size: tuple[int, int],
    *,
    crop_mode: str,
    resize_bg: bool,
    crop_white_thresh: int,
    crop_tb: bool,
    border_tol: int,
    crop_pad_px: int,
) -> tuple[Image.Image, tuple[int, int], tuple[int, int], tuple[int, int]]:
    """
    Load one background image, crop first, then resize.
    Returns:
      bg_final, original_size, cropped_size, final_size
    """
    bg = Image.open(bg_path).convert("RGB")
    original_size = bg.size

    bg = _crop_background(
        bg,
        crop_mode=crop_mode,
        crop_white_thresh=crop_white_thresh,
        crop_tb=crop_tb,
        border_tol=border_tol,
        crop_pad_px=crop_pad_px,
    )
    cropped_size = bg.size

    if bg.size != target_size:
        if resize_bg:
            bg = bg.resize(target_size, resample=Image.BILINEAR)
        else:
            raise ValueError(
                "Background image size does not match target prediction size.\n"
                f"  background size = {bg.size}\n"
                f"  target size     = {target_size}\n"
                "Set resize_micro=True."
            )

    final_size = bg.size
    return bg, original_size, cropped_size, final_size


def overlay_prediction_whitespace_on_microstructure(
    pred_dir: str,
    micro_path: str,
    out_dir: str,
    *,
    threshold: int = 128,
    resize_micro: bool = True,
    resize_predictions: bool = True,
    valid_exts: Optional[Iterable[str]] = None,
    crop_white_thresh: int = 250,
    crop_tb: bool = True,
    border_tol: int = 12,
    crop_pad_px: int = 2,
    out_suffix: str = "_micro_white",
    progress_every: int = 50,
    static_crop_mode: str = "white",
    sequence_crop_mode: str = "border",
    strict_frame_match: bool = True,
) -> dict:
    """
    Overlay fracture predictions onto either:
      1) one static image
      2) a directory of frame-by-frame background images

    crop modes:
      - static_crop_mode:   'none', 'white', 'border'
      - sequence_crop_mode: 'none', 'white', 'border'

    Recommended:
      - microstructure image: static_crop_mode='white'
      - pressure frame dir:   sequence_crop_mode='border'
    """
    if valid_exts is None:
        valid_exts = DEFAULT_VALID_EXTS

    if not os.path.isdir(pred_dir):
        raise FileNotFoundError(f"Prediction folder not found: {pred_dir}")

    os.makedirs(out_dir, exist_ok=True)

    pred_files = _list_image_files(pred_dir, valid_exts)
    if not pred_files:
        raise RuntimeError(f"No usable image files found in prediction folder: {pred_dir}")

    target_size = _choose_target_prediction_size(pred_dir, pred_files)

    print(f"Found {len(pred_files)} prediction images.")
    print(f"Saving to: {out_dir}")

    # --------------------------------
    # MODE 1: single static background
    # --------------------------------
    if os.path.isfile(micro_path):
        mode = "static_background"

        bg_prepared, bg_original_size, bg_cropped_size, bg_final_size = _load_and_prepare_background(
            micro_path,
            target_size,
            crop_mode=static_crop_mode,
            resize_bg=resize_micro,
            crop_white_thresh=crop_white_thresh,
            crop_tb=crop_tb,
            border_tol=border_tol,
            crop_pad_px=crop_pad_px,
        )

        print("Mode: static background image")
        print(f"static_crop_mode         = {static_crop_mode}")
        print(f"Background original size = {bg_original_size}")
        print(f"Background cropped size  = {bg_cropped_size}")
        print(f"Background final size    = {bg_final_size}")

        for i, fname in enumerate(pred_files, 1):
            pred_path = os.path.join(pred_dir, fname)

            pred_img = _load_prediction_as_mask(
                pred_path,
                target_size,
                resize_predictions=resize_predictions,
            )

            pred_np = np.array(pred_img, dtype=np.uint8)
            white_mask = pred_np >= threshold

            bg_np = np.array(bg_prepared, dtype=np.uint8).copy()
            bg_np[white_mask] = (255, 255, 255)

            out_name = os.path.splitext(fname)[0] + f"{out_suffix}.png"
            out_path = os.path.join(out_dir, out_name)
            Image.fromarray(bg_np).save(out_path)

            if progress_every and (i % progress_every == 0 or i == len(pred_files)):
                print(f"  saved {i}/{len(pred_files)}: {out_name}")

        print("Done.")

        return {
            "mode": mode,
            "n_pred_files": len(pred_files),
            "n_background_files": 1,
            "target_size": target_size,
            "background_original_size": bg_original_size,
            "background_cropped_size": bg_cropped_size,
            "background_final_size": bg_final_size,
            "static_crop_mode": static_crop_mode,
            "ignored_filenames": sorted(IGNORE_IMAGE_NAMES),
            "out_dir": out_dir,
        }

    # --------------------------------------
    # MODE 2: framewise background directory
    # --------------------------------------
    elif os.path.isdir(micro_path):
        mode = "framewise_background"

        bg_files = _list_image_files(micro_path, valid_exts)
        if not bg_files:
            raise RuntimeError(f"No usable image files found in background directory: {micro_path}")

        print("Mode: framewise background directory")
        print(f"Found {len(bg_files)} background images.")
        print(f"sequence_crop_mode = {sequence_crop_mode}")

        if strict_frame_match and len(bg_files) != len(pred_files):
            raise ValueError(
                "Number of background images does not match number of prediction images.\n"
                f"  prediction images = {len(pred_files)}\n"
                f"  background images = {len(bg_files)}\n"
                "Either make them match, or set strict_frame_match=False."
            )

        n_to_process = min(len(pred_files), len(bg_files))
        if len(bg_files) != len(pred_files):
            print(
                f"Warning: processing only first {n_to_process} matched frames "
                f"(pred={len(pred_files)}, bg={len(bg_files)})."
            )

        first_bg_path = os.path.join(micro_path, bg_files[0])
        _, bg_original_size, bg_cropped_size, bg_final_size = _load_and_prepare_background(
            first_bg_path,
            target_size,
            crop_mode=sequence_crop_mode,
            resize_bg=resize_micro,
            crop_white_thresh=crop_white_thresh,
            crop_tb=crop_tb,
            border_tol=border_tol,
            crop_pad_px=crop_pad_px,
        )

        print(f"First background original size = {bg_original_size}")
        print(f"First background cropped size  = {bg_cropped_size}")
        print(f"First background final size    = {bg_final_size}")

        for i in range(n_to_process):
            pred_name = pred_files[i]
            bg_name = bg_files[i]

            pred_path = os.path.join(pred_dir, pred_name)
            bg_path = os.path.join(micro_path, bg_name)

            pred_img = _load_prediction_as_mask(
                pred_path,
                target_size,
                resize_predictions=resize_predictions,
            )

            bg_prepared, _, _, _ = _load_and_prepare_background(
                bg_path,
                target_size,
                crop_mode=sequence_crop_mode,
                resize_bg=resize_micro,
                crop_white_thresh=crop_white_thresh,
                crop_tb=crop_tb,
                border_tol=border_tol,
                crop_pad_px=crop_pad_px,
            )

            pred_np = np.array(pred_img, dtype=np.uint8)
            white_mask = pred_np >= threshold

            bg_np = np.array(bg_prepared, dtype=np.uint8).copy()
            bg_np[white_mask] = (255, 255, 255)

            out_name = os.path.splitext(pred_name)[0] + f"{out_suffix}.png"
            out_path = os.path.join(out_dir, out_name)
            Image.fromarray(bg_np).save(out_path)

            if progress_every and ((i + 1) % progress_every == 0 or (i + 1) == n_to_process):
                print(f"  saved {i+1}/{n_to_process}: {out_name}")

        print("Done.")

        return {
            "mode": mode,
            "n_pred_files": len(pred_files),
            "n_background_files": len(bg_files),
            "n_processed": n_to_process,
            "target_size": target_size,
            "first_background_original_size": bg_original_size,
            "first_background_cropped_size": bg_cropped_size,
            "first_background_final_size": bg_final_size,
            "sequence_crop_mode": sequence_crop_mode,
            "ignored_filenames": sorted(IGNORE_IMAGE_NAMES),
            "out_dir": out_dir,
        }

    else:
        raise FileNotFoundError(f"`micro_path` is neither a file nor a directory: {micro_path}")