# source/plot_pred_vs_gt_contours.py

from __future__ import annotations

import os
from typing import Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.lines import Line2D


DEFAULT_VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def is_image_file(name: str, valid_exts: Optional[Iterable[str]] = None) -> bool:
    if valid_exts is None:
        valid_exts = DEFAULT_VALID_EXTS
    valid_exts = {ext.lower() for ext in valid_exts}
    return os.path.splitext(name.lower())[1] in valid_exts


def load_fracture_mask(path, threshold=0.5, fracture_is="black"):
    """
    Binary mask where 1 = fracture pixels.
    """
    img = Image.open(path).convert("L")
    arr = np.array(img).astype(np.float32) / 255.0

    if fracture_is == "black":
        return (arr < threshold).astype(np.uint8)
    if fracture_is == "white":
        return (arr > threshold).astype(np.uint8)

    raise ValueError("fracture_is must be 'black' or 'white'")


def resize_mask_nearest(mask, new_h, new_w):
    """
    Nearest-neighbor resize for binary masks (keeps edges crisp).
    """
    im = Image.fromarray((mask * 255).astype(np.uint8))
    im = im.resize((new_w, new_h), resample=Image.NEAREST)
    return (np.array(im) > 127).astype(np.uint8)


def first_last_fracture_cols(mask, min_count=20):
    """
    Returns (left_col, right_col) where fracture exists.
    Uses min_count to avoid noise triggering the crop.
    """
    col_counts = mask.sum(axis=0)
    cols = np.where(col_counts >= min_count)[0]
    if cols.size == 0:
        return None, None
    return int(cols[0]), int(cols[-1])


def first_last_fracture_rows(mask, min_count=20):
    """
    Returns (top_row, bottom_row) where fracture exists.
    Uses min_count to avoid noise triggering the crop.
    """
    row_counts = mask.sum(axis=1)
    rows = np.where(row_counts >= min_count)[0]
    if rows.size == 0:
        return None, None
    return int(rows[0]), int(rows[-1])


def plot_pred_vs_gt_contours(
    pred_path,
    gt_path,
    out_path=None,
    title=None,
    threshold=0.5,
    fracture_is="black",
    resize_if_needed=True,
    crop_gt_lr=True,
    crop_gt_tb=True,
    min_count=20,
    pred_label="ML",
    gt_label="MOOSE",
    pred_color="red",
    gt_color="green",
    show_legend=False,
):
    """
    Make one contour overlay figure for a single prediction/GT pair.
    """
    pred = load_fracture_mask(pred_path, threshold=threshold, fracture_is=fracture_is)
    gt = load_fracture_mask(gt_path, threshold=threshold, fracture_is=fracture_is)

    Hp, Wp = pred.shape
    gt_for_plot = gt

    # Crop GT to first/last "real fracture" cols/rows
    if crop_gt_lr:
        lcol, rcol = first_last_fracture_cols(gt_for_plot, min_count=min_count)
        if lcol is not None:
            gt_for_plot = gt_for_plot[:, lcol : rcol + 1]

    if crop_gt_tb:
        trow, brow = first_last_fracture_rows(gt_for_plot, min_count=min_count)
        if trow is not None:
            gt_for_plot = gt_for_plot[trow : brow + 1, :]

    # Resize GT to match pred
    Hg, Wg = gt_for_plot.shape
    if (Hg, Wg) != (Hp, Wp):
        if not resize_if_needed:
            raise ValueError(
                f"Shape mismatch: pred={pred.shape}, gt={gt_for_plot.shape}. "
                "Set resize_if_needed=True to resize GT."
            )
        gt_for_plot = resize_mask_nearest(gt_for_plot, Hp, Wp)

    # Plot
    fig, ax = plt.subplots(figsize=(4, 8))
    ax.set_facecolor("white")

    # Invisible image to keep orientation
    ax.imshow(
        1.0 - pred.astype(np.float32),
        cmap="gray",
        interpolation="nearest",
        alpha=0.0,
    )

    ax.contour(
        pred.astype(np.float32),
        levels=[0.5],
        colors=pred_color,
        linewidths=0.5,
    )
    ax.contour(
        gt_for_plot.astype(np.float32),
        levels=[0.5],
        colors=gt_color,
        linewidths=0.5,
    )

    if show_legend:
        legend_handles = [
            Line2D([0], [0], color=pred_color, lw=1.5, label=pred_label),
            Line2D([0], [0], color=gt_color, lw=1.5, label=gt_label),
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            frameon=True,
            facecolor="white",
            framealpha=0.9,
        )

    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=12)

    plt.tight_layout()

    if out_path:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(
            out_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.05,
            facecolor="white",
        )
        plt.close(fig)
    else:
        plt.show()


def plot_pred_vs_gt_contours_for_simulation(
    pred_dir,
    gt_dir,
    out_dir,
    *,
    threshold=0.5,
    fracture_is="black",
    resize_if_needed=True,
    crop_gt_lr=True,
    crop_gt_tb=True,
    min_count=20,
    valid_exts=None,
    pred_label="ML",
    gt_label="MOOSE",
    pred_color="red",
    gt_color="green",
    show_legend=False,
    out_prefix="contour_",
    title_mode="filename",
    progress_every=50,
):
    """
    Make contour overlay PNGs for an entire simulation.

    Pairing rule:
      - all image files in pred_dir are sorted
      - all image files in gt_dir are sorted
      - pairs are matched by sorted position

    This is the safest generic option when filenames differ but time order matches.

    Returns:
      info dict with counts and paths.
    """
    if valid_exts is None:
        valid_exts = DEFAULT_VALID_EXTS

    if not os.path.isdir(pred_dir):
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(f"GT directory not found: {gt_dir}")

    os.makedirs(out_dir, exist_ok=True)

    pred_files = sorted([f for f in os.listdir(pred_dir) if is_image_file(f, valid_exts)])
    gt_files = sorted([f for f in os.listdir(gt_dir) if is_image_file(f, valid_exts)])

    if not pred_files:
        raise RuntimeError(f"No prediction image files found in: {pred_dir}")
    if not gt_files:
        raise RuntimeError(f"No GT image files found in: {gt_dir}")

    n_pairs = min(len(pred_files), len(gt_files))


    print(f"Making contour overlays for {n_pairs} frame pairs.")
    print(f"Prediction dir: {pred_dir}")
    print(f"GT dir:         {gt_dir}")
    print(f"Output dir:     {out_dir}")

    for i in range(n_pairs):
        pred_name = pred_files[i]
        gt_name = gt_files[i]

        pred_path = os.path.join(pred_dir, pred_name)
        gt_path = os.path.join(gt_dir, gt_name)

        if title_mode == "filename":
            title = os.path.splitext(pred_name)[0]
        elif title_mode == "none":
            title = None
        else:
            title = f"Frame {i:04d}"

        out_name = f"{out_prefix}{i:06d}.png"
        out_path = os.path.join(out_dir, out_name)

        plot_pred_vs_gt_contours(
            pred_path=pred_path,
            gt_path=gt_path,
            out_path=out_path,
            title=title,
            threshold=threshold,
            fracture_is=fracture_is,
            resize_if_needed=resize_if_needed,
            crop_gt_lr=crop_gt_lr,
            crop_gt_tb=crop_gt_tb,
            min_count=min_count,
            pred_label=pred_label,
            gt_label=gt_label,
            pred_color=pred_color,
            gt_color=gt_color,
            show_legend=show_legend,
        )

        if progress_every and ((i + 1) % progress_every == 0 or (i + 1) == n_pairs):
            print(f"  saved {i+1}/{n_pairs}: {out_name}")

    print("Done.")

    return {
        "n_pred_files": len(pred_files),
        "n_gt_files": len(gt_files),
        "n_pairs_used": n_pairs,
        "pred_dir": pred_dir,
        "gt_dir": gt_dir,
        "out_dir": out_dir,
    }