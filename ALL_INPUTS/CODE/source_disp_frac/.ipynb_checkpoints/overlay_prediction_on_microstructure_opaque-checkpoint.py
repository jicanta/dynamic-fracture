# fracture_gc_overlay.py

from __future__ import annotations

import numpy as np
from pathlib import Path
from PIL import Image


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resize_to(im: Image.Image, size_wh: tuple[int, int], *, nearest: bool = True) -> Image.Image:
    if im.size == size_wh:
        return im
    resample = Image.NEAREST if nearest else Image.BILINEAR
    return im.resize(size_wh, resample=resample)


def crop_white_borders_rgb(
    img_rgb: Image.Image,
    *,
    white_thresh: int = 250,
    crop_tb: bool = False,
) -> Image.Image:
    """
    Crop away near-white border columns from LEFT/RIGHT.
    Optionally crop TOP/BOTTOM too.
    """
    rgb = img_rgb.convert("RGB")
    arr = np.array(rgb, dtype=np.uint8)

    near_white = (
        (arr[..., 0] >= white_thresh)
        & (arr[..., 1] >= white_thresh)
        & (arr[..., 2] >= white_thresh)
    )

    h, w = near_white.shape

    col_all_white = near_white.all(axis=0)
    nonwhite_cols = np.where(~col_all_white)[0]
    if nonwhite_cols.size == 0:
        raise ValueError("Microstructure appears entirely white; cannot crop borders.")

    left = int(nonwhite_cols[0])
    right = int(nonwhite_cols[-1]) + 1

    top, bottom = 0, h
    if crop_tb:
        row_all_white = near_white.all(axis=1)
        nonwhite_rows = np.where(~row_all_white)[0]
        if nonwhite_rows.size == 0:
            raise ValueError("Microstructure appears entirely white; cannot crop borders.")
        top = int(nonwhite_rows[0])
        bottom = int(nonwhite_rows[-1]) + 1

    return rgb.crop((left, top, right, bottom))


def fracture_binary_mask(
    mask_img: Image.Image,
    *,
    fracture_is: str = "black",   # "black" or "white"
    threshold: int = 127,
) -> Image.Image:
    """
    Return an L-mode binary mask:
      255 where fracture is present, 0 elsewhere.
    """
    g = mask_img.convert("L")

    if fracture_is == "black":
        return g.point(lambda v: 255 if v <= threshold else 0, mode="L")
    elif fracture_is == "white":
        return g.point(lambda v: 255 if v >= threshold else 0, mode="L")
    else:
        raise ValueError("fracture_is must be 'black' or 'white'")


def micro_opacity_by_fracture(
    mic_img: Image.Image,
    mask_img: Image.Image,
    *,
    unfractured_alpha: float = 0.35,
    fracture_is: str = "black",
    threshold: int = 127,
) -> Image.Image:
    """
    Output rule:
      - Fracture region: full-opacity microstructure
      - Unfractured region: reduced-opacity microstructure

    Returns an RGBA image.
    """
    if not (0.0 <= unfractured_alpha <= 1.0):
        raise ValueError("unfractured_alpha must be in [0, 1].")

    mic = mic_img.convert("RGBA")
    mic = resize_to(mic, mask_img.size, nearest=True)

    frac_mask = fracture_binary_mask(
        mask_img,
        fracture_is=fracture_is,
        threshold=threshold,
    )

    base = mic.copy()
    base.putalpha(int(round(unfractured_alpha * 255)))

    out = base.copy()
    out.paste(mic, (0, 0), mask=frac_mask)
    return out


def batch_overlay_single_micro(
    pred_dir,
    micro_path,
    out_dir,
    *,
    pred_glob: str = "mask_*.png",
    unfractured_alpha: float = 0.35,
    fracture_is: str = "black",
    threshold: int = 127,
    save_resized_micro_once: bool = True,
    crop_micro_white_sides: bool = True,
    crop_white_thresh: int = 250,
    crop_tb: bool = False,
    resize_micro_if_needed: bool = False,
):
    """
    Use ONE microstructure image for ALL masks.
    """
    pred_dir = Path(pred_dir)
    out_dir = ensure_dir(out_dir)
    micro_path = Path(micro_path)

    if not micro_path.exists():
        raise FileNotFoundError(f"Microstructure image not found: {micro_path}")

    pred_paths = sorted(pred_dir.glob(pred_glob))
    if not pred_paths:
        raise ValueError(f"No prediction images found in {pred_dir}/{pred_glob}")

    first_mask = Image.open(pred_paths[0])
    ref_size = first_mask.size

    mic_img = Image.open(micro_path).convert("RGBA")

    if crop_micro_white_sides:
        mic_rgb_cropped = crop_white_borders_rgb(
            mic_img.convert("RGB"),
            white_thresh=crop_white_thresh,
            crop_tb=crop_tb,
        )
        mic_img = mic_rgb_cropped.convert("RGBA")

    if mic_img.size != ref_size:
        if resize_micro_if_needed:
            mic_img = resize_to(mic_img, ref_size, nearest=True)
        else:
            raise ValueError(
                "Microstructure size does not match prediction mask size after cropping.\n"
                f"  micro size = {mic_img.size}\n"
                f"  mask size  = {ref_size}\n"
                "Fix cropping threshold, provide a correctly sized micro image, "
                "or set resize_micro_if_needed=True."
            )

    outputs = []
    for pred_path in pred_paths:
        mask_img = Image.open(pred_path)

        if mask_img.size != ref_size:
            raise ValueError(
                f"Mask size mismatch inside pred set:\n"
                f"  {pred_path.name} size = {mask_img.size}\n"
                f"  expected            = {ref_size}"
            )

        comp = micro_opacity_by_fracture(
            mic_img,
            mask_img,
            unfractured_alpha=unfractured_alpha,
            fracture_is=fracture_is,
            threshold=threshold,
        )

        out_path = out_dir / f"{pred_path.stem}.png"
        comp.save(out_path)
        outputs.append(str(out_path))

    if save_resized_micro_once and outputs:
        mic_img.save(out_dir / "microstructure__used_in_overlays.png")

        mic_unfract = mic_img.copy()
        mic_unfract.putalpha(int(round(unfractured_alpha * 255)))
        mic_unfract.save(out_dir / "microstructure__used_unfracturedAlpha.png")

    print(f"Saved {len(outputs)} outputs to: {out_dir}")
    return outputs