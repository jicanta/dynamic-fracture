#!/usr/bin/env python3
"""
Assemble poster/assets/paper_figures/convlstm_collapse_d9.png:
a zoomed best/worst panel of ConvLSTM vs FractureTAU for the four human-confirmed
D-9 frames (two "works", two "collapses"). Cropped tight to the true-crack window
(GT union TAU) so the crack fills each tile; never uses a t=0 frame; no white panels.

ConvLSTM side  -> kathleens-model/OUTPUTS/base_regen_d9frames/<case>/mask_NNNNNN.png (BASE-frozen)
FractureTAU    -> new_model/OUTPUTS/headline_s43/eval/<case>/viz/compare_NNNNNN.png (right half)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DF = Path("/home/jicanta/Desktop/trabajo/surf-purdue-2026/dynamic-fracture")
OUT = Path("/home/jicanta/Desktop/trabajo/surf-purdue-2026/poster/assets/paper_figures/convlstm_collapse_d9.png")

# (case, frame_idx, regime, convlstm_f1, tau_f1)  -- F1 from tau_headline_vs_base_regen.md (D-9 facts)
CASES = [
    ("test_inclusions_2_2", 300, "works",     0.7902, 0.9344),
    ("test_MS5_V400ms",     225, "works",     0.7410, 0.9305),
    ("test_MS206_V200",     275, "collapses", 0.2213, 0.9483),
    ("test_MS206_V100",     275, "collapses", 0.0916, 0.9506),
]

W_PANEL = 161  # each mask / compare-half is 161 px wide
MARGIN = 12    # crop margin around the true-crack bbox


def load_convlstm(case: str, idx: int) -> np.ndarray:
    assert idx != 0, "t=0 frame forbidden"
    p = DF / "kathleens-model/OUTPUTS/base_regen_d9frames" / case / f"mask_{idx:06d}.png"
    return np.array(Image.open(p).convert("L"))


def load_gt_tau(case: str, idx: int):
    p = DF / "new_model/OUTPUTS/headline_s43/eval" / case / "viz" / f"compare_{idx:06d}.png"
    a = np.array(Image.open(p).convert("L"))
    gt = a[:, :W_PANEL]
    tau = a[:, a.shape[1] - W_PANEL:]  # right half (skip the divider)
    return gt, tau


def bbox_of(mask_bool: np.ndarray):
    ys, xs = np.where(mask_bool)
    if ys.size == 0:
        return None
    return ys.min(), ys.max(), xs.min(), xs.max()


def crop_window(gt, tau, cl, H, W):
    # Crop window = bbox of GT union TAU union ConvLSTM, so no model's actual prediction is
    # hidden. For "works" cases all three overlap -> tight zoom on the crack. For "collapses"
    # cases the window widens to include ConvLSTM's misplaced/sparse pixels -> the collapse
    # stays honestly visible instead of being cropped away.
    union = (gt > 127) | (tau > 127) | (cl > 127)
    bb = bbox_of(union)
    if bb is None:  # degenerate safety: full panel
        return 0, H, 0, W
    y0, y1, x0, x1 = bb
    y0 = max(0, y0 - MARGIN); y1 = min(H - 1, y1 + MARGIN)
    x0 = max(0, x0 - MARGIN); x1 = min(W - 1, x1 + MARGIN)
    return y0, y1 + 1, x0, x1 + 1


def main(out: Path = OUT, titleless: bool = False):
    out.parent.mkdir(parents=True, exist_ok=True)
    # Layout: 2 rows (model) x 4 cols (case), transposed from the original 4x2.
    # Same tiles, same crops, same data — but each tile renders far larger inside a
    # poster column, and the block costs ~4x less column height, which is what let
    # the poster enlarge this figure instead of shrinking it to fit (D-11: no dead
    # white space, figures over prose).
    n = len(CASES)
    # F-07 legibility recipe (RESEARCH-PART2 G19): a much smaller figure canvas at a
    # higher dpi so every glyph prints >= 24 pt inside the poster's Fig-3 column.
    # figsize width shrinks 2.6*n -> 1.4*n (W_eff ~= 5.1 in after tight-bbox) and dpi
    # rises 220 -> 400 so printed_ppi = 400 * W_eff / X_in >= 150.
    # FIXED 2026-07-24: figure height 2.7 -> 3.4 and hspace 0.22 -> 0.34 (below). At 2.7in
    # the two axes rows were compressed by the (a)-(d) titles + "F1 x.xx" labels, leaving
    # each row shorter than the rotated "ConvLSTM"/"FractureTAU" ylabel -- so the two labels
    # overflowed their rows and collided into one unreadable vertical block on the left. The
    # extra height gives each row room for its rotated label. Width is UNCHANGED (1.4*n), so
    # the legibility gate (printed_pt = s_min*X_in*D/P_px, P_px = saved WIDTH) is unaffected.
    fig, axes = plt.subplots(2, n, figsize=(1.4 * n, 3.4), dpi=400)
    fig.patch.set_facecolor("white")

    for c, (case, idx, regime, cl_f1, tau_f1) in enumerate(CASES):
        cl = load_convlstm(case, idx)
        gt, tau = load_gt_tau(case, idx)
        H, W = cl.shape
        y0, y1, x0, x1 = crop_window(gt, tau, cl, H, W)
        cl_c = cl[y0:y1, x0:x1]
        tau_c = tau[y0:y1, x0:x1]

        for row, (img, model, f1) in enumerate(
            [(cl_c, "ConvLSTM", cl_f1), (tau_c, "FractureTAU", tau_f1)]
        ):
            ax = axes[row, c]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("#888"); s.set_linewidth(0.8)
            ax.text(0.5, -0.06, f"F1 {f1:.2f}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=12,
                    color=("#B00020" if f1 < 0.4 else "#1B5E20"), fontweight="bold")
            if c == 0:
                ax.set_ylabel(model, fontsize=13, fontweight="bold",
                              rotation=90, labelpad=10, va="center")

        # F-04: short panel label (a)-(d) keyed by column index. The per-case name /
        # regime / frame context moves to the poster caption; a single short label is
        # the ONLY per-column text, which is what lets >= 12 pt type fit a small cell
        # and clear F-07's 24 pt printed bar. Regime colour (works green / collapses
        # red) is data and is kept.
        panel_label = f"({chr(ord('a') + c)})"
        axes[0, c].set_title(panel_label, fontsize=12, pad=6,
                             color=("#B00020" if regime == "collapses" else "#1B5E20"),
                             fontweight="bold")

    if titleless:
        # Reviewer request (v2.0): the poster caption already carries this context,
        # so the embedded suptitle band is removed and the strip it occupied reclaimed.
        # Per-tile "F1 x.xx" labels and the per-case subplot titles are DATA — kept.
        fig.tight_layout(rect=(0, 0, 1, 1))
    else:
        fig.suptitle("D-9 rollout: ConvLSTM collapse vs FractureTAU (BASE-frozen)",
                     fontsize=12.5, fontweight="bold", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.subplots_adjust(hspace=0.34, wspace=0.08)
    fig.savefig(out, dpi=400, facecolor="white", bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Assemble the D-9 best/worst collapse panel.")
    ap.add_argument("--out", type=Path, default=OUT,
                    help="output PNG path (default: the original convlstm_collapse_d9.png)")
    ap.add_argument("--titleless", action="store_true",
                    help="omit the embedded suptitle (reviewer v2.0 variant)")
    args = ap.parse_args()
    main(out=args.out, titleless=args.titleless)
