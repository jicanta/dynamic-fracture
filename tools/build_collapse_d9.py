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
from matplotlib.patches import Patch

# Overlay colours, shared with make_hero_figure.py (Fig 5) so the two comparison
# figures read with ONE legend: correct = black, over-prediction = red, missed = blue.
WHITE = (255, 255, 255)
BLACK = (26, 26, 26)
RED = (196, 30, 58)     # FP: model predicted crack where GT has none (over-prediction)
BLUE = (51, 92, 129)    # FN: GT has crack the model missed


def _binar(m: np.ndarray) -> np.ndarray:
    return (m > 127).astype(np.uint8)


def overlay(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Colour a prediction against GT: black hit, red over-prediction, blue miss."""
    p, g = _binar(pred), _binar(gt)
    img = np.full(g.shape + (3,), WHITE, dtype=np.uint8)
    img[(p == 1) & (g == 1)] = BLACK
    img[(p == 1) & (g == 0)] = RED
    img[(p == 0) & (g == 1)] = BLUE
    return img


def gt_image(gt: np.ndarray) -> np.ndarray:
    """The ground-truth crack alone: black crack on white (the reference row)."""
    g = _binar(gt)
    img = np.full(g.shape + (3,), WHITE, dtype=np.uint8)
    img[g == 1] = BLACK
    return img

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
    # Layout: 3 rows (Ground truth / ConvLSTM / FractureTAU) x 4 cols (case).
    # REBUILT 2026-07-24 (user request): (1) add an explicit GROUND-TRUTH row so the
    # reader sees the target crack before each model; (2) the two model rows are now
    # COLOUR OVERLAYS vs GT (black = correct, red = over-prediction, blue = missed),
    # the same encoding as Fig 5, so a single legend serves both figures and the errors
    # are visible in colour instead of a flat grayscale mask; (3) a larger canvas so the
    # colours are legible on the poster.
    n = len(CASES)
    # F-07 legibility recipe (RESEARCH-PART2 G19): a small figure canvas at high dpi so
    # every glyph prints >= 24 pt inside the poster's Fig-3 column. Width is the gate lever
    # (printed_pt = s_min*X_in*D/P_px, P_px = saved WIDTH), so it stays 1.4*n; the third row
    # and legend only add HEIGHT, which the gate does not constrain. Height 4.7 gives each of
    # the three rows room for its rotated label; the panels stay large enough for the colours.
    N_ROWS = 3
    fig, axes = plt.subplots(N_ROWS, n, figsize=(1.4 * n, 4.7), dpi=400)
    fig.patch.set_facecolor("white")

    for c, (case, idx, regime, cl_f1, tau_f1) in enumerate(CASES):
        cl = load_convlstm(case, idx)
        gt, tau = load_gt_tau(case, idx)
        H, W = cl.shape
        y0, y1, x0, x1 = crop_window(gt, tau, cl, H, W)
        gt_c = gt[y0:y1, x0:x1]
        cl_c = cl[y0:y1, x0:x1]
        tau_c = tau[y0:y1, x0:x1]

        # (image, row-label, F1-or-None). GT row carries no F1 (it is the reference).
        rows = [
            (gt_image(gt_c),          "Ground truth", None),
            (overlay(cl_c, gt_c),     "ConvLSTM",     cl_f1),
            (overlay(tau_c, gt_c),    "FractureTAU",  tau_f1),
        ]
        for row, (img, model, f1) in enumerate(rows):
            ax = axes[row, c]
            ax.imshow(img, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("#888"); s.set_linewidth(0.8)
            if f1 is not None:
                ax.text(0.5, -0.06, f"F1 {f1:.2f}", transform=ax.transAxes,
                        ha="center", va="top", fontsize=12,
                        color=("#B00020" if f1 < 0.4 else "#1B5E20"), fontweight="bold")
            if c == 0:
                ax.set_ylabel(model, fontsize=13, fontweight="bold",
                              rotation=90, labelpad=10, va="center")

        # F-04: short panel label (a)-(d) keyed by column index. Regime colour
        # (works green / collapses red) is data and is kept.
        panel_label = f"({chr(ord('a') + c)})"
        axes[0, c].set_title(panel_label, fontsize=12, pad=6,
                             color=("#B00020" if regime == "collapses" else "#1B5E20"),
                             fontweight="bold")

    # Shared colour legend (matches Fig 5). "correct" is the crack both GT and the model
    # agree on; the GT row above shows that same crack in black.
    legend = [Patch(facecolor=np.array(BLACK) / 255, label="correct"),
              Patch(facecolor=np.array(RED) / 255, label="over-prediction"),
              Patch(facecolor=np.array(BLUE) / 255, label="missed")]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=12,
               frameon=False, bbox_to_anchor=(0.5, 0.0),
               handlelength=1.2, handletextpad=0.5, columnspacing=1.5)

    if titleless:
        fig.tight_layout(rect=(0, 0.05, 1, 1))
    else:
        fig.suptitle("D-9 rollout: ConvLSTM collapse vs FractureTAU (BASE-frozen)",
                     fontsize=12.5, fontweight="bold", y=0.995)
        fig.tight_layout(rect=(0, 0.05, 1, 0.985))
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
