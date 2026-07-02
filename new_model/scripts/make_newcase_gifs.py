#!/usr/bin/env python3
"""Render the New-Case qualitative deliverable: GT | FractureTAU | ConvLSTM.

For each held-out (OOD) new case (new_MS211_V400, new_MS221_V400,
new_MS231_V400) this produces two artifacts under ``OUTPUTS/newcase_viz/``:

  * ``<case>_gt_tau_cnn.gif`` -- an animated 3-column triptych stepping through
    the aligned autoregressive rollout (Ground truth | FractureTAU | ConvLSTM).
  * ``<case>_panel.png`` -- a single still 3-column panel at a shared,
    late-rollout representative timestep (developed fracture).

Plus an optional ``combined_panel.png`` (one still row per case) for reference.

CORRECTNESS (no fabricated / mismatched frames -- user's standing rule):
  * FractureTAU and ConvLSTM roll out INDEPENDENTLY; their per-frame PNG
    sequences can differ in length. Every column of every panel is aligned to
    the SAME rollout step index (step 0 == first predicted frame == GT frame T).
  * The common length is ``L = min(#TAU frames, #CNN frames, #GT frames)``.
    Frames ``0..L-1`` are rendered; if any source is longer it is TRIMMED and
    the trim is LOGGED (per-source counts + L printed). Nothing is padded,
    resampled, or interpolated.
  * Fail loud (SystemExit) if a case's TAU frames dir, CNN mask dir, or the
    FractureTAU ``probs.npz``/``gt.npz`` are missing.

ORIENTATION CONTRACT (documented, single convention):
  * The per-frame ``mask_*.png`` written by both pipelines are ``np.flipud``'d
    into the canonical *display* orientation (old-pipeline convention; see
    ``new_model/src/evaluate.py::save_mask_png`` and
    ``kathleens-model/source/predict_full_simulation_metrics.py``).
  * The raw arrays in ``probs.npz`` (key ``probs``) and ``gt.npz`` (key ``gt``,
    loaded via ``dashboard.probs_io.load_case_probs_gt``) are NOT flipud'd.
  * This script normalizes EVERYTHING to the *display* (flipud'd) orientation:
    the authoritative GT stack is ``np.flipud``'d per frame so GT / TAU / CNN
    columns are all in the same orientation. An orientation sanity IoU
    (GT-vs-TAU, GT-vs-CNN) is printed at the still frame -- informational only.

DATA SOURCES (produced on Gilbreth by the Phase-9 eval; NOT local):
  * FractureTAU (seed 42 chosen as the representative -- noted in the deck):
      new_model/OUTPUTS/headline_s42/eval_newcase/<case>/frames/mask_*.png
      new_model/OUTPUTS/headline_s42/eval_newcase/<case>/{probs,gt}.npz
  * ConvLSTM (frozen ref):
      kathleens-model/OUTPUTS/base_regen_newcase/<case>/mask_*.png
  * Ground truth: gt_stack from the FractureTAU case dir (authoritative, aligned
    to TAU's predicted frames).

Framework rule (CLAUDE.md D-13): numpy + stdlib + matplotlib(Agg) + PIL ONLY.
NO torch, NO tensorflow (this runs anywhere, cluster or laptop).

------------------------------------------------------------------------------
RUN / RSYNC / COMPILE (the exact reproducible sequence)
------------------------------------------------------------------------------
1) Push code to Gilbreth (from your laptop, inside the nested repo):
     git -C dynamic-fracture push
   then on Gilbreth: git -C <repo> pull

2) Render on Gilbreth (where the eval outputs live):
     cd <repo>/dynamic-fracture
     python new_model/scripts/make_newcase_gifs.py
   (add --still-index N to pin a specific shared timestep; default = late rollout)

3) Rsync the media back to your laptop:
     rsync -avz gilbreth:<repo>/dynamic-fracture/new_model/OUTPUTS/newcase_viz/ \
        dynamic-fracture/new_model/OUTPUTS/newcase_viz/

4) Compile the Beamer deck locally (panels must be present from step 3):
     cd dynamic-fracture/paper/slides
     tectonic newcase_slides.tex
   GIFs cannot embed in a PDF -- they ship ALONGSIDE the PDF; each case slide
   footnotes its matching ``<case>_gt_tau_cnn.gif`` filename.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ---- headless, self-sufficient rendering ----
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from PIL import Image  # noqa: E402

# new_model/scripts/make_newcase_gifs.py -> parents[2] == dynamic-fracture repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
# results/ holds the numpy-only dashboard package (the canonical probs/gt loader).
for _p in (REPO_ROOT, REPO_ROOT / "results"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from case_registry import NEW_TEST_CASE_FOLDERS  # noqa: E402
from dashboard.probs_io import load_case_probs_gt  # noqa: E402

# ---- defaults (all overridable via CLI) ----
DEFAULT_TAU_ROOT = REPO_ROOT / "new_model" / "OUTPUTS" / "headline_s42" / "eval_newcase"
DEFAULT_CNN_ROOT = REPO_ROOT / "kathleens-model" / "OUTPUTS" / "base_regen_newcase"
DEFAULT_OUT_DIR = REPO_ROOT / "new_model" / "OUTPUTS" / "newcase_viz"
DEFAULT_CASES = list(NEW_TEST_CASE_FOLDERS.keys())

# Clean binary crack-vs-background colormap: white background, red crack.
CRACK_CMAP = ListedColormap(["white", "#c1121f"])
COL_TITLES = ("Ground truth", "FractureTAU", "ConvLSTM")


# ---- IO helpers ----
def _load_mask_pngs(frames_dir: Path) -> np.ndarray:
    """Load a sorted stack of binary ``mask_*.png`` as (n, H, W) uint8 {0,1}.

    The PNGs are already in *display* (flipud'd) orientation. Fail loud if the
    directory has no ``mask_*.png``.
    """
    paths = sorted(frames_dir.glob("mask_*.png"))
    if not paths:
        raise SystemExit(f"[newcase-viz] FATAL: no mask_*.png under {frames_dir}")
    frames = []
    for p in paths:
        arr = np.asarray(Image.open(p).convert("L"))
        frames.append((arr >= 128).astype(np.uint8))
    return np.stack(frames, axis=0)


def _resolve_cnn_dir(cnn_root: Path, case: str) -> Path:
    """Resolve the ConvLSTM mask dir for a case, trying the registry key then the
    raw data-folder name (e.g. F_MS211_V400_out). Fail loud if neither exists."""
    candidates = [cnn_root / case]
    folder = NEW_TEST_CASE_FOLDERS.get(case)
    if folder:
        candidates.append(cnn_root / folder)
    for c in candidates:
        if c.is_dir() and any(c.glob("mask_*.png")):
            return c
    raise SystemExit(
        f"[newcase-viz] FATAL: no ConvLSTM mask dir for '{case}' "
        f"(tried: {', '.join(str(c) for c in candidates)})"
    )


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


# ---- rendering ----
def _build_triptych(
    gt: np.ndarray, tau: np.ndarray, cnn: np.ndarray, title: str, dpi: int
) -> Image.Image:
    """Render a single GT | FractureTAU | ConvLSTM triptych to a PIL image."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.7), dpi=dpi)
    for ax, img, col in zip(axes, (gt, tau, cnn), COL_TITLES):
        ax.imshow(img, cmap=CRACK_CMAP, vmin=0, vmax=1,
                  interpolation="nearest", aspect="equal", origin="upper")
        ax.set_title(col, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("0.7")
    fig.suptitle(title, fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return Image.fromarray(rgb)


def _short(case: str) -> str:
    """new_MS211_V400 -> MS211 (readable title)."""
    import re

    m = re.search(r"(MS\d+)", case)
    return m.group(1) if m else case


# ---- per-case driver ----
def process_case(
    case: str, tau_root: Path, cnn_root: Path, out_dir: Path,
    still_index: int, duration_ms: int, gif_dpi: int, panel_dpi: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, str]:
    """Render one case's GIF + still panel. Returns the still-frame (gt,tau,cnn),
    the still index used, and the panel title (for the combined panel)."""
    tau_case = tau_root / case
    tau_frames_dir = tau_case / "frames"
    if not tau_frames_dir.is_dir():
        raise SystemExit(f"[newcase-viz] FATAL: missing TAU frames dir {tau_frames_dir}")

    # Authoritative GT (+ probs) via the canonical numpy-only loader (fail loud
    # inside if probs.npz/gt.npz absent). GT is raw orientation -> flipud to display.
    _probs, gt_raw = load_case_probs_gt(tau_case)          # (n, H, W)
    gt_disp = np.stack([np.flipud(f) for f in gt_raw], axis=0).astype(np.uint8)

    tau = _load_mask_pngs(tau_frames_dir)                  # display orientation
    cnn_dir = _resolve_cnn_dir(cnn_root, case)
    cnn = _load_mask_pngs(cnn_dir)                          # display orientation

    n_tau, n_cnn, n_gt = len(tau), len(cnn), len(gt_disp)
    L = min(n_tau, n_cnn, n_gt)
    if L < 1:
        raise SystemExit(f"[newcase-viz] FATAL: zero common frames for {case} "
                         f"(TAU={n_tau}, CNN={n_cnn}, GT={n_gt})")

    print(f"[newcase-viz] {case}: TAU={n_tau} frames, CNN={n_cnn} frames, "
          f"GT={n_gt} frames -> common L={L}")
    if (n_tau, n_cnn, n_gt) != (L, L, L):
        print(f"[newcase-viz] {case}: TRIMMED to L={L} "
              f"(dropped TAU={n_tau - L}, CNN={n_cnn - L}, GT={n_gt - L}); "
              f"no padding/resampling.")

    # shared representative timestep: default late-rollout (developed fracture).
    if still_index < 0:
        s = min(int(round(0.8 * (L - 1))), L - 1)
    else:
        s = min(still_index, L - 1)
        if still_index > L - 1:
            print(f"[newcase-viz] {case}: --still-index {still_index} > L-1; "
                  f"clamped to {s}.")
    print(f"[newcase-viz] {case}: still panel @ shared step {s} (of 0..{L - 1})")

    # orientation sanity (informational, never a gate).
    print(f"[newcase-viz] {case}: orientation-sanity IoU@{s} "
          f"GT-vs-TAU={_iou(gt_disp[s], tau[s]):.3f}, "
          f"GT-vs-CNN={_iou(gt_disp[s], cnn[s]):.3f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    short = _short(case)

    # --- animated GIF over the aligned rollout ---
    frames_img: List[Image.Image] = []
    for i in range(L):
        title = f"{short} ({case}) -- AR step {i + 1}/{L}"
        frames_img.append(_build_triptych(gt_disp[i], tau[i], cnn[i], title, gif_dpi))
    gif_path = out_dir / f"{case}_gt_tau_cnn.gif"
    frames_img[0].save(
        gif_path, save_all=True, append_images=frames_img[1:],
        loop=0, duration=duration_ms, disposal=2,
    )
    print(f"[newcase-viz] {case}: wrote {gif_path} ({L} frames @ {duration_ms}ms)")

    # --- still panel at the shared representative timestep ---
    panel_title = f"{short} ({case}) -- AR step {s + 1}/{L}"
    panel_img = _build_triptych(gt_disp[s], tau[s], cnn[s], panel_title, panel_dpi)
    panel_path = out_dir / f"{case}_panel.png"
    panel_img.save(panel_path)
    print(f"[newcase-viz] {case}: wrote {panel_path}")

    return gt_disp[s], tau[s], cnn[s], s, panel_title


def _write_combined_panel(rows, out_dir: Path, dpi: int) -> None:
    """Optional: one still row per case (GT | FractureTAU | ConvLSTM)."""
    if not rows:
        return
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(10.5, 2.7 * n), dpi=dpi, squeeze=False)
    for r, (gt, tau, cnn, _s, title) in enumerate(rows):
        for c, (img, col) in enumerate(zip((gt, tau, cnn), COL_TITLES)):
            ax = axes[r][c]
            ax.imshow(img, cmap=CRACK_CMAP, vmin=0, vmax=1,
                      interpolation="nearest", aspect="equal", origin="upper")
            if r == 0:
                ax.set_title(col, fontsize=11)
            if c == 0:
                ax.set_ylabel(title.split(" --")[0], fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("New-case qualitative rollout (still @ shared late timestep)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = out_dir / "combined_panel.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"[newcase-viz] wrote {out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tau-root", default=str(DEFAULT_TAU_ROOT),
                    help="FractureTAU eval_newcase root (<case>/frames + <case>/probs.npz)")
    ap.add_argument("--cnn-root", default=str(DEFAULT_CNN_ROOT),
                    help="ConvLSTM base_regen_newcase root (<case>/mask_*.png)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                    help="output dir for GIFs + panels")
    ap.add_argument("--cases", nargs="*", default=DEFAULT_CASES,
                    help="registry case keys (default: the 3 new cases)")
    ap.add_argument("--still-index", type=int, default=-1,
                    help="shared still timestep; <0 = auto late-rollout (0.8*L)")
    ap.add_argument("--duration", type=int, default=120,
                    help="GIF per-frame duration in ms")
    ap.add_argument("--gif-dpi", type=int, default=90)
    ap.add_argument("--panel-dpi", type=int, default=150)
    ap.add_argument("--no-combined", action="store_true",
                    help="skip the combined multi-case panel")
    args = ap.parse_args(argv)

    tau_root = Path(args.tau_root)
    cnn_root = Path(args.cnn_root)
    out_dir = Path(args.out_dir)

    print(f"[newcase-viz] TAU root: {tau_root}")
    print(f"[newcase-viz] CNN root: {cnn_root}")
    print(f"[newcase-viz] out dir : {out_dir}")
    print(f"[newcase-viz] cases   : {args.cases}")

    rows = []
    for case in args.cases:
        gt, tau, cnn, s, title = process_case(
            case, tau_root, cnn_root, out_dir,
            args.still_index, args.duration, args.gif_dpi, args.panel_dpi,
        )
        rows.append((gt, tau, cnn, s, title))

    if not args.no_combined:
        _write_combined_panel(rows, out_dir, args.panel_dpi)

    print(f"[newcase-viz] done: {len(rows)} case(s) -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
