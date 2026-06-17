#!/usr/bin/env python3
"""Slide-ready figure generator for the Kathleen-reproduction demo (Phase 6).

Builds one high-res comparison PNG per case showing the EVOLUTION of the fracture
over the autoregressive rollout for, on stacked rows:

    GT      (ground-truth fracture)         -- optional, needs a --gt source
    BASE    (mode-1 ConvLSTM prediction)
    SED     (mode-4 ConvLSTM prediction)    -- optional

across a handful of key frames (early -> final), with a per-case F1-over-time
curve (BASE vs SED, from per_frame_metrics.csv) underneath. Drop the PNGs straight
into PowerPoint / Keynote / Google Slides.

It consumes ONLY what `evaluate.py` already writes -- the per-frame prediction
masks ``<eval-dir>/<case>/mask_NNNNNN.png`` and ``per_frame_metrics.csv`` -- plus
an optional ground-truth source. It NEVER imports torch or tensorflow and never
touches Kathleen's verified ``source/`` modules (numpy + matplotlib + Pillow only).

Ground truth (optional): pass ``--gt`` as either
  * a ``.npz`` with a (T, H, W) uint8 stack (e.g. the new-model's saved gt.npz), or
  * a directory of ``*.png`` GT masks (sorted by name).
If omitted, the figure renders the BASE/SED rows only and labels GT "not provided".

Run (after pull_run.sh brings the eval outputs local):
    python kathleens-model/scripts/make_repro_slides.py \
        --case test_inclusions_1_2 \
        --base-eval kathleens-model/OUTPUTS/base_repro_eval \
        --sed-eval  kathleens-model/OUTPUTS/sed_repro_eval \
        --gt        new_model/OUTPUTS/tau_base/eval/test_inclusions_1_2/gt.npz \
        --out       slides/

    # batch over several cases (GT auto-resolved per case under --gt-root if given):
    python kathleens-model/scripts/make_repro_slides.py \
        --cases test_MS206_V1000 test_inclusions_1_2 \
        --base-eval kathleens-model/OUTPUTS/base_repro_eval \
        --sed-eval  kathleens-model/OUTPUTS/sed_repro_eval --out slides/
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")                      # headless, no display required
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ---- crack-on-dark colormap (nice, high-contrast on slides) ----
_CRACK_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "crack", ["#0d1b2a", "#e0e1dd"]        # deep navy background -> bone crack
)
PRED_PREFIX = "mask"
F1_KEYS = ("f1", "F1", "f1_score")          # tolerate header variants


# ---- loaders (framework-free) --------------------------------------------
def _load_mask_stack(case_dir: Path, prefix: str = PRED_PREFIX) -> np.ndarray:
    """Stack ``<case_dir>/<prefix>_NNNNNN.png`` (sorted) into (T, H, W) float in [0,1]."""
    pngs = sorted(case_dir.glob(f"{prefix}_*.png"))
    if not pngs:
        raise FileNotFoundError(f"[slides] no {prefix}_*.png under {case_dir}")
    frames = [np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0 for p in pngs]
    return np.stack(frames, axis=0)


def _load_gt_stack(gt_arg: Optional[Path]) -> Optional[np.ndarray]:
    """Load a (T, H, W) GT stack from an .npz (first/`gt`/`arr_0` key) or a dir of PNGs."""
    if gt_arg is None:
        return None
    if gt_arg.is_dir():
        return _load_mask_stack(gt_arg, prefix="")  # any *.png, sorted
    with np.load(gt_arg, allow_pickle=False) as z:
        key = "gt" if "gt" in z else ("arr_0" if "arr_0" in z else z.files[0])
        arr = np.asarray(z[key], dtype=np.float32)
    arr = arr.squeeze()
    if arr.ndim != 3:
        raise ValueError(f"[slides] GT {gt_arg} squeezed to {arr.shape}, expected (T,H,W)")
    return (arr > 0.5).astype(np.float32)


def _read_f1_curve(case_dir: Path) -> Optional[np.ndarray]:
    """Per-frame F1 column from ``<case_dir>/per_frame_metrics.csv`` (None if absent)."""
    csv_path = case_dir / "per_frame_metrics.csv"
    if not csv_path.exists():
        return None
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    key = next((k for k in F1_KEYS if k in rows[0]), None)
    if key is None:
        return None
    return np.array([float(r[key]) for r in rows], dtype=np.float32)


# ---- key-frame selection -------------------------------------------------
def _key_frame_indices(n: int, n_cols: int) -> List[int]:
    """Evenly spaced indices across the rollout, always including the final frame."""
    if n <= n_cols:
        return list(range(n))
    fracs = np.linspace(0.10, 1.0, n_cols)            # early -> final
    return sorted({min(n - 1, int(round(f * (n - 1)))) for f in fracs})


def _orient(a: np.ndarray, flip: bool) -> np.ndarray:
    return np.flipud(a) if flip else a


# ---- one-case figure -----------------------------------------------------
def build_case_figure(
    case: str,
    base_eval: Path,
    *,
    sed_eval: Optional[Path] = None,
    gt: Optional[np.ndarray] = None,
    n_cols: int = 4,
    gt_flip: bool = False,
    out_path: Optional[Path] = None,
    dpi: int = 160,
) -> Path:
    """Assemble the GT/BASE/SED evolution filmstrip + F1 curve for one case."""
    base = _load_mask_stack(base_eval / case)
    sed = _load_mask_stack(sed_eval / case) if sed_eval else None

    rows: List[tuple[str, Optional[np.ndarray]]] = [("Ground truth", gt),
                                                    ("BASE prediction", base)]
    if sed is not None:
        rows.append(("SED prediction", sed))

    ref_len = base.shape[0]
    cols = _key_frame_indices(ref_len, n_cols)
    n_row, n_col = len(rows), len(cols)

    fig = plt.figure(figsize=(2.5 * n_col, 2.5 * n_row + 1.6))
    gs = fig.add_gridspec(n_row + 1, n_col, height_ratios=[*([1] * n_row), 0.7],
                          hspace=0.18, wspace=0.06)

    for r, (label, stack) in enumerate(rows):
        for c, fi in enumerate(cols):
            ax = fig.add_subplot(gs[r, c])
            ax.set_xticks([]); ax.set_yticks([])
            if stack is None:
                ax.text(0.5, 0.5, "GT\nnot provided", ha="center", va="center",
                        fontsize=9, color="#888")
                ax.set_facecolor("#0d1b2a")
            else:
                idx = min(int(round(fi / max(ref_len - 1, 1) * (stack.shape[0] - 1))),
                          stack.shape[0] - 1)
                frame = _orient(stack[idx], gt_flip if label.startswith("Ground") else False)
                ax.imshow(frame, cmap=_CRACK_CMAP, vmin=0.0, vmax=1.0, aspect="auto")
            if r == 0:
                pct = int(round(100 * fi / max(ref_len - 1, 1)))
                ax.set_title(f"frame {fi}  ({pct}%)", fontsize=9)
            if c == 0:
                ax.set_ylabel(label, fontsize=11, fontweight="bold")

    # ---- F1-over-time curve (BASE vs SED) ----
    axf = fig.add_subplot(gs[n_row, :])
    f1_base = _read_f1_curve(base_eval / case)
    if f1_base is not None:
        axf.plot(f1_base, label=f"BASE (mean {f1_base.mean():.3f})", color="#1f77b4", lw=1.8)
    if sed_eval is not None:
        f1_sed = _read_f1_curve(sed_eval / case)
        if f1_sed is not None:
            axf.plot(f1_sed, label=f"SED (mean {f1_sed.mean():.3f})", color="#d62728", lw=1.8)
    axf.set_xlabel("rollout frame"); axf.set_ylabel("F1")
    axf.set_ylim(0, 1.02); axf.grid(alpha=0.3); axf.legend(fontsize=9, loc="lower left")

    fig.suptitle(f"{case} — fracture evolution: ground truth vs ConvLSTM prediction",
                 fontsize=13, fontweight="bold")

    out_path = out_path or Path("slides") / f"{case}_evolution.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[slides] wrote {out_path}")
    return out_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Generate reproduction comparison slides (PNG).")
    ap.add_argument("--case", default=None, help="single case name")
    ap.add_argument("--cases", nargs="*", default=None, help="batch of case names")
    ap.add_argument("--base-eval", required=True, type=Path,
                    help="dir whose children are <case>/mask_*.png (BASE eval run)")
    ap.add_argument("--sed-eval", default=None, type=Path, help="SED eval run dir (optional)")
    ap.add_argument("--gt", default=None, type=Path,
                    help="single-case GT: .npz (T,H,W) or dir of PNGs")
    ap.add_argument("--gt-root", default=None, type=Path,
                    help="batch GT: <gt-root>/<case>/gt.npz resolved per case")
    ap.add_argument("--gt-flip", action="store_true",
                    help="flipud the GT to match the prediction orientation (her masks are flipud'd)")
    ap.add_argument("--n-cols", type=int, default=4, help="key frames per row")
    ap.add_argument("--out", default=Path("slides"), type=Path, help="output dir")
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args(argv)

    cases = args.cases or ([args.case] if args.case else None)
    if not cases:
        raise SystemExit("[slides] provide --case or --cases")

    for case in cases:
        gt_arg = args.gt if args.gt else (
            (args.gt_root / case / "gt.npz") if args.gt_root else None)
        gt_stack = _load_gt_stack(gt_arg) if (gt_arg and gt_arg.exists()) else None
        build_case_figure(
            case, args.base_eval, sed_eval=args.sed_eval, gt=gt_stack,
            n_cols=args.n_cols, gt_flip=args.gt_flip,
            out_path=args.out / f"{case}_evolution.png", dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
