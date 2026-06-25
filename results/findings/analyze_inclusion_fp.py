#!/usr/bin/env python3
"""Reproduces the inclusion false-positive analysis (see inclusion_false_positives.md).

Shows that FractureTAU's persistent false-positive blobs on test_inclusions_1_2 sit on
the WEAKEST (low-Gc) inclusions, which are NEARLY UNSTRESSED early in the rollout — i.e.
the model conflates low fracture toughness with imminent fracture, ignoring the stress state.

Inputs (all first-party, already pulled from Gilbreth):
  - new_model/OUTPUTS/headline_s42/eval/test_inclusions_1_2/{probs.npz,gt.npz,summary.json}
  - new_model/DATASET/F_inclusions_1_2_out/V400_F_inclusions_1_2_out/*t{0001,0030,0125}*.csv
    (columns: x,y,ux,uy,Gc,pressure,vonmises,fracture_mask,SED,time_ns)

Run from dynamic-fracture/:
    .venv/bin/python results/findings/analyze_inclusion_fp.py
"""
from __future__ import annotations
import glob, json
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation

EVAL = "new_model/OUTPUTS/headline_s42/eval/test_inclusions_1_2"
RAW = "new_model/DATASET/F_inclusions_1_2_out/V400_F_inclusions_1_2_out"


def _load_csv(t):
    return pd.read_csv(sorted(glob.glob(f"{RAW}/*t{t}_*.csv"))[0])


def _grid(df, col, xi, yi, ny, nx):
    """Scatter a column onto the (H=y, W=x) eval grid = (321,161)."""
    g = np.full((ny, nx), np.nan)
    g[df.y.map(yi).values, df.x.map(xi).values] = df[col].values
    return g


def main():
    df1 = _load_csv("0001")
    xs, ys = np.sort(df1.x.unique()), np.sort(df1.y.unique())
    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: i for i, v in enumerate(ys)}
    ny, nx = len(ys), len(xs)  # (321, 161) = (H=y, W=x)

    # --- alignment proof: scattered fracture_mask(t0125) == gt.npz[124] (eval drops t0000)
    fm = _grid(_load_csv("0125"), "fracture_mask", xi, yi, ny, nx) > 0.5
    gt124 = np.load(f"{EVAL}/gt.npz")["gt"][124].astype(bool)
    iou = (fm & gt124).sum() / max(1, (fm | gt124).sum())
    print(f"[align] scattered fracture_mask(t0125) vs gt.npz[124]: IoU={iou:.3f} "
          f"exact-px={np.mean(fm == gt124):.3f}  (>0.8 / ~1.0 confirms alignment)")

    # --- persistent false positives: predicted >30% of frames, GT never cracks there
    p = np.load(f"{EVAL}/probs.npz")["probs"].astype(np.float32)
    g = np.load(f"{EVAL}/gt.npz")["gt"].astype(bool)
    thr = json.load(open(f"{EVAL}/summary.json")).get("threshold", 0.5)
    pred = p > thr
    fp = (~g.any(0)) & (pred.mean(0) > 0.3)

    Gc = _grid(df1, "Gc", xi, yi, ny, nx)
    vm = _grid(_load_csv("0030"), "vonmises", xi, yi, ny, nx)
    low_gc = Gc < (np.nanmean(Gc) * 0.7)

    print(f"[fp] persistent-FP pixels: {fp.sum()}")
    print(f"[Gc] mean Gc on FP={np.nanmean(Gc[fp]):.3f} vs sample={np.nanmean(Gc):.3f} "
          f"(low=weak=inclusion)")
    print(f"[Gc] FP pixels on low-Gc inclusion material: {low_gc[fp].mean():.2f} "
          f"vs base rate {low_gc.mean():.2f}")
    print(f"[stress] mean vonMises(t0030) on FP={np.nanmean(vm[fp]):.2e} "
          f"vs sample={np.nanmean(vm):.3f}  -> FP sites are ~unstressed")

    # --- diagnostic figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    def outline(m, it=2):
        return binary_dilation(m, iterations=it) & ~m
    fpo, gto = outline(fp), outline(g[-1])
    fig, ax = plt.subplots(1, 2, figsize=(9, 7))
    ax[0].imshow(Gc, cmap="cividis")
    ax[0].set_title("Fracture toughness Gc\n(dark = weak = inclusions)", fontsize=11)
    ax[0].imshow(np.dstack([fpo, fpo * 0, fpo * 0, fpo.astype(float)]))
    ax[0].imshow(np.dstack([gto * 0, gto.astype(float), gto * 0, gto.astype(float) * 0.6]))
    ax[1].imshow(vm, cmap="magma")
    ax[1].set_title("von Mises stress (early, t0030)\n(bright = stressed)", fontsize=11)
    ax[1].imshow(np.dstack([fpo, fpo * 0, fpo * 0, fpo.astype(float)]))
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Red = model hallucination  ·  Green = real crack", fontsize=12, y=1.0)
    fig.tight_layout()
    out = "results/findings/inclusion_false_positives_Gc_vs_stress.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[fig] wrote {out}")


if __name__ == "__main__":
    main()
