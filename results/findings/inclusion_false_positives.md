# Finding: why FractureTAU (and the ConvLSTM) hallucinate crack at inclusions

**Date:** 2026-06-25
**Case studied:** `test_inclusions_1_2` (raw `F_inclusions_1_2_out`, V400), headline model `headline_s42`
**Status:** confirmed from the model's own input fields (not inferred)
**Reproduce:** `cd dynamic-fracture && .venv/bin/python results/findings/analyze_inclusion_fp.py`
**Figure:** `inclusion_false_positives_Gc_vs_stress.png` (this folder)

---

## TL;DR

On `inclusions_1_2`, FractureTAU produces two **persistent false-positive crack blobs** that are present from the first frame and never disappear. They sit **exactly on the weakest inclusions** in the microstructure — and those inclusions are **almost completely unstressed**. The model is keying on **material weakness** (low fracture toughness `Gc`) and predicting a crack there, while **ignoring that there is no stress to actually drive a crack**. Kathleen's ConvLSTM makes the same error because it is fed the same toughness field. This is a **physics-input / feature-attribution failure mode, not a bug in either model**, and the test case itself is clean.

> An earlier hypothesis ("the inclusions are stress *concentrators*") was **wrong** — the measured von Mises stress at the false-positive sites is essentially zero. The real story is the opposite and more precise: *weak but unstressed* material.

---

## Evidence (measured, aligned to the model's input grid)

Alignment was proven first: the raw `fracture_mask` scattered onto the grid matches `gt.npz` at 99.7% of pixels (IoU 0.83), so the overlays below are trustworthy.

| Quantity at the false-positive pixels | Value | Whole-sample | Reading |
|---|---|---|---|
| Fracture toughness `Gc` | **0.079** | 0.443 | blobs sit on the weakest material |
| Fraction on low-`Gc` (inclusion) material | **89%** | 15% (base rate) | **the blobs ARE inclusions** |
| von Mises stress (early, t0030) | **~1×10⁻⁵** | 0.203 | **essentially zero stress there** |
| Persistent-FP pixel count | 695 | — | two compact, fixed blobs |
| First appearance | frame 0 (real crack starts frame 4) | — | driven by static input, not rollout drift |
| Distance to the eventual real crack | 0 / 695 px within ~5 px | — | genuinely wrong location, not early timing |

The microstructure has ~10 inclusions at two toughness levels (`Gc`≈0.02 very weak, and an intermediate grey level) embedded in a tough matrix (`Gc`=0.5).

---

## Mechanism

1. The model's BASE input includes the **toughness map `Gc`**, where inclusions appear as low-`Gc` (weak) regions.
2. In training, low-`Gc` regions correlate strongly with cracking, so the model learns **"weak material → crack."**
3. It fires on the weakest inclusions — but those inclusions carry **almost no stress**, so there is no mechanical driving force; the real crack forms elsewhere, where stress *and* weakness coincide.
4. Result: a **persistent false positive** at weak-but-unstressed inclusions.

**Why the headline model can't tell the difference:** the headline run is **BASE (`--extra none`)**. Its inputs are the displacements (`ux`,`uy`) + `Gc` + velocity + coords — it is **never given the von Mises / SED stress field directly.** So it cannot easily distinguish "weak *and* stressed" (will crack) from "weak *but* unstressed" (won't). The ConvLSTM, fed the same fields, makes the identical mistake — which is the confirmation that the error lives in the inputs/physics, not in either architecture.

**Why it mainly hurts early-rollout F1:** the two blobs are a fixed ~700-px "crack budget." Early, when the true crack is tiny, they dominate → precision collapses (whole-rollout macro-F1 ≈ 0.59). Late, when the true crack is large, they are a small fraction → F1 recovers (late-rollout ≈ 0.85).

---

## Is the test case broken? No.

`Gc`, stress, and ground truth all scatter correctly (99.7% alignment), the crack is coherent, nothing is corrupt or mislabeled. `inclusions_1_2` is a legitimate microstructure that **exposes a specific, now fully-characterized model limitation** — it is not a data defect.

---

## Open questions / caveats

- **Why these 2 weak inclusions and not all ~5 dark ones?** Low `Gc` is necessary but not sufficient. A secondary factor (likely proximity to the developing crack / loaded edge) selects these two. Not yet pinned down.
- **Would an explicit stress channel fix it?** Plausibly — giving the model von Mises or SED would let it gate "weak" by "actually stressed." **But** Phase-4 breadth runs showed the stress-channel modes scored *lower overall* (VONMISES 0.75, SED 0.58 vs BASE 0.86), so this is a hypothesis for *this failure mode*, not a guaranteed net improvement. Worth a targeted ablation, not a blind config switch.

## Cross-pipeline gotcha: ConvLSTM mask PNGs are vertically flipped vs the new_model grid

**The TF/Keras pipeline saves its mask PNGs with the vertical axis reversed relative to the PyTorch `gt.npz`/`probs.npz` grid.** This does NOT affect any reported metric (each pipeline computes F1 against its own consistent GT), but it silently corrupts any *cross-pipeline visual overlay* (GT-vs-ConvLSTM error maps).

Verified on `test_inclusions_1_2`, late-rollout F1 of ConvLSTM masks vs `gt.npz`:

| transform | `base_regen` (thr 0.8) | thesis `ALL_OUTPUTS` (thr ~0.5) |
|---|---|---|
| identity (as-saved) | 0.376 *(misaligned)* | 0.294 *(misaligned)* |
| **flipud (correct)** | **0.645** ✓ matches headline CSV | **0.581** |

→ **Always apply `mask[::-1, :]` (flipud) to ConvLSTM mask PNGs before overlaying on new_model arrays.** The `make_3way_fair.py` figure does this.

## Why the two ConvLSTM artifacts differ (after correcting the flip): threshold + rollout length

Once aligned, the gap shrinks and is explained by **decision threshold** on the **same frozen weights** (not two different models):

- **thesis `ALL_OUTPUTS`** — default threshold ~0.5 → aligned late-F1 ≈ **0.58**
- **`base_regen`** (headline baseline) — `calibration.json` threshold **0.80** (val-F1 0.945) → aligned late-F1 ≈ **0.65**

The higher threshold suppresses low-confidence hallucinated inclusion rings; `base_regen` also rolled 355 frames vs 251, so its late-20% window sits at a more-developed (higher-F1) stage.

**Fairness:** per Phase-1, val-only threshold calibration is applied identically to both models (CMP-04) — FractureTAU's calibrated threshold is 0.275, the ConvLSTM's is 0.80. So `base_regen` (0.65) is the ConvLSTM at its properly-tuned best, and the headline win (FractureTAU 0.85 vs ConvLSTM 0.65) is against the stronger, fairer baseline.

> **Correction note:** an earlier draft of this doc reported the thesis ConvLSTM at "0.29" and attributed the whole 0.29→0.65 gap to threshold. That 0.29 was mostly an artifact of the vertical-flip misalignment above; the real aligned thesis number is ~0.58.

## Implications for the paper

This is a clean, honest failure-mode analysis: a data-driven fracture model **conflates low toughness with imminent fracture, under-weighting the local stress state** — a limitation shared by the prior ConvLSTM. It motivates **stress-aware inputs or a physics-informed gating term** as future work, and it is a credible "limitations / error analysis" subsection.
