# FractureTAU Dashboard -- tau_base

> # ⚠ DRAFT — NOT A RESULT
>
> **This report is NOT paper-ready and must not be cited as a result.** The Phase-2
> deliverable is the dashboard *machinery* (`make_report.py` + the aggregate/plots/matrix
> producers), which is accepted. The numbers and figures below are **provisional** and mix
> two non-publishable sources:
>
> - **REAL but OLD:** the **Headline (macro F1)** table, the **Rollout stability** band
>   (`figures/stability.png`), and the **Coverage matrix** come from the **June-10, 2026
>   baseline `tau_base` evaluation** — an early, unoptimized BASE model. These are genuine
>   numbers, but they are stale and do not reflect a current or optimized model.
> - **SYNTHETIC (removed):** the **Threshold sensitivity** curve and the **Qualitative
>   (FP/FN) panels** were generated from *synthetic placeholder* `probs.npz`/`gt.npz` during a
>   smoke run, because the June-10 eval predates the Plan 02-03 raw-probability emission.
>   Those synthetic figures have been **removed** from the repository so they cannot be
>   mistaken for results. The sections are retained (D-14 order intact) but marked *pending*.
>
> **A real Gilbreth re-eval that emits real `probs.npz`/`gt.npz` (Plan 02-03) is required
> before this report is paper-ready.** After that re-eval, regenerate everything with one
> command: `python results/dashboard/make_report.py --run-name tau_base`.

Authoritative, git-diffable static report scaffold (D-08). Regenerate with `python results/dashboard/make_report.py --run-name tau_base`. Sections follow the locked D-14 order.

> **Provenance:** Headline / stability / coverage are computed from the REAL **June-10, 2026
> baseline** `tau_base` per-frame metrics (16 BASE cases; `per_frame_metrics.csv` dated
> 2026-06-10) — an early, unoptimized BASE model, NOT a current/optimized result. The
> threshold-sensitivity curve and FP/FN panels were generated from SYNTHETIC placeholder
> `probs.npz`/`gt.npz` (seeded) because the June-10 eval predates the Plan 02-03
> raw-probability emission; those synthetic figures have been **removed** pending a Gilbreth
> re-eval that emits real saved probs. `calibration.json` was likewise a synthetic placeholder.
> The ConvLSTM-reference macro-F1 column reads 'not yet evaluated' until canonical-keyed
> reference CSVs are regenerated.

## Headline (macro F1)

> _Source: **June-10, 2026 baseline `tau_base` evaluation** — an early, unoptimized BASE
> model. REAL numbers, but stale; not a current or optimized result._

Macro F1 (D-01) is the head-to-head **win claim** (per-frame F1 averaged over the rollout); micro F1 (D-02) is a labelled **secondary** column. Degenerate convention: an empty-GT frame scores 1.0 when the prediction is also empty (0/0 -> 1.0).

| Case | FractureTAU macro F1 | ConvLSTM-ref macro F1 | micro F1 (secondary) | late-rollout F1 |
|---|---:|---:|---:|---:|
| test_MS206_V100 | 0.2746 | not yet evaluated | 0.0000 | 0.0000 |
| test_MS206_V1000 | 0.7603 | not yet evaluated | 0.8033 | 0.7887 |
| test_MS206_V200 | 0.0989 | not yet evaluated | 0.0000 | 0.0000 |
| test_MS206_V400 | 0.6666 | not yet evaluated | 0.7306 | 0.6843 |
| test_MS210_V400 | 0.4852 | not yet evaluated | 0.4650 | 0.4679 |
| test_MS5_V150ms_inc | 0.5903 | not yet evaluated | 0.0000 | 0.0000 |
| test_MS5_V400ms | 0.7333 | not yet evaluated | 0.7984 | 0.7711 |
| test_PBX1_V400_true | 0.3558 | not yet evaluated | 0.3299 | 0.3305 |
| test_PBX_2_V400 | 0.1750 | not yet evaluated | 0.1904 | 0.1934 |
| test_emergency_horizontal | 0.5167 | not yet evaluated | 0.4638 | 0.4318 |
| test_horizontal_layers_3 | 0.6624 | not yet evaluated | 0.7059 | 0.7329 |
| test_horizontal_layers_4 | 0.5723 | not yet evaluated | 0.4684 | 0.4313 |
| test_inclusions_1_2 | 0.6674 | not yet evaluated | 0.6934 | 0.6901 |
| test_inclusions_2_2 | 0.7115 | not yet evaluated | 0.7414 | 0.7200 |
| test_inclusions_3_2 | 0.6563 | not yet evaluated | 0.6464 | 0.6124 |
| test_inclusions_true_V400 | 0.6926 | not yet evaluated | 0.8049 | 0.7950 |

## Rollout stability

> _Source: **June-10, 2026 baseline `tau_base` evaluation** (REAL but stale, early
> unoptimized BASE model). The figure below is real; the underlying model is not current._

Per-case foreground-F1 curves are resampled onto a common normalized **rollout fraction** [0, 1] grid (`np.interp`); the band is the **median + IQR (25-75%)** across cases at matched rollout fractions (OQ3/D-11) -- never absolute frame index, never mean +/- std.

![rollout stability band](figures/stability.png)

## Coverage matrix (16x4)

> _Source: **June-10, 2026 baseline `tau_base` evaluation** (REAL but stale, early
> unoptimized BASE model). Only the BASE column reflects an actual evaluation._

Every canonical case x mode cell is reported explicitly; cells with no evaluation read **not yet evaluated** (D-09) rather than being dropped. Only BASE is fully regenerated in Phase 1.

| Case | BASE | SED | VON | PRESSURE |
|---|---|---|---|---|
| test_MS206_V100 | new only (0.0000) | baseline only (0.0000) | not yet evaluated | not yet evaluated |
| test_MS206_V200 | new only (0.0000) | baseline only (0.5205) | not yet evaluated | not yet evaluated |
| test_MS206_V400 | new only (0.7306) | baseline only (0.6907) | not yet evaluated | not yet evaluated |
| test_MS206_V1000 | new only (0.8033) | not yet evaluated | not yet evaluated | not yet evaluated |
| test_MS210_V400 | new only (0.4650) | not yet evaluated | not yet evaluated | not yet evaluated |
| test_PBX1_V400_true | new only (0.3299) | not yet evaluated | not yet evaluated | not yet evaluated |
| test_PBX_2_V400 | new only (0.1904) | not yet evaluated | not yet evaluated | not yet evaluated |
| test_MS5_V150ms_inc | new only (0.0000) | not yet evaluated | not yet evaluated | not yet evaluated |
| test_MS5_V400ms | new only (0.7984) | not yet evaluated | baseline only (0.6201) | baseline only (0.8947) |
| test_emergency_horizontal | new only (0.4638) | not yet evaluated | not yet evaluated | not yet evaluated |
| test_horizontal_layers_3 | new only (0.7059) | not yet evaluated | not yet evaluated | not yet evaluated |
| test_horizontal_layers_4 | new only (0.4684) | not yet evaluated | not yet evaluated | not yet evaluated |
| test_inclusions_1_2 | new only (0.6934) | not yet evaluated | not yet evaluated | not yet evaluated |
| test_inclusions_2_2 | new only (0.7414) | not yet evaluated | not yet evaluated | not yet evaluated |
| test_inclusions_3_2 | new only (0.6464) | not yet evaluated | not yet evaluated | not yet evaluated |
| test_inclusions_true_V400 | new only (0.8049) | not yet evaluated | not yet evaluated | not yet evaluated |

## Threshold sensitivity

> ⚠ **PENDING REAL EVALUATION** — _The synthetic placeholder figure has been removed. This
> section requires real saved probabilities (`probs.npz`/`gt.npz`) from a Gilbreth re-eval
> with the Plan 02-03 probability emission. Regenerate with
> `python results/dashboard/make_report.py --run-name tau_base` after that re-eval._

FractureTAU macro F1 swept over the binarization threshold on the saved probabilities, no-healing applied. This is **FractureTAU-only** (OQ4): the ConvLSTM reference is pre-binarized / fixed-threshold and is never re-rolled. The threshold is reused, not recomputed -- the calibrated threshold is marked (dashed) once a real `calibration.json` exists.

## Qualitative panels

> ⚠ **PENDING REAL EVALUATION** — _The synthetic placeholder panels (curated subset under
> `results/figures/` and all-16 under `results/diagnostics/`) have been removed. FP/FN panels
> require real saved probabilities (`probs.npz`/`gt.npz`) from a Gilbreth re-eval with the
> Plan 02-03 probability emission. Regenerate with
> `python results/dashboard/make_report.py --run-name tau_base` after that re-eval._

FP/FN overlays on a dark background: **TP gray (160,160,160)**, **FP red (220,40,40)**, **FN blue (40,90,220)** (D-14). GT and prediction are binarized from the SAME saved probabilities in one orientation (no mirror-flip; Pitfall 2). All-case panels are written under `results/diagnostics/`; a curated subset (best / worst / representative by macro F1) is embedded once a real evaluation exists.

