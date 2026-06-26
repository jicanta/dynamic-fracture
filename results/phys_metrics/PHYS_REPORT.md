# FractureTAU Physical-Metrics Report -- tau_refined

Static, git-diffable physical-fidelity report (D-13). Regenerate with `python results/phys_metrics/make_phys_report.py --run-name tau_refined`. Every reported number traces to a measured artifact (correctness-over-convenience mandate).

## 1. Headline result (3-seed mean +/- std + significance)

Late-rollout (last-20%) macro F1, mean +/- std over **3 seeds** headline_s42, headline_s43, headline_s44 (D-05). bf16 AMP + cuDNN are not byte-deterministic, so the seed spread IS the honest run-to-run variation (Pitfall 9 -- mean+/-std, never bit-determinism).

| Case | late-rollout F1 (mean +/- std) | n seeds |
|---|---:|---:|
| test_MS206_V100 | 0.9506 +/- 0.0186 | 3 |
| test_MS206_V1000 | 0.9674 +/- 0.0074 | 3 |
| test_MS206_V200 | 0.9483 +/- 0.0038 | 3 |
| test_MS206_V400 | 0.9569 +/- 0.0102 | 3 |
| test_MS210_V400 | 0.9321 +/- 0.0019 | 3 |
| test_MS5_V150ms_inc | 0.5971 +/- 0.3230 | 3 |
| test_MS5_V400ms | 0.9305 +/- 0.0332 | 3 |
| test_PBX1_V400_true | 0.7240 +/- 0.1292 | 3 |
| test_PBX_2_V400 | 0.7841 +/- 0.0680 | 3 |
| test_emergency_horizontal | 0.8589 +/- 0.0186 | 3 |
| test_horizontal_layers_3 | 0.9390 +/- 0.0102 | 3 |
| test_horizontal_layers_4 | 0.9359 +/- 0.0167 | 3 |
| test_inclusions_1_2 | 0.8520 +/- 0.0180 | 3 |
| test_inclusions_2_2 | 0.9344 +/- 0.0258 | 3 |
| test_inclusions_3_2 | 0.8234 +/- 0.0715 | 3 |
| test_inclusions_true_V400 | 0.6584 +/- 0.1839 | 3 |
| **OVERALL (per-case mean)** | **0.8621** | 3 |

_not yet available: the 16-case from-scratch ConvLSTM reference (`base_regen`) is not present in this checkout (found 0 paired case(s)); the Wilcoxon p is computed in the on-cluster significance run. No p-value is fabricated here (T-05-16)._

> **Provenance (DATA-04):** winning `tau_refined/checkpoints/best.pt` sha256 `f1a0786c519b`. Per-seed `best.pt` sha256:

> - `headline_s42`: `0f55814f02ae`
> - `headline_s43`: `6a09694abf51`
> - `headline_s44`: `f97f429e8d9f`

> **From-scratch ConvLSTM reproduction scope (D-08):** 4 BASE / 2 SED (corrupt MS206 SED reference exports excluded -- honest verifiable scope).

## 2. Boundary metrics (HD95 + boundary-F1)

Crack-front localization on the binarized (no-healing) AR prediction vs GT, per D-03: **HD95** (reviewer-familiar 95th-pct Hausdorff; empty-mask -> NaN, excluded from the mean) and **boundary-F1** (BF score within 2 px). Per-frame metrics are averaged within each case.

| Case | mean HD95 (px) | mean boundary-F1 |
|---|---:|---:|
| test_MS206_V100 | NaN | 0.2746 |
| test_MS206_V1000 | 2.558 | 0.9656 |
| test_MS206_V200 | 1.813 | 0.9440 |
| test_MS206_V400 | 31.543 | 0.9265 |
| test_MS210_V400 | 58.613 | 0.6747 |
| test_MS5_V150ms_inc | 4.529 | 0.7976 |
| test_MS5_V400ms | 7.234 | 0.8918 |
| test_PBX1_V400_true | 111.173 | 0.0865 |
| test_PBX_2_V400 | 70.247 | 0.1530 |
| test_emergency_horizontal | 50.350 | 0.7630 |
| test_horizontal_layers_3 | 11.404 | 0.8834 |
| test_horizontal_layers_4 | 7.260 | 0.9234 |
| test_inclusions_1_2 | 16.100 | 0.6733 |
| test_inclusions_2_2 | 13.464 | 0.6361 |
| test_inclusions_3_2 | 84.188 | 0.1891 |
| test_inclusions_true_V400 | 123.297 | 0.2301 |
| **MACRO (over 16 cases)** | **39.585** | **0.6258** |

## 3. Physical metrics (crack length-over-time + onset)

Crack length = skeleton-pixel count per frame (reported in PIXELS; pixel-pitch left at 1.0 -- A3 honesty); time-to-onset = first frame with any predicted crack. GT vs FractureTAU (AR). The length-over-time figure below is the representative case (median macro F1).

| Case | onset (GT) | onset (pred) | final length GT (px) | final length pred (px) |
|---|---:|---:|---:|---:|
| test_MS206_V100 | 81 | -1 | 15.0 | 0.0 |
| test_MS206_V1000 | 1 | 4 | 652.0 | 596.0 |
| test_MS206_V200 | 28 | 36 | 77.0 | 57.0 |
| test_MS206_V400 | 10 | 13 | 304.0 | 193.0 |
| test_MS210_V400 | 3 | 9 | 585.0 | 177.0 |
| test_MS5_V150ms_inc | 170 | 218 | 49.0 | 34.0 |
| test_MS5_V400ms | 25 | 31 | 573.0 | 455.0 |
| test_PBX1_V400_true | 23 | 0 | 1144.0 | 784.0 |
| test_PBX_2_V400 | 3 | 0 | 2330.0 | 810.0 |
| test_emergency_horizontal | 23 | 27 | 574.0 | 253.0 |
| test_horizontal_layers_3 | 23 | 24 | 406.0 | 270.0 |
| test_horizontal_layers_4 | 24 | 25 | 435.0 | 305.0 |
| test_inclusions_1_2 | 4 | 7 | 559.0 | 287.0 |
| test_inclusions_2_2 | 4 | 10 | 644.0 | 389.0 |
| test_inclusions_3_2 | 4 | 0 | 610.0 | 470.0 |
| test_inclusions_true_V400 | 4 | 0 | 305.0 | 499.0 |

![crack length-over-time (test_emergency_horizontal)](phys_metrics/figures/length_test_emergency_horizontal.png)

## 4. Error accumulation (teacher-forced vs autoregressive F1)

Per-frame F1 gap `TF - AR` over the rollout horizon (D-13 fidelity evidence): how far the autoregressive rollout drifts from the teacher-forced pass as the horizon grows. Computed from the REAL `per_frame_metrics_tf.csv` (D-14 teacher-forced pass, inference-only / freeze-legal) against the AR `per_frame_metrics.csv`, reusing the D-03 F1 (never the stored `f1` column, S4).

| Case | mean gap (TF-AR) | late-rollout gap (last 20%) |
|---|---:|---:|
| test_MS206_V100 | +0.7136 | +0.9942 |
| test_MS206_V1000 | +0.0508 | +0.0289 |
| test_MS206_V200 | +0.0963 | +0.0876 |
| test_MS206_V400 | +0.0751 | +0.1072 |
| test_MS210_V400 | +0.2774 | +0.4073 |
| test_MS5_V150ms_inc | +0.2126 | +0.1841 |
| test_MS5_V400ms | +0.1274 | +0.1074 |
| test_PBX1_V400_true | +0.7112 | +0.8061 |
| test_PBX_2_V400 | +0.6713 | +0.5787 |
| test_emergency_horizontal | +0.1808 | +0.2356 |
| test_horizontal_layers_3 | +0.0996 | +0.1190 |
| test_horizontal_layers_4 | +0.0833 | +0.0784 |
| test_inclusions_1_2 | +0.2439 | +0.2468 |
| test_inclusions_2_2 | +0.1767 | +0.1886 |
| test_inclusions_3_2 | +0.5031 | +0.3959 |
| test_inclusions_true_V400 | +0.6385 | +0.5607 |

![TF vs AR F1 over horizon (test_emergency_horizontal)](phys_metrics/figures/f1_horizon_test_emergency_horizontal.png)

## 5. Calibration (reliability + ECE)

Reliability of the raw predicted probabilities over the rollout, binned by PREDICTED PROBABILITY (never a binarized mask, Pitfall 5). ECE per case + an aggregate reliability diagram over all cases.

| Case | ECE |
|---|---:|
| test_MS206_V100 | 0.0012 |
| test_MS206_V1000 | 0.0039 |
| test_MS206_V200 | 0.0009 |
| test_MS206_V400 | 0.0034 |
| test_MS210_V400 | 0.0191 |
| test_MS5_V150ms_inc | 0.0004 |
| test_MS5_V400ms | 0.0087 |
| test_PBX1_V400_true | 0.1496 |
| test_PBX_2_V400 | 0.2807 |
| test_emergency_horizontal | 0.0155 |
| test_horizontal_layers_3 | 0.0078 |
| test_horizontal_layers_4 | 0.0056 |
| test_inclusions_1_2 | 0.0235 |
| test_inclusions_2_2 | 0.0297 |
| test_inclusions_3_2 | 0.1188 |
| test_inclusions_true_V400 | 0.0848 |

Aggregate ECE over 16 cases: **0.0435**.

![reliability diagram](phys_metrics/figures/reliability.png)

## 6. Rollout stability

Per-case foreground-F1 curves resampled onto a common normalized rollout fraction [0,1] grid; band = median + IQR (25-75%) across cases (reused from `dashboard.plots.stability_band`, S4).

![rollout stability band](phys_metrics/figures/stability.png)

## 7. Compute efficiency (params / MACs / FLOPs / A100-h)

FractureTAU compute, introspected from the frozen `best.pt`. MACs via fvcore/ptflops (manual conv/linear fallback if absent); **FLOPs = 2 x MACs** (Pitfall 4 -- report tool + convention). The frozen ConvLSTM reference is profiled in its OWN TensorFlow env and tabulated separately (cross-framework FLOP counters are not directly comparable).

| Model | params | MACs (1 fwd) | FLOPs (1 fwd) | A100-hours |
|---|---:|---:|---:|---:|
| FractureTAU (PyTorch) | 10,323,201 | 1.155e+11 | 2.310e+11 | 3.826 |
| ConvLSTM (frozen ref) | _tabulated in TF env_ | _tabulated_ | _tabulated_ | _tabulated_ |

