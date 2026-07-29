# FractureTAU (3-seed headline) vs frozen published ConvLSTM (base_regen)
> Note: "base_regen" denotes regenerated baseline OUTPUTS (published frozen .keras run inference-only through the shared metric seam), not retrained weights.

_Computed 2026-06-26 locally from measured per-frame CSVs (D-05/D-06). Late-rollout (last-20%) macro F1, D-03 degenerate-aware. Headline = headline_s42/43/44 final tuned recipe (tversky_beta=0.602, tau translator). Regenerate: `python -m scripts.significance --tau OUTPUTS --cnn ../kathleens-model/OUTPUTS/base_regen`._

```
case                                TAU      CNN    delta
----------------------------------------------------------
test_MS206_V100                  0.9506   0.0916  +0.8590
test_MS206_V1000                 0.9674   0.7044  +0.2630
test_MS206_V200                  0.9483   0.2213  +0.7270
test_MS206_V400                  0.9569   0.6537  +0.3033
test_MS210_V400                  0.9321   0.6564  +0.2757
test_MS5_V150ms_inc              0.5971   0.3467  +0.2504
test_MS5_V400ms                  0.9305   0.7410  +0.1896
test_PBX1_V400_true              0.7240   0.3575  +0.3665
test_PBX_2_V400                  0.7841   0.6993  +0.0848
test_emergency_horizontal        0.8589   0.7199  +0.1391
test_horizontal_layers_3         0.9390   0.5864  +0.3526
test_horizontal_layers_4         0.9359   0.7693  +0.1666
test_inclusions_1_2              0.8520   0.6454  +0.2066
test_inclusions_2_2              0.9344   0.7902  +0.1442
test_inclusions_3_2              0.8234   0.7400  +0.0834
test_inclusions_true_V400        0.6584   0.5044  +0.1540
----------------------------------------------------------
OVERALL (per-case mean)          0.8621   0.5767  +0.2854
[sig] W=136.0  p(one-sided TAU>CNN)=1.526e-05  n=16  (nonzero=16)  median_delta=+0.2285
[sig] TAU provenance (sha256 best.pt per seed):
[sig]   headline_s42: 0f55814f02aec68d76fc21b181b795a5d0ec44fe0c12173448447e93d804d805
[sig]   headline_s43: 6a09694abf51408c40d04450294bce4a3f3aca65dc80aaab405dad613df7fd85
[sig]   headline_s44: f97f429e8d9f19ecb26d7b68b55e5412bc8b6a9c88d1bc96cc690a02a8270553
[sig] CNN reference (frozen baseline): /home/jicanta/Desktop/trabajo/surf-purdue-2026/dynamic-fracture/kathleens-model/OUTPUTS/base_regen
```

**Overall (per-case mean):** FractureTAU **0.8621** vs ConvLSTM **0.5767** (mean
delta +0.2854); one-sided Wilcoxon `p=1.526e-5` (= `1.526e-05`), `n=16`. These
are the per-case means of the table above; they match the overall in
`results/phys_metrics/PHYS_REPORT.md` and `new_model/OUTPUTS/seed_meanstd.csv`.
