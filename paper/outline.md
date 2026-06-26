# Paper Outline — Physically-Faithful Fracture Forecasting (PAPER-01)

Workshop draft outline + evidence map. Framing (D-04): **physically-faithful
fracture forecasting** — the autoregressive (AR) per-frame F1 win over the
ConvLSTM reference, *plus* a battery of physical-fidelity metrics that show the
win is not a thresholding artifact but a genuine improvement in predicted
fracture physics.

Authoring note (deviation, approved): this outline and the accompanying
`main.tex` / `refs.bib` were authored directly as a NeurIPS/ICML-style workshop
LaTeX draft following the D-11 skeleton, rather than through the ARS plugin
slash-commands (`/ars-outline`, `/ars-full`), which cannot be invoked from this
execution context. The user will run `/ars-revision` for polish.

Every results number in this map traces to a measured artifact under
`results/` or `new_model/OUTPUTS/` (correctness-over-convenience mandate). State
results as **mean±std**, never bit-determinism (bf16 AMP + cuDNN are not
byte-deterministic, so the 3-seed spread is the honest run-to-run variation).

---

## Section skeleton (D-11) + evidence map

### 1. Abstract
- One-paragraph statement of the contribution: a SimVP+TAU spatiotemporal model
  (FractureTAU) that beats a faithfully-reproduced from-scratch ConvLSTM
  reference on AR fracture-mask forecasting, with physical-fidelity evidence.
- Headline number: late-rollout (last-20%) macro F1 **0.8621** (FractureTAU,
  3-seed mean) vs **0.5767** (ConvLSTM `base_regen`), +0.285 mean delta,
  **16/16** cases won, one-sided Wilcoxon **p = 1.526e-5**.
- Evidence: `results/comparisons/tau_headline_vs_base_regen.md`,
  `new_model/OUTPUTS/seed_meanstd.csv`.

### 2. Introduction (D-12: may be rough)
- Problem: forecasting spatiotemporal fracture evolution in microstructures.
- Why AR rollout is the honest evaluation (error accumulates; teacher-forced
  numbers overstate skill).
- Contribution bullets: (i) FractureTAU architecture; (ii) head-to-head AR win
  with significance; (iii) physical-fidelity analysis (boundary localization,
  crack length/onset, TF–AR error accumulation, calibration); (iv) a faithful
  from-scratch baseline reproduction as a credibility anchor (D-08).

### 3. Related Work (brief; D-09 = DISCUSSION-only positioning)
- Video / spatiotemporal prediction architectures: ConvLSTM [shi2015convlstm],
  PredRNN [wang2017predrnn], E3D-LSTM [wang2019e3dlstm], SimVP
  [gao2022simvp], TAU [tan2023tau].
- ML for fracture / materials: [hsu2020fracture], [lew2021fracture].
- Metric provenance: BF-score [csurka2013bfscore], ECE [guo2017calibration].
- **Non-comparability caveat (D-09, Pitfall 10):** published spatiotemporal-SOTA
  numbers are NOT an experimental gate here. Different datasets, grids, physics,
  and AR protocols make cross-paper F1 incomparable. We position FractureTAU
  against these architectures conceptually and benchmark *only* against the
  in-house frozen ConvLSTM reference under an identical protocol.

### 4. Method — FractureTAU (D-12: core prose REQUIRED)
- SimVP encoder/decoder (conv stack, pad-to-multiple-of-4) + TAU temporal
  translator (intra-frame statical + inter-frame dynamical attention).
- Two-stage training: Stage 1 teacher-forced; Stage 2 AR fine-tune with
  scheduled sampling + EMA.
- Loss: BCE + soft Dice + Tversky (tversky_beta = 0.602) with growth-mask
  upweighting. Threshold calibration on val (grid-search AR F1).
- AR protocol constraints: only the fracture channel (channel 0) is fed back;
  all other channels from GT; no-healing monotonicity `state = max(state,
  pred_bin)`.
- Evidence: `.planning/PROJECT.md`, `new_model/src/{model,losses,train}.py`,
  CLAUDE.md architecture map.

### 5. Experimental Setup
- Fixed dataset: 321×161 grid, 16 held-out test cases.
- Continuous AR rollout protocol (shared by both pipelines), no-healing.
- Metric definitions: D-03 degenerate-aware macro F1; late-rollout = last 20%.
- 3 seeds (42/43/44); report mean±std.
- Evidence: CLAUDE.md data section, `new_model/scripts/significance.py`.

### 6. Results (D-12: core prose REQUIRED; D-05 rigor anchor)
- **6.1 Headline head-to-head:** FractureTAU 0.8621 vs ConvLSTM 0.5767, mean
  delta +0.285, median delta +0.2285, 16/16 cases won. Wilcoxon W=136.0,
  p=1.526e-5 (one-sided, exact, n=16). Per-case table from
  `results/comparisons/tau_headline_vs_base_regen.md`.
- **6.2 Seed aggregate:** per-case mean±std from
  `new_model/OUTPUTS/seed_meanstd.csv` (overall 0.8621).
- **6.3 Compact ablation (ONE table, key rows; D-05):** input-channel ablation
  — extra=none (BASE) 0.86 vs +von Mises 0.75 vs +SED 0.58 — motivating the
  extra=none headline. Full 5-row leave-one-out + extended significance →
  appendix. Evidence: `results/findings/inclusion_false_positives.md`,
  `results/notes/thesis-crosscheck.md`.
- **6.4 Faithful baseline reproduction (D-08 subsection):** the Phase-6
  from-scratch ConvLSTM reproduction at **honest 4 BASE / 2 SED** scope (corrupt
  MS206 SED reference exports excluded). This anchors the head-to-head
  credibility — the baseline is retrained, not just inherited. Evidence:
  `results/phys_metrics/PHYS_REPORT.md` §1, Phase-6 SUMMARYs.
- Provenance recorded: per-seed best.pt sha256 — headline_s42 `0f55814f02ae`,
  headline_s43 `6a09694abf51`, headline_s44 `f97f429e8d9f`; frozen reference
  `kathleens-model/OUTPUTS/base_regen`.

### 7. Physical-fidelity analysis (D-12: core prose REQUIRED; D-13 integrated)
- **7.1 Boundary localization (PHYS-02):** macro HD95 = 39.585 px, macro
  boundary-F1 = 0.6258. `PHYS_REPORT.md` §2.
- **7.2 Crack length-over-time + onset (PHYS-03):** skeleton-pixel length and
  time-to-onset, GT vs AR. Figure `length_test_emergency_horizontal.png`.
  `PHYS_REPORT.md` §3.
- **7.3 Error accumulation, TF vs AR (PHYS-04):** per-frame F1 gap (TF−AR) over
  horizon; quantifies AR drift. Figure
  `f1_horizon_test_emergency_horizontal.png`. `PHYS_REPORT.md` §4.
- **7.4 Calibration (PHYS-05, appendix):** aggregate ECE = 0.0435; reliability
  diagram `reliability.png`. `PHYS_REPORT.md` §5.
- **7.5 Rollout stability (D-06 main):** median+IQR foreground-F1 band over
  normalized rollout fraction. Figure `stability.png`. `PHYS_REPORT.md` §6.
- TODO (do not fabricate): qualitative FP/FN spatial panels were not generated;
  noted as a draft TODO.

### 8. Reproducibility (short)
- Pinned environment (uv.lock), Slurm on Gilbreth A100, documented train/val
  split, seed=42 base. Results reported as **mean±std**, NOT bit-determinism
  (bf16 AMP + cuDNN non-determinism). Efficiency: FractureTAU 10,323,201
  params, 2.310e11 FLOPs/fwd, 3.826 A100-h. `PHYS_REPORT.md` §7.

### 9. Conclusion
- Recap: a faithfully-benchmarked, physically-faithful AR fracture forecaster
  that significantly beats the reproduced ConvLSTM. Limitations + future work
  (qualitative panels, displacement-field prediction, broader SOTA positioning).

---

## Figure budget (D-06)
| Figure | File | Section |
|---|---|---|
| Rollout stability band | `figures/stability.png` | 7.5 (main) |
| Crack length + onset | `figures/length_test_emergency_horizontal.png` | 7.2 (main) |
| TF vs AR F1 over horizon | `figures/f1_horizon_test_emergency_horizontal.png` | 7.3 (main) |
| Reliability / ECE | `figures/reliability.png` | 7.4 (appendix) |
| Qualitative FP/FN panels | NOT GENERATED — TODO | 7 (deferred) |

## Tables
| Table | Source | Section |
|---|---|---|
| Headline head-to-head (per-case) | `results/comparisons/tau_headline_vs_base_regen.md` | 6.1 |
| Seed mean±std | `new_model/OUTPUTS/seed_meanstd.csv` | 6.2 |
| Compact input-channel ablation | `results/findings/`, `results/notes/` | 6.3 |
| Boundary HD95 / boundary-F1 | `PHYS_REPORT.md` §2 | 7.1 |
| Efficiency | `PHYS_REPORT.md` §7 | 8 |
| Full 5-row ablation + extended significance | Phase-4 dashboard | Appendix |
