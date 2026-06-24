# Data split (FractureTAU vs ConvLSTM)

This document pins the **fixed, documented data split** used for every reported
number (PAPER-03). The split is deterministic (`seed=42`) and never reshuffled
between runs, so old-vs-new comparisons stay apples-to-apples.

All paths below are repository-relative; the dataset itself lives on cluster
scratch and is symlinked into `new_model/DATASET/` (not committed).

## Grid

- Grid resolution: **W=321 × H=161**, inferred once from the first training CSV
  and frozen in `DATASET/_cache/grid.json`. Both pipelines re-use the identical
  grid so masks are pixel-aligned.

## Train / validation split (DATA-02)

The split is a **per-run temporal frame-boundary split**, not a random shuffle:

- `val_fraction = 0.1` — the **last 10% of frames of each run** is the validation
  tail; the earlier 90% is training. The cut is at frame
  `f_cut = round(n_frames * (1 - val_fraction))`.
- It is a **frame-boundary** split: windows are formed *after* the cut, and any
  window that would **straddle** the train/val boundary is **dropped** (a guard
  band of width `window_len - 1`). This guarantees the train and validation
  **frame sets share no frame** — i.e. **no window leakage**.
- This disjointness is **asserted**, independently of the training code, by
  `tests/test_val_split.py` across three representative window sizes (`T+1`,
  `L_train`, `L_val`): for every run the latest training frame index is strictly
  less than the earliest validation frame index.
- Checkpoint selection (`best.pt`) uses the **validation-only** late-rollout macro
  F1 (`val_rollout_steps = 20`, Nyquist horizon). The 16 test cases below are
  **never** consulted for selection (leakage guard, ported into `rank_variants` /
  `sweep`).

## Test cases (16, held out)

The 16 test cases are entirely separate simulation runs, used only for final
autoregressive evaluation — never for training or checkpoint selection:

| # | Case (OUTPUTS key) | Dataset folder |
|---|---|---|
| 1 | test_MS206_V100 | F_MS206_V100_out |
| 2 | test_MS206_V200 | F_MS206_V200_out |
| 3 | test_MS206_V400 | F_MS206_V400_out |
| 4 | test_MS206_V1000 | F_MS206_V1000_out |
| 5 | test_MS210_V400 | F_MS210_V400_out |
| 6 | test_PBX1_V400_true | F_PBX1_V400_true |
| 7 | test_PBX_2_V400 | F_PBX_2_V400_out |
| 8 | test_MS5_V150ms_inc | MS5_V150ms_inc |
| 9 | test_MS5_V400ms | MS205_V400ms_MS5 |
| 10 | test_emergency_horizontal | F_emergency_horizontal_out |
| 11 | test_horizontal_layers_3 | F_horizontal-layers_3_out |
| 12 | test_horizontal_layers_4 | F_horizontal-layers_4_out |
| 13 | test_inclusions_1_2 | F_inclusions_1_2_out |
| 14 | test_inclusions_2_2 | F_inclusions_2_2_out |
| 15 | test_inclusions_3_2 | F_inclusions_3_2_out |
| 16 | test_inclusions_true_V400 | F_inclusions_true_V400 |

The canonical registry is `case_registry.py` (stdlib-only, imported by both the
PyTorch and TensorFlow pipelines so the list cannot silently drift; threat
T-02-01).

## Honest baseline coverage (DATA-03)

The frozen original-ConvLSTM reference outputs (`ALL_OUTPUTS/`) cover **only 8 of
the 16** test cases — a data-key↔folder mismatch surfaced and documented in
Phase 1. We report this honestly:

- The **BASE head-to-head** (the headline win) is computed on the case subset
  where **both** pipelines have outputs; the dashboard coverage matrix (D-09)
  shows the case×mode grid explicitly and marks any case as `not yet evaluated`
  rather than fabricating a number.
- No reported comparison silently averages over cases the reference never ran.

## Phase-6 reproduction scope (D-08)

Reproducing Kathleen's ConvLSTM **from scratch** is scoped to a **curated deck of
4 BASE + 2 SED** cases (favoring the overlapping, verifiable cases such as
`MS206_V*`, `inclusions_1_2`, `MS0_400_Vert`) rather than all 16 — enough to
establish a faithful, verifiable reproduction under the scope-freeze without
re-running the full matrix.

## Reproducibility stance

Numbers are reported as **mean ± std over 3 seeds (42 / 43 / 44)** with one-sided
Wilcoxon significance over the BASE cases, in a `uv.lock`-pinned environment. We
do **not** claim bit-determinism (see `README.md` → *Reproducibility honesty*).
