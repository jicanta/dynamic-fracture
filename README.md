# FractureTAU — dynamic-fracture

Machine-learning prediction of the spatiotemporal evolution of fracture in
microstructures. A new PyTorch model (**FractureTAU**, a SimVP encoder/decoder +
TAU translator) is trained to **beat** the original ConvLSTM reference from
Kathleen's master's thesis (TensorFlow/Keras), evaluated under an **identical
autoregressive (AR) rollout protocol** on the same simulation dataset.

> **Core result:** FractureTAU **0.6723** vs ConvLSTM **0.5767** late-rollout
> macro-F1 (3-seed mean; see *Reproducibility honesty* below for how to read this
> number). The headline win is then broadened into a multi-metric + physical
> metrics dashboard and a workshop paper.

## Two-codebase layout

This repo holds **two separate pipelines** that must never cross-import — the
shared CSV dataset is the only seam between them:

| | `new_model/` | `kathleens-model/` |
|---|---|---|
| Framework | **PyTorch** (Python 3.11) | **TensorFlow/Keras** (Python 3.12) |
| Model | FractureTAU (SimVP + TAU) | ConvLSTM (frozen reference) |
| Source | `src/` package | `source/` flat modules |
| Entry points | `python -m src.train` / `python -m src.evaluate` | `python train.py --mode N` / `python evaluate.py` |

The TensorFlow reference checkpoints are SHA256-verified and **never modified** —
they are the frozen, honest baseline.

Shared, framework-free glue (e.g. `case_registry.py`, the dashboard +
`results/phys_metrics/` analysis modules) is stdlib/numpy only so it imports under
either environment.

## Reproducible evaluation

The analysis/eval environment is **version-pinned** via `uv.lock` (PAPER-03).

```bash
cd dynamic-fracture
uv sync --group dev            # recreate the pinned env from uv.lock (incl. dev/test deps)
```

Regenerate the **headline comparison + physical metrics** report in **one
command** (PAPER-04):

```bash
bash scripts/repro_eval.sh                  # full regen for run 'tau_refined'
bash scripts/repro_eval.sh --run-name foo   # a different run
bash scripts/repro_eval.sh --skip-eval      # reuse existing eval outputs, rebuild reports only
bash scripts/repro_eval.sh --smoke          # dry-run the wiring (no GPU/artifacts) — used by CI
```

`scripts/repro_eval.sh` chains the three first-party drivers behind one entry
point:

1. `new_model/` → `python -m src.evaluate` — per-case AR (and optional
   `--teacher-forced`) rollout.
2. `results/dashboard/make_report.py` — the locked-order multi-metric dashboard
   (`results/REPORT.md` + figures).
3. `results/phys_metrics/make_phys_report.py` — the PHYS-01..06 physical-metrics
   report.

It runs end-to-end only on a machine that has the pulled run artifacts /
checkpoints; `--smoke` validates the wiring everywhere else.

### Cluster runs

Training and full evaluation run on the Slurm cluster via the existing batch
scripts — use these, do not duplicate them:

- `new_model/scripts/*.sbatch` — FractureTAU train / sweep / evaluate / seeds /
  ablation jobs.
- `kathleens-model/scripts/*.sbatch` — ConvLSTM reproduction + reference-baseline
  jobs (activate the TF env via `scripts/tf_env.sh` first).

## Reproducibility honesty

- **Results are reported as mean ± std over 3 seeds (42 / 43 / 44)**, with a
  one-sided Wilcoxon significance test over the BASE cases. Each headline number
  is recorded next to the winning checkpoint's hash — `best.pt` SHA `f1a0786c…`
  for the `tau_refined` (`ar_both`) config.
- The environment is **version-pinned via `uv.lock`**. We **do NOT claim
  bit-determinism**: GPU non-determinism, cuDNN kernel selection, and AMP make
  exact bit-for-bit reproduction unrealistic and unnecessary (Pitfall 9).
  Reproducibility means *the same pipeline + pinned env + fixed split + seeds
  reproduces the result within reported variance*, not identical bits.
- This is an **internal-only reproducible artifact** this cycle (D-07): a
  one-command eval regenerates the headline + physical metrics, with a pinned env,
  Slurm scripts, and a documented data split. **No public release** this cycle.

## Data split

The fixed train/val split, the 16 held-out test cases, the honest 8/16
old-pipeline coverage, and the Phase-6 reproduction scope are documented in
[`docs/DATA_SPLIT.md`](docs/DATA_SPLIT.md).

## Tests

```bash
cd dynamic-fracture
uv run pytest tests/        # includes tests/test_repro_smoke.py (the PAPER-04 wiring gate)
```
