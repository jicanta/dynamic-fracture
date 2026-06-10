# new_model — FractureTAU: SOTA dynamic fracture prediction

A PyTorch replacement for the ConvLSTM workflow in `ALL_INPUTS/CODE`. Same
data, same task (predict the next fracture mask from the last T frames of
physics fields), same autoregressive evaluation protocol — but a modern
architecture and training recipe designed to score higher on exactly the
metrics the project already reports (per-frame precision / recall / **F1** /
accuracy / BCE over a continuous rollout).

**None of the original code is touched.** Everything lives in this folder.

> **Just want to run it?** See **`RUN.md`** — the step-by-step operational
> guide (setup, pilot run, overnight training on Gilbreth, reading results).

---

## 1. Why this should beat the ConvLSTM

| | Old ConvLSTM | FractureTAU (this folder) |
|---|---|---|
| Architecture | 7 stacked `ConvLSTM2D(128)` layers | **SimVP encoder/decoder + TAU translator** (Temporal Attention Unit, CVPR 2023) — the current SOTA family on video-prediction benchmarks (OpenSTL). Large-kernel spatial attention + dynamic channel attention instead of recurrent gates: better long-range crack propagation, fully parallel over time (much faster per step) |
| Loss | plain BCE | **BCE + soft Dice** (+ optional focal). Fracture pixels are a tiny fraction of the 321×161 grid; Dice directly optimizes overlap, which is a surrogate of the F1 you report |
| Train/test mismatch | trained teacher-forced, tested autoregressively → errors compound in rollouts | **Stage-2 autoregressive fine-tuning**: the model is rolled out feeding back its own predictions (straight-through binarized, exactly like the eval loop) and trained on those rollouts. This directly attacks the long-rollout F1 decay |
| Model selection | best `val_loss` (teacher-forced) | best **validation AR-rollout F1** — selected on the metric that matters at test time |
| Recipe | Adam, fixed LR + plateau decay | AdamW + cosine schedule + warmup, **EMA weights**, mixed precision (bf16 on A100/A30), grad clipping |
| Data pipeline | re-parses CSVs through `tf.py_function` every epoch | one-time preprocessing into memory-mapped `.npy` per run → GPU stays busy |

Everything that defined the problem is kept identical for comparability:
grid scattering with collision averaging, T=10 / shift=1 sliding windows,
`ux,uy / 1e4`, velocity channel from the folder name (`/1000`), x/y coordinate
channels, mask binarized at 0.5, first CSV of each run dropped, no-healing
rollout with ground-truth exogenous channels, and the same
`per_frame_metrics.csv` columns.

Input channels: `[fracture_mask, ux, uy, Gc, (extra), velocity, x_norm, y_norm]`
where `extra` is selected with `--extra none|pressure|vonmises|SED` — the same
four modes (BASE / PRESSURE / VON / SED) as the old workflow.

## 2. Folder layout

```
new_model/
├── README.md                ← you are here
├── requirements.txt
├── DATASET/                 ← load the data here (see DATASET/README.md)
├── OUTPUTS/                 ← standardized results (see OUTPUTS/README.md)
├── scripts/
│   ├── train.sbatch         ← submit training on Gilbreth (auto-evaluates at the end)
│   └── evaluate.sbatch      ← evaluate an existing checkpoint
└── src/
    ├── config.py            ← all knobs + the TEST_CASE_FOLDERS mapping
    ├── data.py              ← CSV → grid cache → datasets
    ├── preprocess.py        ← optional standalone preprocessing CLI
    ├── model.py             ← FractureTAU (SimVP + TAU translator)
    ├── losses.py            ← BCE + Dice (+ focal), F1 helper
    ├── train.py             ← two-stage training, resume support
    ├── evaluate.py          ← continuous AR rollout on all test cases
    └── utils.py             ← EMA, schedulers, logging, plots
```

## 3. Setup on Gilbreth (one time)

```bash
ssh <username>@gilbreth.rcac.purdue.edu
cd /path/to/dynamic-fracture/new_model

module load anaconda
conda create -n fracture-sota python=3.11 -y
conda activate fracture-sota
pip install -r requirements.txt        # pulls CUDA-enabled torch wheels
```

Then load the dataset into `DATASET/` (copy or symlink from scratch —
`DATASET/README.md` shows the exact expected layout; it is the same as the
old `DATA_ROOT`).

## 4. Training

### Batch job (recommended)

```bash
cd new_model
sbatch scripts/train.sbatch                          # base mode
RUN_NAME=tau_sed EXTRA=SED sbatch scripts/train.sbatch   # SED mode, custom name
```

Notes:

- The script uses `-A standby` (4 h limit). Training auto-resumes from the
  last checkpoint, so if the job times out, **just resubmit the same command**
  and it continues. With your lab account (check `slist`), raise
  `#SBATCH --time` and finish in one job.
- Training ends by running the full evaluation automatically, so one job
  produces a complete `OUTPUTS/<run_name>/` with `RESULTS.md`.
- Monitor: `squeue -u $USER`, `tail -f slurm-train-<jobid>.out`.

### Interactive (debugging)

```bash
sinteractive -A standby --gpus-per-node=1 --cpus-per-task=8 --mem=32G -t 2:00:00
module load anaconda && conda activate fracture-sota
cd new_model
python -m src.train --run-name debug --epochs-stage1 2 --epochs-stage2 1
```

### Useful flags (full list: `python -m src.train --help`)

```
--extra none|pressure|vonmises|SED   feature mode (default none = BASE)
--run-name NAME                      output folder name under OUTPUTS/
--epochs-stage1 60 --epochs-stage2 15
--rollout-steps 4                    AR steps in stage-2 fine-tuning
--batch-size 4                       raise to 8 on an 80 GB A100
--hid-s 64 --hid-t 384 --n-temporal 6   model capacity
--resume auto                        continue from OUTPUTS/<run>/checkpoints/last.pt
```

Preprocessing (CSV → cache) happens automatically on the first run; to do it
ahead of time on a CPU node: `python -m src.preprocess --data-root DATASET`.

## 5. Evaluation

Automatic after training, or standalone:

```bash
sbatch scripts/evaluate.sbatch                        # all test cases found
RUN_NAME=tau_sed sbatch scripts/evaluate.sbatch
CASES="test_MS206_V400 test_inclusions_1_2" sbatch scripts/evaluate.sbatch
```

The protocol mirrors `predict_full_simulation_continuous_ar_with_metrics`:
initialize with the first T true frames, then roll forward feeding back the
predicted fracture mask (binary, no-healing) while taking all other channels
from ground truth, scoring every predicted frame against the true mask.

Missing test-case folders in `DATASET/` are skipped, so you can evaluate with
whatever subset of the test data you have loaded.

## 6. Reading the results

Open **`OUTPUTS/<run_name>/RESULTS.md`** — one table: precision, recall, F1,
accuracy, mean BCE per test case plus the micro-averaged total. Details,
per-frame CSVs, mask PNGs, F1/BCE-over-time plots and GT|prediction snapshots
are organized per case under `OUTPUTS/<run_name>/eval/` — the full layout is
documented in `OUTPUTS/README.md`.

To compare against the ConvLSTM: the `per_frame_metrics.csv` columns and the
total-metric definitions are identical to the old pipeline's outputs in
`ALL_OUTPUTS/METRICS`, so existing comparison/plotting scripts apply directly.

## 7. Suggested experiment order

1. `RUN_NAME=tau_base EXTRA=none` — direct comparison with the old BASE model.
2. `RUN_NAME=tau_sed EXTRA=SED` (and `pressure`, `vonmises`) — same ablation
   you ran with the ConvLSTM.
3. If long-rollout F1 still decays: increase `--rollout-steps` to 6–8 and/or
   `--epochs-stage2` to 25.
4. If underfitting (train ≈ val loss, both high): `--hid-s 96 --hid-t 512
   --n-temporal 8`.

## 8. Sanity smoke test (any machine, CPU is fine)

```bash
python -m src.train --data-root DATASET --run-name smoke \
    --epochs-stage1 1 --epochs-stage2 1 --rollout-steps 2 \
    --batch-size 2 --num-workers 0 --hid-s 16 --hid-t 32 --n-temporal 2
python -m src.evaluate --data-root DATASET --run-name smoke
```

This was verified end-to-end (preprocess → train both stages → resume →
evaluate → reports) on synthetic data shaped exactly like the real CSVs.
