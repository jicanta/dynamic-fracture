# RUN.md — How to run FractureTAU on Gilbreth, start to finish

This is the operational guide: every command you need, in order, with
explanations of what each step does and what can go wrong. For *what* the
model is and *why* it should beat the ConvLSTM, see `README.md`.

The short version of the whole workflow:

```
setup env  →  link data into DATASET/  →  15-min pilot run  →
overnight chained training  →  read OUTPUTS/<run>/RESULTS.md
```

---

## 0. What you need before starting

- A Gilbreth account (`ssh <user>@gilbreth.rcac.purdue.edu`).
- The simulation data (the same folders you use as `DATA_ROOT` in the old
  ConvLSTM workflow: `trainDS/` + the test-case folders), ideally already on
  Gilbreth scratch.
- This `new_model/` folder on Gilbreth (clone the repo or `scp -r` it).

Useful commands you'll see throughout:

| Command | What it tells you |
|---|---|
| `slist` | which SLURM accounts you can submit to |
| `myquota` | how full your home and scratch are |
| `squeue -u $USER` | your running/pending jobs |
| `scancel <jobid>` / `scancel -u $USER --name=frac-tau` | kill a job / all training jobs |

---

## 1. One-time setup: the conda environment (~10 min)

On a Gilbreth login node:

```bash
cd /path/to/dynamic-fracture/new_model

module load anaconda
conda create -n fracture-sota python=3.11 -y
conda activate fracture-sota
pip install -r requirements.txt
```

`pip install torch` on Linux pulls the CUDA-enabled wheels automatically —
you do not need to install CUDA yourself.

Quick check (on a login node `cuda.is_available()` prints `False`; that's
expected, login nodes have no GPU):

```bash
python -c "import torch; print(torch.__version__)"
```

**If `module load anaconda` fails:** run `module spider anaconda` and use the
exact module name it suggests. If you change it, update the `module load`
line in `scripts/train.sbatch` and `scripts/evaluate.sbatch` too.

**If your home quota is tight** (the env is ~6 GB; check `myquota`): create
the env on scratch instead and adjust the activate lines in the sbatch
scripts accordingly:

```bash
conda create --prefix $CLUSTER_SCRATCH/envs/fracture-sota python=3.11 -y
conda activate $CLUSTER_SCRATCH/envs/fracture-sota
```

---

## 2. One-time setup: the data

**Keep the data (and therefore the cache) on scratch, not in home.** The
preprocessing cache is written *inside* `DATASET/` (as `DATASET/_cache/`) and
can be tens of GB — it will blow a 25 GB home quota.

Symlink your data folders into `DATASET/`:

```bash
cd new_model
ln -s $CLUSTER_SCRATCH/<your_data>/trainDS              DATASET/trainDS
ln -s $CLUSTER_SCRATCH/<your_data>/F_MS206_V100_out     DATASET/F_MS206_V100_out
ln -s $CLUSTER_SCRATCH/<your_data>/F_MS206_V400_out     DATASET/F_MS206_V400_out
# ... one line per test-case folder you have
```

The expected folder names are the 16 test cases listed in
`DATASET/README.md` (same names as `TEST_CASE_FOLDERS` in the old
`build_datasets.py`). **You don't need all of them** — missing folders are
skipped at evaluation time. `trainDS/` is the only mandatory one.

If `new_model/` itself lives in home and home is tight, you can also point
the cache elsewhere explicitly with `--cache-dir $CLUSTER_SCRATCH/frac_cache`
on every command below.

---

## 3. The 15-minute pilot (DO NOT SKIP)

Never launch an overnight run untested. This pilot catches bad paths, missing
CSV columns, env problems, and OOM — in minutes, while you're awake.

Grab an interactive GPU node:

```bash
sinteractive -A standby --gpus-per-node=1 --cpus-per-task=8 --mem=32G -t 1:00:00
module load anaconda && conda activate fracture-sota
cd /path/to/new_model
```

Run a 1-epoch training on the real data:

```bash
python -m src.train --run-name pilot --epochs-stage1 1 --epochs-stage2 0
```

What happens, in order:

1. **Preprocessing** — every CSV run is scattered onto the grid and cached as
   one `.npy` under `DATASET/_cache/`. This happens only once, ever; later
   jobs reuse the cache. If your training set is large, this is the slowest
   part of the pilot — and time spent here is *not wasted* (the overnight
   jobs skip it).
2. It prints the detected runs, input channels, window counts and model size.
3. One training epoch runs and prints a line like
   `[epoch 000/1] stage1 train=0.412 val=0.387 F1tf=0.71 F1ar=0.63 (84s)`.

**Write down the epoch time in parentheses.** You'll use it to size the
overnight run: total time ≈ epoch_time × 75 (the default 60 stage-1 + 15
stage-2 epochs; stage-2 epochs are ~4× slower than stage-1 ones, so budget
generously).

Then clean up and exit the interactive node:

```bash
rm -rf OUTPUTS/pilot
exit
```

**If the pilot fails:**

- `No CSV files found` / `FileNotFoundError ... trainDS` → your symlinks in
  step 2 are wrong.
- `missing required columns` → that run's CSVs lack one of
  `x, y, ux, uy, Gc, fracture_mask`; the error names the file.
- `CUDA out of memory` → add `--batch-size 2` (and use the same flag
  overnight by editing `scripts/train.sbatch`).
- Quota errors → re-read step 2.

---

## 4. The overnight run

### Option A — standby queue with job chaining (works for everyone, free)

Standby jobs are limited to 4 hours, but training checkpoints every epoch and
resumes automatically (`--resume auto`), so we simply submit several jobs in
a chain — each starts when the previous one ends, for any reason:

```bash
cd new_model
bash scripts/chain_train.sh 6        # 6 jobs x 4 h = up to 24 h of compute
```

Then **log out and go to sleep**. Jobs keep running without you.

Notes:

- Pick the number of jobs from your pilot epoch time
  (e.g. 90 s/epoch × ~75 epochs ≈ 2 h ⇒ 2 jobs is plenty; 5 min/epoch ⇒
  use 6+ and expect to top it up the next day).
- When training completes, the *same job* runs the full evaluation, so the
  morning result is complete. Leftover chained jobs notice training is done,
  re-run the (idempotent) evaluation quickly and exit — harmless.
- Different feature mode / name:
  `RUN_NAME=tau_sed EXTRA=SED bash scripts/chain_train.sh 6`
  (`EXTRA` ∈ `none`, `pressure`, `vonmises`, `SED` — the BASE/PRESSURE/VON/SED
  modes of the old workflow.)

### Option B — your lab account, single job (simpler, if you have one)

Check `slist`. If you have a non-standby account, edit two lines in
`scripts/train.sbatch`:

```
#SBATCH -A <your_account>
#SBATCH --time=24:00:00
```

then a single submission does everything:

```bash
sbatch scripts/train.sbatch
```

### Monitoring (optional, from your phone/laptop before bed)

```bash
squeue -u $USER                      # R = running, PD = pending/queued
tail -f slurm-train-<jobid>.out      # live training log
```

---

## 5. The morning after: reading results

Everything lands in `OUTPUTS/<run_name>/` (default run name: `tau_base`).

```bash
cat OUTPUTS/tau_base/RESULTS.md
```

That's the headline table — precision / recall / **F1** / accuracy / mean BCE
for every test case plus the micro-averaged total, from the same continuous
autoregressive no-healing rollout protocol as the old pipeline.

Dig deeper:

| Where | What |
|---|---|
| `logs/train_log.csv` | per-epoch losses, val F1 (teacher-forced and AR), epoch times |
| `curves/training_curves.png` | loss + F1 curves; dashed line = stage-2 start |
| `eval/total_metrics.csv` | the RESULTS.md table, machine-readable |
| `eval/<case>/per_frame_metrics.csv` | per-frame metrics, same columns as the old `per_frame_metrics.csv` |
| `eval/<case>/f1_over_time.png` | **the key plot** — does F1 hold up deep into the rollout? |
| `eval/<case>/frames/` | predicted binary masks (white = fracture, same orientation as old outputs) |
| `eval/<case>/viz/` | side-by-side GT \| prediction snapshots |

Triage guide:

- **Training didn't finish** (last epoch in `train_log.csv` < total): just run
  `bash scripts/chain_train.sh 4` again — it resumes where it stopped.
- **Job died early**: `grep -i error slurm-train-*.out`. OOM → lower
  `--batch-size`; quota → step 2; everything else, read the Python traceback
  at the bottom of the log.
- **Finished but F1 is poor**: see "Suggested experiment order" in
  `README.md` §7 (more stage-2 epochs / rollout steps, larger model).

---

## 6. Re-running things independently

Evaluate an existing checkpoint without retraining (e.g. after adding more
test-case data):

```bash
sbatch scripts/evaluate.sbatch                                  # all cases found
RUN_NAME=tau_sed sbatch scripts/evaluate.sbatch                 # another run
CASES="test_MS206_V400 test_inclusions_1_2" sbatch scripts/evaluate.sbatch
```

Preprocess ahead of time (so the first GPU job doesn't spend time on CSVs):

```bash
python -m src.preprocess --data-root DATASET
```

Force re-preprocessing (e.g. the raw CSVs changed): delete the cache,
`rm -rf DATASET/_cache`, and it rebuilds on the next run.

Start a fresh training from scratch with the same name: delete or rename
`OUTPUTS/<run_name>/` (otherwise `--resume auto` continues the old run).

All flags: `python -m src.train --help`.

---

## 7. Smoke test without the real data (any machine, no GPU needed)

Verifies the whole pipeline end-to-end in ~2 minutes using generated data:

```bash
python scripts/make_synth_data.py --out /tmp/synth_dataset
python -m src.train --data-root /tmp/synth_dataset --run-name smoke \
    --epochs-stage1 1 --epochs-stage2 1 --rollout-steps 2 \
    --batch-size 2 --num-workers 0 --hid-s 16 --hid-t 32 --n-temporal 2
python -m src.evaluate --data-root /tmp/synth_dataset --run-name smoke
rm -rf OUTPUTS/smoke
```

---

## TL;DR for tonight

```bash
# on Gilbreth, once:
module load anaconda && conda create -n fracture-sota python=3.11 -y
conda activate fracture-sota && pip install -r requirements.txt
ln -s $CLUSTER_SCRATCH/<data>/trainDS DATASET/trainDS   # + test folders

# pilot (interactive GPU node, ~15 min):
sinteractive -A standby --gpus-per-node=1 --cpus-per-task=8 --mem=32G -t 1:00:00
python -m src.train --run-name pilot --epochs-stage1 1 --epochs-stage2 0
rm -rf OUTPUTS/pilot && exit

# overnight:
bash scripts/chain_train.sh 6

# morning:
cat OUTPUTS/tau_base/RESULTS.md
```
