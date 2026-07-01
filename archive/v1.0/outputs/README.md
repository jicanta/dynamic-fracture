# OUTPUTS — standardized results layout

Every training run gets one folder here, named after `--run-name`:

```
OUTPUTS/<run_name>/
├── config.json                     # exact config the run was trained with
├── RESULTS.md                      # ← START HERE: per-case + total metrics table
├── checkpoints/
│   ├── best.pt                     # best val AR-F1 (EMA weights) — used by evaluate
│   └── last.pt                     # latest epoch (for --resume auto)
├── logs/
│   └── train_log.csv               # epoch, stage, lr, losses, val F1 (TF + AR), time
├── curves/
│   └── training_curves.png         # loss + F1 curves, stage-2 boundary marked
└── eval/
    ├── total_metrics.csv           # one row per test case + machine-readable totals
    └── <test_case>/                # e.g. test_MS206_V400
        ├── per_frame_metrics.csv   # frame_id, tp/tn/fp/fn, precision, recall,
        │                           # f1, accuracy, bce  (same columns as the
        │                           # old per_frame_metrics.csv)
        ├── summary.json            # aggregate metrics for this case
        ├── f1_over_time.png
        ├── bce_over_time.png
        ├── frames/                 # predicted binary masks, mask_000000.png ...
        │                           # (white = fracture, flipped like the old output)
        └── viz/                    # side-by-side  GT | prediction  snapshots
```

How to judge a run quickly:

1. Open `RESULTS.md` — F1 per test case and the micro-averaged total.
2. Check `eval/<case>/f1_over_time.png` — autoregressive rollouts degrade over
   time; a good model holds F1 high deep into the simulation.
3. Compare against the ConvLSTM by diffing `per_frame_metrics.csv` files —
   the metric definitions and the rollout protocol are the same.
