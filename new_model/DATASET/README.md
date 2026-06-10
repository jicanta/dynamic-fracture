# DATASET (placeholder)

Load the data here using the **same layout as the original workflow's
`DATA_ROOT`** — you can copy or symlink the folder you already use for the
ConvLSTM model:

```
DATASET/
├── trainDS/                      # training simulations
│   ├── <run_folder_V###_...>/    # one folder per simulation run
│   │   ├── <run>_t0001_*.csv     # one CSV per time frame
│   │   └── ...
│   └── ...
├── F_MS206_V100_out/             # test cases (one folder each, same names
├── F_MS206_V200_out/             #  as TEST_CASE_FOLDERS in src/config.py)
├── F_MS206_V400_out/
├── F_MS206_V1000_out/
├── F_MS210_V400_out/
├── F_PBX1_V400_true/
├── F_PBX_2_V400_out/
├── MS5_V150ms_inc/
├── MS205_V400ms_MS5/
├── F_emergency_horizontal_out/
├── F_horizontal-layers_3_out/
├── F_horizontal-layers_4_out/
├── F_inclusions_1_2_out/
├── F_inclusions_2_2_out/
├── F_inclusions_3_2_out/
└── F_inclusions_true_V400/
```

Requirements per CSV (same as before):

- Columns: `x, y, ux, uy, Gc, fracture_mask` (+ optional `pressure`,
  `vonmises`, `SED` — legacy `a_pos` is accepted and renamed to `SED`).
- Run folder names must contain the impact velocity as `_V###_` (e.g.
  `F_MS206_V400_out`); a lenient `V###` fallback is used otherwise.
- Frame index parsed from `_t####` in the filename (legacy `_####` also works).
- The first CSV of each run is dropped (matches `drop_first_csv=True`).

Missing test-case folders are simply skipped at evaluation time, so you can
load only the cases you have.

A `_cache/` folder will appear here after the first run — that is the
preprocessed grid cache (`.npy` per run). Delete it to force re-preprocessing.

Tip on Gilbreth: keep the actual data on scratch and symlink it:

```bash
ln -s $CLUSTER_SCRATCH/fracture_data/trainDS DATASET/trainDS
ln -s $CLUSTER_SCRATCH/fracture_data/F_MS206_V400_out DATASET/F_MS206_V400_out
# ... etc
```
