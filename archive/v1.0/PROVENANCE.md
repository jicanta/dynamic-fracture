# v1.0 Archive Provenance

- code_commit_sha: `07446f32ab440c795f1d1f90c379b8f1a9959993`
- dataset_sha: `unavailable: 321x161 dataset resides on Gilbreth scratch (config.json data_root='DATASET' is a scratch symlink), not locally hashable -- no in-repo grid.json/cache meta and gate_provenance.sh records no dataset sha; see PROVENANCE.md > Dataset provenance`
- generated_at: `2026-07-01T14:04:06Z`

Every quantitative claim in the v1.0 milestone traces to a measured first-party artifact bundled in this archive -- no hand-edited or remembered numbers (mirrors `paper/scripts/gate_provenance.sh`).

## Number -> source map

| number(s) | source artifact (in this archive) |
| --- | --- |
| `0.8621`, `0.5767`, `1.526e-5` | `comparisons/tau_headline_vs_base_regen.md` |
| seed mean +/- std (per case) | `outputs/seed_meanstd.csv` |
| per-mode vs-ConvLSTM tables | `comparisons/tau_{base,pressure,sed}_vs_convlstm.{csv,md}` |

## Durable checkpoints

The heavy v1.0 checkpoints are NOT copied into this gitignored repo (D-02). They live at a durable, non-scratch location on Gilbreth and are recorded by SHA256 + size in `durable_checkpoints.json` (and mirrored into `manifest.json`):

- `convlstm_base_ref.keras` -> `/depot/aamp/data/jcantare/fracture-v1.0-archive/checkpoints/convlstm_base_ref.keras` (`90e16bad89cd`)
- `tau_refined_best.pt` -> `/depot/aamp/data/jcantare/fracture-v1.0-archive/checkpoints/tau_refined_best.pt` (`f1a0786c519b`)
- `headline_s42_best.pt` -> `/depot/aamp/data/jcantare/fracture-v1.0-archive/checkpoints/headline_s42_best.pt` (`0f55814f02ae`)
- `headline_s43_best.pt` -> `/depot/aamp/data/jcantare/fracture-v1.0-archive/checkpoints/headline_s43_best.pt` (`6a09694abf51`)
- `headline_s44_best.pt` -> `/depot/aamp/data/jcantare/fracture-v1.0-archive/checkpoints/headline_s44_best.pt` (`f97f429e8d9f`)

A **missing/changed durable checkpoint = FAIL**: `python scripts/assert_provenance.py --archive` re-hashes each durable_path and exits non-zero on any missing, unreachable, or drifted checkpoint (D-08). The durable-checkpoint portion of the verify must run where the durable store is reachable (Gilbreth); off-cluster, run with `--no-checkpoints` to verify the bundled files only.

## Dataset provenance

The 321x161 simulation dataset is **not locally hashable**: it resides on Gilbreth scratch and `tau_refined/config.json` records only `data_root='DATASET'` (a scratch symlink). No in-repo `grid.json`/cache meta exists and `gate_provenance.sh` records no dataset sha. Accordingly `dataset_sha` is recorded as a documented `unavailable:` sentinel -- an honest pointer, NOT a fabricated 64-hex hash:

> unavailable: 321x161 dataset resides on Gilbreth scratch (config.json data_root='DATASET' is a scratch symlink), not locally hashable -- no in-repo grid.json/cache meta and gate_provenance.sh records no dataset sha; see PROVENANCE.md > Dataset provenance

## Verify before relying

Before trusting any archived number, verify the archive integrity:

```bash
cd dynamic-fracture
python scripts/assert_provenance.py --archive              # files + durable checkpoints
python scripts/assert_provenance.py --archive --no-checkpoints  # bundled files only (off-cluster)
```

A clean tree prints `ARCHIVE PROVENANCE OK`; any drift is a fail-loud `ARCHIVE PROVENANCE FAIL` with a non-zero exit.

