# src/data.py
"""CSV -> grid cache -> PyTorch datasets.

The raw data layout is identical to the original ConvLSTM workflow:

    DATASET/
      trainDS/<run_folder>/<run>_t0000_0.00ns.csv ...
      F_MS206_V400_out/<run_folder>/*.csv          (one folder per test case)

Each CSV holds one time frame as a point cloud on a regular grid with columns
  x, y, ux, uy, Gc, fracture_mask  (+ optionally pressure / vonmises / SED|a_pos)

Preprocessing scatters every frame onto the global (H, W) grid exactly like
source/mapping.py (searchsorted indices, collision averaging) and stores one
float32 .npy per run so training never touches CSVs again:

    <cache>/grid.json
    <cache>/<top_folder>/<run_folder>/features.npy   (n_frames, C_all, H, W)
    <cache>/<top_folder>/<run_folder>/meta.json      channels, velocity, ...

Model input channel layout (built on the fly by the datasets):
    [0] fracture_mask (binary)   [1] ux/1e4   [2] uy/1e4   [3] Gc
    [4] extra (optional)         [.] velocity [.] x_norm   [.] y_norm
Channel 0 is the autoregressive channel; everything else is exogenous.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

BASE_CHANNELS = ["fracture_mask", "ux", "uy", "Gc"]
OPTIONAL_CHANNELS = ["pressure", "vonmises", "SED"]

_V_PAT = re.compile(r"(?:^|_)V(\d+(?:\.\d+)?)(?:_|$)")
_V_PAT_LENIENT = re.compile(r"[Vv](\d+(?:\.\d+)?)")


# ----------------------------------------------------------------------
# Filename / folder parsing (same conventions as import_datasets.py)
# ----------------------------------------------------------------------
def parse_velocity(run_name: str) -> float:
    m = _V_PAT.search(run_name)
    if m is None:
        m = _V_PAT_LENIENT.search(run_name)
        if m is None:
            raise ValueError(f"Cannot parse velocity from run folder name: {run_name}")
        print(f"[data] WARNING: lenient velocity parse for '{run_name}' -> {m.group(1)}")
    return float(m.group(1))


def parse_frame_idx(path: str, run_folder: str) -> int:
    base = os.path.splitext(os.path.basename(path))[0]
    rest = base
    if base.startswith(run_folder):
        rest = base[len(run_folder):].lstrip("_")
    m = re.search(r"(?:^|_)t(\d+)(?:_|$)", rest)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:^|_)(\d+)$", rest)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:^|_)t(\d+)(?:_|$)", base)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot parse frame index from: {path}")


def discover_runs(top_dir: Path) -> Dict[str, List[Path]]:
    """Map run_folder_name -> sorted list of csv paths, dropping the first frame
    of each run if requested by the caller (handled in preprocess_run)."""
    runs: Dict[str, List[Path]] = {}
    for p in sorted(top_dir.rglob("*.csv")):
        # skip hidden dirs (e.g. Jupyter's .ipynb_checkpoints copies of CSVs)
        if any(part.startswith(".") for part in p.relative_to(top_dir).parts):
            continue
        runs.setdefault(p.parent.name, []).append(p)
    return runs


# ----------------------------------------------------------------------
# Grid
# ----------------------------------------------------------------------
def infer_grid_from_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    import pandas as pd
    df = pd.read_csv(csv_path)
    x_vals = np.sort(np.unique(df["x"].to_numpy(np.float32)))
    y_vals = np.sort(np.unique(df["y"].to_numpy(np.float32)))
    if len(x_vals) <= 1 or len(y_vals) <= 1:
        raise ValueError(f"Degenerate grid inferred from {csv_path}")
    return x_vals, y_vals


def load_grid(cache_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    with open(cache_dir / "grid.json") as f:
        g = json.load(f)
    return np.asarray(g["x_vals"], np.float32), np.asarray(g["y_vals"], np.float32)


# ----------------------------------------------------------------------
# Scattering (numpy port of mapping.py: searchsorted + collision averaging)
# ----------------------------------------------------------------------
def scatter_frame(df, channels: Sequence[str],
                  x_vals: np.ndarray, y_vals: np.ndarray) -> np.ndarray:
    H, W = len(y_vals), len(x_vals)
    x_idx = np.clip(np.searchsorted(x_vals, df["x"].to_numpy(np.float32), side="left"), 0, W - 1)
    y_idx = np.clip(np.searchsorted(y_vals, df["y"].to_numpy(np.float32), side="left"), 0, H - 1)
    flat = y_idx * W + x_idx
    counts = np.bincount(flat, minlength=H * W).astype(np.float32)
    counts_safe = np.maximum(counts, 1.0)

    out = np.empty((len(channels), H, W), dtype=np.float32)
    for ci, name in enumerate(channels):
        vals = df[name].to_numpy(np.float32)
        sums = np.bincount(flat, weights=vals, minlength=H * W).astype(np.float32)
        out[ci] = (sums / counts_safe).reshape(H, W)
    np.nan_to_num(out, copy=False)
    return out


# ----------------------------------------------------------------------
# Preprocessing: CSV runs -> cached .npy
# ----------------------------------------------------------------------
def preprocess_run(run_folder: str, csv_paths: List[Path], out_dir: Path,
                   x_vals: np.ndarray, y_vals: np.ndarray,
                   drop_first_csv: bool = True) -> None:
    import pandas as pd

    paths = sorted(csv_paths, key=lambda p: parse_frame_idx(str(p), run_folder))
    if drop_first_csv and len(paths) > 1:
        paths = paths[1:]

    df0 = pd.read_csv(paths[0])
    if "SED" not in df0.columns and "a_pos" in df0.columns:
        pass  # handled per-frame below
    present_optional = [c for c in OPTIONAL_CHANNELS
                        if c in df0.columns or (c == "SED" and "a_pos" in df0.columns)]
    channels = BASE_CHANNELS + present_optional

    missing = [c for c in BASE_CHANNELS + ["x", "y"] if c not in df0.columns]
    if missing:
        raise ValueError(f"[{run_folder}] missing required columns {missing} in {paths[0]}")

    H, W = len(y_vals), len(x_vals)
    feats = np.empty((len(paths), len(channels), H, W), dtype=np.float32)
    for i, p in enumerate(paths):
        df = pd.read_csv(p)
        if "SED" in channels and "SED" not in df.columns and "a_pos" in df.columns:
            df = df.rename(columns={"a_pos": "SED"})
        feats[i] = scatter_frame(df, channels, x_vals, y_vals)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "features.npy", feats)
    meta = {
        "run_folder": run_folder,
        "channels": channels,
        "velocity": parse_velocity(run_folder),
        "n_frames": int(len(paths)),
        "H": int(H),
        "W": int(W),
        "first_csv": str(paths[0]),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[cache] {run_folder}: {len(paths)} frames, channels={channels}")


def ensure_cache(data_root: Path, cache_dir: Path, top_folders: Sequence[str],
                 drop_first_csv: bool = True) -> None:
    """Preprocess every run under each top folder that is not cached yet.
    The grid is inferred once from the first training CSV and reused everywhere
    (same behavior as map_existing_datasets with grid_source_key=trainDS)."""
    data_root, cache_dir = Path(data_root), Path(cache_dir)
    grid_file = cache_dir / "grid.json"

    if not grid_file.exists():
        train_dir = data_root / "trainDS"
        runs = discover_runs(train_dir) if train_dir.exists() else {}
        if not runs:
            raise FileNotFoundError(
                f"No CSVs under {train_dir} and no cached grid at {grid_file}. "
                "Load the dataset into DATASET/ first (see DATASET/README.md)."
            )
        first_run = sorted(runs.keys())[0]
        x_vals, y_vals = infer_grid_from_csv(sorted(runs[first_run])[0])
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(grid_file, "w") as f:
            json.dump({"x_vals": x_vals.tolist(), "y_vals": y_vals.tolist()}, f)
        print(f"[cache] grid: H={len(y_vals)} W={len(x_vals)} (from {first_run})")

    x_vals, y_vals = load_grid(cache_dir)

    for top in top_folders:
        top_dir = data_root / top
        if not top_dir.exists():
            print(f"[cache] skipping missing folder: {top_dir}")
            continue
        for run_folder, csvs in sorted(discover_runs(top_dir).items()):
            out_dir = cache_dir / top / run_folder
            if (out_dir / "features.npy").exists() and (out_dir / "meta.json").exists():
                continue
            preprocess_run(run_folder, csvs, out_dir, x_vals, y_vals,
                           drop_first_csv=drop_first_csv)


# ----------------------------------------------------------------------
# Cached run handle (lazy mmap, safe with DataLoader workers)
# ----------------------------------------------------------------------
class RunCache:
    def __init__(self, cache_run_dir: Path):
        self.dir = Path(cache_run_dir)
        with open(self.dir / "meta.json") as f:
            self.meta = json.load(f)
        self.channels: List[str] = self.meta["channels"]
        self.velocity: float = float(self.meta["velocity"])
        self.n_frames: int = int(self.meta["n_frames"])
        self.H: int = int(self.meta["H"])
        self.W: int = int(self.meta["W"])
        self._feats = None

    @property
    def feats(self) -> np.ndarray:
        if self._feats is None:
            self._feats = np.load(self.dir / "features.npy", mmap_mode="r")
        return self._feats

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_feats"] = None  # re-open mmap inside each worker process
        return d


def load_runs(cache_dir: Path, top_folder: str) -> List[RunCache]:
    top = Path(cache_dir) / top_folder
    if not top.exists():
        return []
    return [RunCache(d) for d in sorted(top.iterdir()) if (d / "meta.json").exists()]


# ----------------------------------------------------------------------
# Frame assembly: cached channels -> model input channels
# ----------------------------------------------------------------------
class FrameAssembler:
    """Builds the (C, H, W) model input for one frame of one run."""

    def __init__(self, *, channels_wanted: List[str], x_vals: np.ndarray,
                 y_vals: np.ndarray, uxuy_scale: float, velocity_scale: float):
        self.channels_wanted = channels_wanted          # cache channels to read
        self.uxuy_scale = float(uxuy_scale)
        self.velocity_scale = float(velocity_scale)
        H, W = len(y_vals), len(x_vals)
        xn = (x_vals - x_vals[0]) / (x_vals[-1] - x_vals[0] + 1e-12)
        yn = (y_vals - y_vals[0]) / (y_vals[-1] - y_vals[0] + 1e-12)
        self.x_norm = np.tile(xn[None, :], (H, 1)).astype(np.float32)
        self.y_norm = np.tile(yn[:, None], (1, W)).astype(np.float32)
        self.n_channels = len(channels_wanted) + 3      # + velocity + x_norm + y_norm
        self.mask_channel = 0

    def channel_names(self) -> List[str]:
        return list(self.channels_wanted) + ["velocity", "x_norm", "y_norm"]

    def select_indices(self, run: RunCache) -> List[int]:
        missing = [c for c in self.channels_wanted if c not in run.channels]
        if missing:
            raise ValueError(
                f"Run {run.meta['run_folder']} lacks channels {missing} "
                f"(has {run.channels}). Re-run preprocessing or change --extra."
            )
        return [run.channels.index(c) for c in self.channels_wanted]

    def frames(self, run: RunCache, start: int, stop: int) -> np.ndarray:
        """Return (L, C, H, W) float32 for frames [start, stop)."""
        idx = self.select_indices(run)
        raw = np.asarray(run.feats[start:stop][:, idx], dtype=np.float32).copy()
        raw[:, 0] = (raw[:, 0] > 0.5).astype(np.float32)            # binarize mask
        raw[:, 1] /= self.uxuy_scale                                 # ux
        raw[:, 2] /= self.uxuy_scale                                 # uy
        L, _, H, W = raw.shape
        vel = np.full((L, 1, H, W), run.velocity * self.velocity_scale, np.float32)
        xg = np.broadcast_to(self.x_norm, (L, 1, H, W))
        yg = np.broadcast_to(self.y_norm, (L, 1, H, W))
        return np.concatenate([raw, vel, xg, yg], axis=1)


def make_assembler(cfg, cache_dir: Path) -> FrameAssembler:
    x_vals, y_vals = load_grid(cache_dir)
    wanted = list(BASE_CHANNELS)
    if cfg.extra != "none":
        wanted.append(cfg.extra)
    return FrameAssembler(
        channels_wanted=wanted, x_vals=x_vals, y_vals=y_vals,
        uxuy_scale=cfg.uxuy_scale, velocity_scale=cfg.velocity_scale,
    )


# ----------------------------------------------------------------------
# Sliding-window dataset
# ----------------------------------------------------------------------
class WindowDataset(Dataset):
    """Yields (L, C, H, W) float32 tensors of L consecutive frames.

    L = sequence_length + rollout_steps. Targets are derived from channel 0
    (fracture mask) of later frames by the training loop, exactly as the
    original pipeline derived y from the same CSV mask column.

    Train/val split definition (D-08/D-10, frame-boundary form):
      Per run, one frame boundary ``f_cut = round(n_frames * (1 - val_fraction))``
      is chosen ONCE, independent of window_len. A window starting at ``s`` spans
      frames ``[s, s + window_len - 1]``. Train windows are those whose LAST frame
      is < f_cut; val windows are those whose FIRST frame is >= f_cut. Windows that
      straddle f_cut are dropped — that dropped band (``window_len - 1`` frames wide)
      IS the guard gap, so no train and val window can share a frame. Because the
      boundary is on frames (not on a per-dataset window count), the disjointness
      holds simultaneously for every window_len (T+1, L_train, L_val).
    """

    def __init__(self, runs: List[RunCache], assembler: FrameAssembler,
                 window_len: int, *, split: str = "all", val_fraction: float = 0.1):
        self.runs = runs
        self.assembler = assembler
        self.window_len = int(window_len)
        self.val_fraction = float(val_fraction)
        self.index: List[Tuple[int, int]] = []

        for ri, run in enumerate(runs):
            n_windows = run.n_frames - self.window_len + 1
            if n_windows <= 0:
                continue
            # ONE per-run frame boundary, window_len-independent (C-04).
            f_cut = int(round(run.n_frames * (1.0 - val_fraction)))
            for s in range(n_windows):
                last = s + self.window_len - 1
                if split == "train":
                    if last < f_cut:                 # whole window before the cut
                        self.index.append((ri, s))
                elif split == "val":
                    if s >= f_cut:                   # whole window at/after the cut
                        self.index.append((ri, s))
                else:                                # "all": no split
                    self.index.append((ri, s))

        if not self.index:
            raise ValueError(f"No windows produced (split={split}, L={self.window_len}). "
                             "Check that runs are long enough.")

    def __len__(self) -> int:
        return len(self.index)

    def window_velocities(self) -> np.ndarray:
        """Impact velocity of the run each window comes from (per window)."""
        return np.array([self.runs[ri].velocity for ri, _ in self.index],
                        dtype=np.float64)

    def __getitem__(self, i: int) -> torch.Tensor:
        ri, start = self.index[i]
        arr = self.assembler.frames(self.runs[ri], start, start + self.window_len)
        return torch.from_numpy(arr)


def assert_split_disjoint(runs: List[RunCache], val_fraction: float,
                          window_len: int) -> None:
    """Assert the frame-boundary split leaves train and val windows disjoint (D-10).

    For the given ``window_len`` builds the train and val window index sets the same
    way WindowDataset does, then per run checks that the last frame touched by any
    train window is strictly before the first frame touched by any val window.
    Raises ValueError (data.py error style) on any overlap. Runs that contribute no
    train OR no val windows at this window_len are skipped (nothing to compare).
    """
    w = int(window_len)
    for ri, run in enumerate(runs):
        n_windows = run.n_frames - w + 1
        if n_windows <= 0:
            continue
        f_cut = int(round(run.n_frames * (1.0 - val_fraction)))
        train_starts = [s for s in range(n_windows) if s + w - 1 < f_cut]
        val_starts = [s for s in range(n_windows) if s >= f_cut]
        if not train_starts or not val_starts:
            continue
        tr_last = max(s + w - 1 for s in train_starts)
        val_first = min(val_starts)
        if not (tr_last < val_first):
            raise ValueError(
                f"run {ri}: train/val frame overlap ({tr_last} >= {val_first}) "
                f"at window_len={w}, val_fraction={val_fraction}"
            )
