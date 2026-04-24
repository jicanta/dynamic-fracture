# source_disp_frac/import_datasets.py

from __future__ import annotations

import os
import re
from typing import Dict, List, Sequence, Optional

import numpy as np
import pandas as pd
import tensorflow as tf


def csv_dataset_from_directory(
    directory,
    *,
    pattern="**/*.csv",
    batch_size=5,
    shuffle=False,
    # IMPORTANT: downstream expects "SED" as the canonical column name.
    # Files may contain either "SED" or legacy "a_pos"; we normalize to "SED".
    feature_cols=("fracture_mask", "x", "y", "ux", "uy", "Gc", "pressure", "vonmises", "SED"),
    target_col="fracture_mask",
    target_cols: Optional[Sequence[str]] = None,
    # --- autoregressive settings ---
    sequence_length=10,
    shift=1,
    # --- extra scalar channel ---
    add_velocity=True,
    velocity_scale=1.0 / 1000.0,
    ignore_bad_files=False,
    # --- cleanup + diagnostics ---
    drop_first_csv=True,
    print_run_stats=True,
    stats_max_files=None,
    dataset_name="dataset",
    # --- mask cleaning ---
    clip_mask_to_01=True,
    binarize_mask=False,
    mask_threshold=0.5,
    # --- velocity detection print ---
    print_detected_velocities=True,
    max_examples_detected=5,
):
    """
    Returns tf.data.Dataset yielding:
      X: (B, T, N, Cin)
      y: (B, T, N, Cy)

    X is frames [t..t+T-1] built from `feature_cols`
    plus optional velocity channel.

    y is frames [t+shift..t+shift+T-1] built from `target_cols`.

    Backward compatibility:
      If target_cols is None, then target_cols = (target_col,).

    For fracture + displacement prediction:
      target_cols=("fracture_mask", "ux", "uy")

    Then:
      y[..., 0] = shifted fracture_mask
      y[..., 1] = shifted ux
      y[..., 2] = shifted uy

    """

    directory = str(directory)

    if target_cols is None:
        target_cols = (target_col,)
    else:
        target_cols = tuple(target_cols)

    # -----------------------------
    # 1) Find CSV files
    # -----------------------------
    glob_pat = f"{directory}/{pattern}"
    all_paths = sorted(tf.io.gfile.glob(glob_pat))

    if len(all_paths) == 0:
        raise ValueError(f"No CSV files found under: {glob_pat}")

    print(f"[{dataset_name}] Found {len(all_paths)} CSV files under {directory}")

    # -----------------------------
    # 2) Helpers
    # -----------------------------
    _V_PAT = re.compile(r"(?:^|_)V(\d+(?:\.\d+)?)(?:_|$)")

    def _standardize_sed_inplace(df: pd.DataFrame) -> pd.DataFrame:
        """
        Accept either 'SED' or 'a_pos' in the CSV, but ensure downstream code
        always sees a column named 'SED'.
        """
        if "SED" in df.columns:
            return df

        if "a_pos" in df.columns:
            df = df.copy()
            df["SED"] = df["a_pos"]

        return df

    def _parse_velocity_from_run(run_name: str) -> float:
        """
        Extract velocity from run folder names like:
          F_MS201_V100_out -> 100
        then applies velocity_scale.
        """
        m = _V_PAT.search(run_name)

        if not m:
            raise ValueError(f"Cannot parse velocity from run folder name: {run_name}")

        return float(m.group(1)) * velocity_scale

    def _parse_frame_idx(path: str, run_folder: str) -> int:
        """
        Extract frame index from filename.

        Supports:
          <run_folder>_t0000_0.00ns.csv  -> 0
          <run_folder>_t0123.csv         -> 123
          <run_folder>_0007.csv          -> 7
          anything_0042.csv              -> 42
        """
        base = os.path.splitext(os.path.basename(path))[0]

        rest = base
        if base.startswith(run_folder):
            rest = base[len(run_folder):]
            if rest.startswith("_"):
                rest = rest[1:]

        # Preferred: parse `_t####`
        m = re.search(r"(?:^|_)t(\d+)(?:_|$)", rest)
        if m:
            return int(m.group(1))

        # Legacy: ends with `_####`
        m = re.search(r"(?:^|_)(\d+)$", rest)
        if m:
            return int(m.group(1))

        # Last resort: look in full basename
        m = re.search(r"(?:^|_)t(\d+)(?:_|$)", base)
        if m:
            return int(m.group(1))

        raise ValueError(f"Cannot parse frame index from: {path}")

    # -----------------------------
    # 3) Group by run folder
    # -----------------------------
    runs: Dict[str, List[str]] = {}

    for p in all_paths:
        run_folder = os.path.basename(os.path.dirname(p))
        runs.setdefault(run_folder, []).append(p)

    # -----------------------------
    # 3b) Print detected velocities
    # -----------------------------
    if add_velocity and print_detected_velocities:
        detected: Dict[str, float] = {}
        failed: List[str] = []

        for run_folder in sorted(runs.keys()):
            try:
                detected[run_folder] = _parse_velocity_from_run(run_folder)
            except ValueError:
                failed.append(run_folder)

        unique_vels = sorted(set(detected.values()))

        print(
            f"[{dataset_name}] Detected velocities (scaled) in '{directory}': {unique_vels} "
            f"(count={len(unique_vels)})"
        )

        for rf, vv in list(sorted(detected.items()))[: int(max_examples_detected)]:
            print(f"[{dataset_name}]   {rf} -> {vv}")

        if failed:
            print(
                f"[{dataset_name}] WARNING: {len(failed)} run folder(s) contained CSVs but did not match "
                f"velocity pattern '*_V###_*'. Examples: {failed[:3]}"
            )

    # -----------------------------
    # 4) Infer num_points + validate columns
    # -----------------------------
    num_points = None
    first_good = None
    first_df = None

    for p in all_paths:
        try:
            tmp = pd.read_csv(p)
            tmp = _standardize_sed_inplace(tmp)
            num_points = len(tmp)
            first_good = p
            first_df = tmp
            break
        except Exception:
            if not ignore_bad_files:
                raise

    if num_points is None:
        raise ValueError("Could not read any CSVs to infer num_points.")

    required_cols = set(feature_cols) | set(target_cols)
    missing = sorted(required_cols - set(first_df.columns))

    if missing:
        raise ValueError(
            f"[{dataset_name}] Missing required columns in CSV.\n"
            f"  Missing: {missing}\n"
            f"  Example file: {first_good}\n"
            f"  Present columns: {list(first_df.columns)}\n"
            f"Note: time_ns is intentionally ignored and not required.\n"
            f"Also: for SED we accept either 'SED' or 'a_pos' as legacy input."
        )

    base_num_features = len(feature_cols)
    Cin = base_num_features + (1 if add_velocity else 0)
    Cy = len(target_cols)

    feature_names = list(feature_cols)
    if add_velocity:
        feature_names.append("velocity")

    target_names = list(target_cols)

    print(
        f"[{dataset_name}] NUM_POINTS: {num_points}  "
        f"BASE_FEATURES: {base_num_features}  Cin: {Cin}  Cy: {Cy}"
    )
    print(f"[{dataset_name}] Feature columns: {list(feature_cols)}")
    print(f"[{dataset_name}] Target columns:  {target_names}")
    print(f"[{dataset_name}] First readable CSV: {first_good}")

    # -----------------------------
    # 5) Cleaning helpers
    # -----------------------------
    def _clean_mask(arr: np.ndarray) -> np.ndarray:
        if clip_mask_to_01:
            arr = np.clip(arr, 0.0, 1.0)

        if binarize_mask:
            arr = (arr > mask_threshold).astype(np.float32)

        return arr

    def _clean_target_array(tgt: np.ndarray, cols: Sequence[str]) -> np.ndarray:
        """
        Clean only fracture-like target columns.
        Continuous targets such as ux and uy are left continuous.
        """
        tgt = np.asarray(tgt, dtype=np.float32).copy()

        for j, col in enumerate(cols):
            if col == "fracture_mask":
                tgt[:, j:j + 1] = _clean_mask(tgt[:, j:j + 1])

        return tgt

    # -----------------------------
    # 6) Optional stats
    # -----------------------------
    def _init_stats(nc: int) -> dict:
        return {
            "min": np.full((nc,), np.inf, dtype=np.float64),
            "max": np.full((nc,), -np.inf, dtype=np.float64),
            "sum": np.zeros((nc,), dtype=np.float64),
            "count": 0,
        }

    def _update_stats(stats: dict, arr2d: np.ndarray) -> None:
        arr2d = np.asarray(arr2d, dtype=np.float64)

        stats["min"] = np.minimum(stats["min"], arr2d.min(axis=0))
        stats["max"] = np.maximum(stats["max"], arr2d.max(axis=0))
        stats["sum"] += arr2d.sum(axis=0)
        stats["count"] += arr2d.shape[0]

    def _print_stats_table(
        run_folder: str,
        stats: dict,
        names: Sequence[str],
        prefix: str = "",
    ) -> None:
        means = stats["sum"] / max(stats["count"], 1)

        print(prefix + f"Run {run_folder} stats (min / max / mean):")
        for i, nm in enumerate(names):
            print(
                prefix
                + f"  {nm:>15}: "
                + f"{stats['min'][i]: .4e}  "
                + f"{stats['max'][i]: .4e}  "
                + f"{means[i]: .4e}"
            )

    if print_run_stats:
        print(
            f"[{dataset_name}] Computing per-run stats... "
            f"(drop_first_csv={drop_first_csv}, stats_max_files={stats_max_files})"
        )

        for run_folder, paths in sorted(runs.items()):
            paths_sorted_full = sorted(paths, key=lambda p: _parse_frame_idx(p, run_folder))

            if drop_first_csv and len(paths_sorted_full) > 0:
                dropped_path = paths_sorted_full[0]
                paths_sorted = paths_sorted_full[1:]
                print(
                    f"[{dataset_name}] {run_folder}: dropped 1 file "
                    f"(dropped={os.path.basename(dropped_path)})"
                )
            else:
                paths_sorted = paths_sorted_full

            if len(paths_sorted) == 0:
                continue

            scan_paths = paths_sorted if stats_max_files is None else paths_sorted[: int(stats_max_files)]

            v = _parse_velocity_from_run(run_folder) if add_velocity else None

            Xstats = _init_stats(Cin)
            Ystats = _init_stats(Cy)

            ok = 0

            for p in scan_paths:
                try:
                    df = pd.read_csv(p)
                    df = _standardize_sed_inplace(df)

                    if len(df) != num_points:
                        raise ValueError(f"{p} has {len(df)} rows, expected {num_points}")

                    feats = df[list(feature_cols)].to_numpy(dtype=np.float32)

                    # Clean fracture_mask channel inside X if present.
                    if "fracture_mask" in feature_cols:
                        j = list(feature_cols).index("fracture_mask")
                        feats[:, j:j + 1] = _clean_mask(feats[:, j:j + 1])

                    if add_velocity:
                        vcol = np.full((num_points, 1), v, dtype=np.float32)
                        feats = np.concatenate([feats, vcol], axis=1)

                    _update_stats(Xstats, feats)

                    tgt = df[list(target_cols)].to_numpy(dtype=np.float32)
                    tgt = _clean_target_array(tgt, target_cols)
                    _update_stats(Ystats, tgt)

                    ok += 1

                except Exception:
                    if ignore_bad_files:
                        continue
                    raise

            if ok:
                prefix = f"[{dataset_name}] "
                _print_stats_table(run_folder, Xstats, feature_names, prefix=prefix)
                _print_stats_table(run_folder, Ystats, target_names, prefix=prefix)

        print(f"[{dataset_name}] Done computing per-run stats.\n")

    # -----------------------------
    # 7) Build sliding windows
    # -----------------------------
    in_windows: List[List[str]] = []
    out_windows: List[List[str]] = []
    vels: List[float] = []

    T = int(sequence_length)
    S = int(shift)

    for run_folder, paths in runs.items():
        paths_sorted_full = sorted(paths, key=lambda p: _parse_frame_idx(p, run_folder))

        if drop_first_csv and len(paths_sorted_full) > 0:
            paths_sorted = paths_sorted_full[1:]
        else:
            paths_sorted = paths_sorted_full

        if len(paths_sorted) < T + S:
            continue

        v = _parse_velocity_from_run(run_folder) if add_velocity else 0.0

        for start in range(0, len(paths_sorted) - (T + S) + 1):
            in_windows.append(paths_sorted[start : start + T])
            out_windows.append(paths_sorted[start + S : start + S + T])
            vels.append(v)

    if len(in_windows) == 0:
        raise ValueError("No valid sliding windows produced. Check folder structure and sequence_length/shift.")

    in_windows = np.asarray(in_windows, dtype=np.str_)
    out_windows = np.asarray(out_windows, dtype=np.str_)
    vels = np.asarray(vels, dtype=np.float32)

    print(f"[{dataset_name}] Built {len(in_windows)} sliding-window samples (T={T}, shift={S})")

    # -----------------------------
    # 8) Per-window reader
    # -----------------------------
    feature_cols_list = list(feature_cols)
    target_cols_list = list(target_cols)

    x_mask_idx = feature_cols_list.index("fracture_mask") if "fracture_mask" in feature_cols_list else None

    def _read_window_py(in_paths_bytes, out_paths_bytes, vel_scalar):
        in_paths = [p.decode("utf-8") for p in in_paths_bytes.tolist()]
        out_paths = [p.decode("utf-8") for p in out_paths_bytes.tolist()]
        v = float(vel_scalar)

        X_seq, y_seq = [], []

        # Input window: frames t ... t+T-1
        for p in in_paths:
            df = pd.read_csv(p)
            df = _standardize_sed_inplace(df)

            if len(df) != num_points:
                raise ValueError(f"{p} has {len(df)} rows, expected {num_points}")

            feats = df[feature_cols_list].to_numpy(dtype=np.float32)

            # Clean fracture_mask channel inside X if present.
            if x_mask_idx is not None:
                feats[:, x_mask_idx : x_mask_idx + 1] = _clean_mask(
                    feats[:, x_mask_idx : x_mask_idx + 1]
                )

            if add_velocity:
                vcol = np.full((num_points, 1), v, dtype=np.float32)
                feats = np.concatenate([feats, vcol], axis=1)

            X_seq.append(feats)

        # Output window: frames t+shift ... t+shift+T-1
        # Multi-channel targets, e.g. [fracture_mask, ux, uy].
        for p in out_paths:
            df = pd.read_csv(p)
            df = _standardize_sed_inplace(df)

            if len(df) != num_points:
                raise ValueError(f"{p} has {len(df)} rows, expected {num_points}")

            tgt = df[target_cols_list].to_numpy(dtype=np.float32)
            tgt = _clean_target_array(tgt, target_cols_list)

            y_seq.append(tgt)

        X_seq = np.stack(X_seq, axis=0)  # (T,N,Cin)
        y_seq = np.stack(y_seq, axis=0)  # (T,N,Cy)

        return X_seq, y_seq

    def _read_window_tf(in_paths, out_paths, vel):
        X, y = tf.py_function(
            func=lambda ip, op, vv: _read_window_py(ip.numpy(), op.numpy(), vv.numpy()),
            inp=[in_paths, out_paths, vel],
            Tout=[tf.float32, tf.float32],
        )

        X.set_shape([T, num_points, Cin])
        y.set_shape([T, num_points, Cy])

        return X, y

    # -----------------------------
    # 9) Build tf.data dataset
    # -----------------------------
    ds = tf.data.Dataset.from_tensor_slices((in_windows, out_windows, vels))

    if shuffle:
        ds = ds.shuffle(min(len(in_windows), 2000), reshuffle_each_iteration=True)

    ds = ds.map(_read_window_tf, num_parallel_calls=tf.data.AUTOTUNE)

    if ignore_bad_files:
        ds = ds.apply(tf.data.experimental.ignore_errors())

    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    return ds