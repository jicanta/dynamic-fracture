# source/worst_frame.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

PathLike = Union[str, Path]


def _score_fp_rate(
    df: pd.DataFrame,
    *,
    fp_rate_col: Optional[str],
    fp_col: str,
    tn_col: str,
    n_pixels_col: str,
) -> pd.Series:
    """
    Returns a Series of fp_rate for each row, using best available columns.
    """
    if fp_rate_col is not None and fp_rate_col in df.columns:
        s = pd.to_numeric(df[fp_rate_col], errors="coerce")
        return s

    if n_pixels_col in df.columns:
        n = pd.to_numeric(df[n_pixels_col], errors="coerce")
        fp = pd.to_numeric(df[fp_col], errors="coerce")
        return fp / n

    # fallback: FP/(FP+TN)
    fp = pd.to_numeric(df[fp_col], errors="coerce")
    tn = pd.to_numeric(df[tn_col], errors="coerce")
    denom = (fp + tn).replace(0, pd.NA)
    return fp / denom


def _score_gt_pos(
    df: pd.DataFrame,
    *,
    gt_pos_col: Optional[str],
    tp_col: str,
    fn_col: str,
) -> pd.Series:
    """
    Returns a Series of gt_pos for each row, using best available columns.
    """
    if gt_pos_col is not None and gt_pos_col in df.columns:
        s = pd.to_numeric(df[gt_pos_col], errors="coerce")
        # fill missing with tp+fn if needed
        tp = pd.to_numeric(df[tp_col], errors="coerce")
        fn = pd.to_numeric(df[fn_col], errors="coerce")
        return s.fillna(tp + fn)

    tp = pd.to_numeric(df[tp_col], errors="coerce")
    fn = pd.to_numeric(df[fn_col], errors="coerce")
    return tp + fn


def _print_top_rows(
    title: str,
    df_sorted: pd.DataFrame,
    *,
    top_k: int,
    frame_col: str,
    score_col: str,
    key_cols: List[str],
):
    print(f"\n{title}")
    if df_sorted.empty:
        print("  (no rows)")
        return

    cols = [c for c in ([frame_col, score_col] + key_cols) if c in df_sorted.columns]
    show = df_sorted[cols].head(top_k).copy()

    # nice formatting
    for c in show.columns:
        if pd.api.types.is_float_dtype(show[c]):
            show[c] = show[c].map(lambda x: f"{x:.6f}" if pd.notna(x) else x)

    print(show.to_string(index=False))


def top_worst_cases_from_csv(
    csv_path: PathLike,
    *,
    top_k: int = 10,

    frame_col: str = "frame_id",
    f1_col: str = "f1",

    # GT-positive detection
    gt_pos_col: Optional[str] = "gt_pos",
    tp_col: str = "tp",
    fn_col: str = "fn",
    min_gt_pos_pixels: int = 1,  # >= => Case A, < => Case B

    # Case B score (FP rate)
    fp_rate_col: Optional[str] = "fp_rate",
    fp_col: str = "fp",
    tn_col: str = "tn",
    n_pixels_col: str = "n_pixels",

    exclude_first_frame: bool = True,
    first_frame_value: int = 0,

    # tie-break behavior is irrelevant for top-k, but keep deterministic ordering
    stable_sort: bool = True,
    print_tables: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Prints and returns the TOP-K worst frames for both cases:

      Case A (GT fracture present): lowest F1 (ascending)
      Case B (GT fracture absent):  highest FP-rate (descending)

    Returns:
      {
        "case_A": df_case_A_sorted,   # includes helper columns _gt_pos and _fp_rate if computed
        "case_B": df_case_B_sorted,
      }
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {frame_col, f1_col, tp_col, fn_col, fp_col, tn_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}. Found: {list(df.columns)}")

    df = df.copy()
    df[frame_col] = pd.to_numeric(df[frame_col], errors="coerce")
    df[f1_col] = pd.to_numeric(df[f1_col], errors="coerce")
    for c in [tp_col, fn_col, fp_col, tn_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=[frame_col, f1_col, tp_col, fn_col, fp_col, tn_col])

    if exclude_first_frame:
        df = df[df[frame_col] != first_frame_value]

    if df.empty:
        raise ValueError("No valid rows left after filtering (including exclude_first_frame).")

    # helper columns
    df["_gt_pos"] = _score_gt_pos(df, gt_pos_col=gt_pos_col, tp_col=tp_col, fn_col=fn_col)
    df["_fp_rate"] = _score_fp_rate(df, fp_rate_col=fp_rate_col, fp_col=fp_col, tn_col=tn_col, n_pixels_col=n_pixels_col)

    # Split cases
    df_A = df[df["_gt_pos"] >= int(min_gt_pos_pixels)].copy()
    df_B = df[df["_gt_pos"] < int(min_gt_pos_pixels)].copy()

    # Sort
    kind = "mergesort" if stable_sort else "quicksort"

    # Case A: worst = smallest F1
    df_A_sorted = df_A.sort_values(by=[f1_col, frame_col], ascending=[True, True], kind=kind)

    # Case B: worst = largest FP-rate
    df_B_sorted = df_B.sort_values(by=["_fp_rate", frame_col], ascending=[False, True], kind=kind)

    if print_tables:
        key_cols_A = ["mode", "precision", "recall", "accuracy", tp_col, tn_col, fp_col, fn_col, "_gt_pos", "_fp_rate", "time_ns", "batch", "i_in_batch"]
        key_cols_B = ["mode", "precision", "recall", f1_col, "accuracy", tp_col, tn_col, fp_col, fn_col, "_gt_pos", "_fp_rate", "time_ns", "batch", "i_in_batch"]

        _print_top_rows(
            f"CASE A (GT fracture present, _gt_pos >= {min_gt_pos_pixels}): TOP {top_k} worst by MIN {f1_col}",
            df_A_sorted,
            top_k=top_k,
            frame_col=frame_col,
            score_col=f1_col,
            key_cols=key_cols_A,
        )

        _print_top_rows(
            f"CASE B (GT fracture absent, _gt_pos < {min_gt_pos_pixels}): TOP {top_k} worst by MAX fp_rate",
            df_B_sorted,
            top_k=top_k,
            frame_col=frame_col,
            score_col="_fp_rate",
            key_cols=key_cols_B,
        )

    return {"case_A": df_A_sorted, "case_B": df_B_sorted}
