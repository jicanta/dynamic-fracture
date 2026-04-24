# source/summarize_total_metrics_across_simulations.py

from __future__ import annotations

import os
from typing import Dict

import pandas as pd


def _safe_div(num: float, den: float) -> float:
    return num / den if den != 0 else 0.0


def _metrics_from_totals(tp: int, tn: int, fp: int, fn: int) -> dict:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)

    return {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def _summarize_one_model(csv_dict: Dict[str, str], model_name: str) -> dict:
    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0
    total_rows = 0

    per_simulation = []

    for sim_label, csv_path in csv_dict.items():
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Could not find CSV for {model_name} / {sim_label}: {csv_path}")

        df = pd.read_csv(csv_path)

        required_cols = {"tp", "tn", "fp", "fn"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV missing required columns {missing} for {model_name} / {sim_label}: {csv_path}"
            )

        df = df.copy()
        for col in ["tp", "tn", "fp", "fn"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["tp", "tn", "fp", "fn"]).reset_index(drop=True)

        sim_tp = int(df["tp"].sum())
        sim_tn = int(df["tn"].sum())
        sim_fp = int(df["fp"].sum())
        sim_fn = int(df["fn"].sum())

        sim_metrics = _metrics_from_totals(sim_tp, sim_tn, sim_fp, sim_fn)
        sim_metrics["simulation"] = sim_label
        sim_metrics["csv_path"] = csv_path
        sim_metrics["n_rows"] = int(len(df))
        per_simulation.append(sim_metrics)

        total_tp += sim_tp
        total_tn += sim_tn
        total_fp += sim_fp
        total_fn += sim_fn
        total_rows += len(df)

    overall = _metrics_from_totals(total_tp, total_tn, total_fp, total_fn)
    overall["model"] = model_name
    overall["n_simulations"] = len(csv_dict)
    overall["total_rows"] = int(total_rows)
    overall["per_simulation"] = per_simulation

    return overall


def summarize_total_metrics_across_simulations(
    csv_map: Dict[str, Dict[str, str]],
    *,
    print_results: bool = True,
):
    """
    Sum confusion counts across multiple simulations for each model type
    and compute total metrics.

    Parameters
    ----------
    csv_map : dict
        Expected form:
        {
            "Base": {
                "100 m/s": "/path/to/base_100/per_frame_metrics.csv",
                "200 m/s": "/path/to/base_200/per_frame_metrics.csv",
                "400 m/s": "/path/to/base_400/per_frame_metrics.csv",
                "1000 m/s": "/path/to/base_1000/per_frame_metrics.csv",
            },
            "Base + SED": {
                "100 m/s": "/path/to/sed_100/per_frame_metrics.csv",
                "200 m/s": "/path/to/sed_200/per_frame_metrics.csv",
                "400 m/s": "/path/to/sed_400/per_frame_metrics.csv",
                "1000 m/s": "/path/to/sed_1000/per_frame_metrics.csv",
            },
        }

    print_results : bool
        If True, prints a readable summary.

    Returns
    -------
    results : dict
        Nested dictionary containing totals for each model type.
    """
    results = {}

    for model_name, model_csvs in csv_map.items():
        results[model_name] = _summarize_one_model(model_csvs, model_name)

    if print_results:
        print("\n" + "=" * 80)
        print("TOTAL METRICS ACROSS SIMULATIONS")
        print("=" * 80)

        for model_name, res in results.items():
            print(f"\nMODEL: {model_name}")
            print("-" * 80)
            print(f"Number of simulations: {res['n_simulations']}")
            print(f"Total rows summed:     {res['total_rows']}")
            print(f"TP: {res['tp']}")
            print(f"TN: {res['tn']}")
            print(f"FP: {res['fp']}")
            print(f"FN: {res['fn']}")
            print(f"Precision: {res['precision']:.6f}")
            print(f"Recall:    {res['recall']:.6f}")
            print(f"F1:        {res['f1']:.6f}")
            print(f"Accuracy:  {res['accuracy']:.6f}")

            print("\nPer-simulation totals:")
            for sim in res["per_simulation"]:
                print(
                    f"  {sim['simulation']:>8s} | "
                    f"TP={sim['tp']}  TN={sim['tn']}  FP={sim['fp']}  FN={sim['fn']}  "
                    f"F1={sim['f1']:.6f}"
                )

        if "Base" in results and "Base + SED" in results:
            b = results["Base"]
            s = results["Base + SED"]

            print("\n" + "=" * 80)
            print("COMPARISON: Base + SED minus Base")
            print("=" * 80)
            print(f"Δ Precision: {s['precision'] - b['precision']:+.6f}")
            print(f"Δ Recall:    {s['recall'] - b['recall']:+.6f}")
            print(f"Δ F1:        {s['f1'] - b['f1']:+.6f}")
            print(f"Δ Accuracy:  {s['accuracy'] - b['accuracy']:+.6f}")
            print(f"Δ TP:        {s['tp'] - b['tp']:+d}")
            print(f"Δ TN:        {s['tn'] - b['tn']:+d}")
            print(f"Δ FP:        {s['fp'] - b['fp']:+d}")
            print(f"Δ FN:        {s['fn'] - b['fn']:+d}")

    return results