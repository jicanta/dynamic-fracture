# src/evaluate.py
"""Continuous autoregressive evaluation on every test case.

Protocol (identical in spirit to source/predict_full_simulation_metrics.py):
  * window of the first T true frames initializes the model
  * each step predicts frame t+1; only the FRACTURE channel is fed back,
    all exogenous channels (ux, uy, Gc, extra, velocity, coords) come from
    ground truth
  * optional no-healing: once a pixel fractures it stays fractured
    (applied to the fed-back state, the saved masks and the binary metrics;
    BCE always uses the raw probabilities)
  * per-frame precision / recall / F1 / accuracy / BCE + PNG masks (flipud,
    white = fracture, same orientation as the old pipeline)

Outputs (standardized):
  OUTPUTS/<run>/eval/<case>/frames/mask_000000.png ...
  OUTPUTS/<run>/eval/<case>/viz/compare_*.png        (GT | prediction)
  OUTPUTS/<run>/eval/<case>/per_frame_metrics.csv
  OUTPUTS/<run>/eval/<case>/{f1,bce}_over_time.png
  OUTPUTS/<run>/eval/<case>/summary.json
  OUTPUTS/<run>/eval/total_metrics.csv
  OUTPUTS/<run>/RESULTS.md

Run:
    python -m src.evaluate --data-root DATASET --run-name tau_base
    python -m src.evaluate --run-name tau_base --cases test_MS206_V400
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .config import Config, parse_config, TEST_CASE_FOLDERS
from .data import RunCache, ensure_cache, load_runs, make_assembler
from .model import build_model
from .utils import amp_dtype, get_device, plot_metric_over_time, seed_everything

EPS = 1e-7


def frame_metrics(gt_bin: np.ndarray, pred_bin: np.ndarray) -> Dict[str, float]:
    tp = float(np.sum((pred_bin == 1) & (gt_bin == 1)))
    tn = float(np.sum((pred_bin == 0) & (gt_bin == 0)))
    fp = float(np.sum((pred_bin == 1) & (gt_bin == 0)))
    fn = float(np.sum((pred_bin == 0) & (gt_bin == 1)))
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    acc = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1, "accuracy": acc}


def save_mask_png(path: Path, mask: np.ndarray) -> None:
    arr = np.flipud(mask)  # same orientation as the original pipeline
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(path)


def save_compare_png(path: Path, gt: np.ndarray, pred: np.ndarray) -> None:
    g = (np.clip(np.flipud(gt), 0, 1) * 255).astype(np.uint8)
    p = (np.clip(np.flipud(pred), 0, 1) * 255).astype(np.uint8)
    sep = np.full((g.shape[0], 3), 128, np.uint8)
    Image.fromarray(np.concatenate([g, sep, p], axis=1)).save(path)


@torch.no_grad()
def evaluate_case(model, run: RunCache, assembler, cfg: Config, case_dir: Path,
                  device, dtype) -> Optional[Dict]:
    T = cfg.sequence_length
    n = run.n_frames
    if n < T + 1:
        print(f"[eval] {run.meta['run_folder']}: too short ({n} frames), skipped")
        return None

    frames_dir = case_dir / "frames"
    viz_dir = case_dir / "viz"
    frames_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    feats = assembler.frames(run, 0, n)                    # (n, C, H, W) float32
    window = torch.from_numpy(feats[:T]).to(device)        # true initial window
    state = window[T - 1, 0].clone()                       # binary fracture state

    rows: List[Dict] = []
    TP = TN = FP = FN = 0.0
    bce_sum = 0.0
    t0 = time.time()

    for t in range(T - 1, n - 1):
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            logits = model(window[None])                   # (1, T, 1, H, W)
        prob = torch.sigmoid(logits[0, -1, 0].float())     # prediction for frame t+1
        pred_bin = (prob >= cfg.eval_threshold).float()
        if cfg.enforce_no_healing:
            state = torch.maximum(state, pred_bin)
        else:
            state = pred_bin

        gt = torch.from_numpy(feats[t + 1, 0]).to(device)
        prob_c = prob.clamp(EPS, 1 - EPS)
        bce = -(gt * prob_c.log() + (1 - gt) * (1 - prob_c).log()).mean().item()

        state_np = state.cpu().numpy()
        gt_np = gt.cpu().numpy()
        m = frame_metrics(gt_np, state_np)
        TP += m["tp"]; TN += m["tn"]; FP += m["fp"]; FN += m["fn"]
        bce_sum += bce

        step = t - (T - 1)
        rows.append({"frame_id": step, "frame_index": t + 1,
                     "H": run.H, "W": run.W, "n_pixels": run.H * run.W,
                     **{k: (int(m[k]) if k in ("tp", "tn", "fp", "fn") else m[k])
                        for k in m}, "bce": bce})

        save_mask_png(frames_dir / f"mask_{step:06d}.png", state_np)
        if cfg.viz_every > 0 and step % cfg.viz_every == 0:
            save_compare_png(viz_dir / f"compare_{step:06d}.png", gt_np, state_np)

        # slide window: next frame uses TRUE exogenous channels + predicted mask
        nxt = torch.from_numpy(feats[t + 1]).to(device).clone()
        nxt[0] = state
        window = torch.cat([window[1:], nxt[None]], dim=0)

    df = pd.DataFrame(rows)
    df.to_csv(case_dir / "per_frame_metrics.csv", index=False)
    plot_metric_over_time(df, case_dir / "f1_over_time.png", y_col="f1", y_label="F1")
    plot_metric_over_time(df, case_dir / "bce_over_time.png", y_col="bce", y_label="BCE loss")

    prec = TP / (TP + FP) if TP + FP > 0 else 0.0
    rec = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    acc = (TP + TN) / max(TP + TN + FP + FN, 1.0)
    summary = {
        "case": case_dir.name,
        "run_folder": run.meta["run_folder"],
        "n_frames_evaluated": len(rows),
        "precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
        "mean_bce": bce_sum / max(len(rows), 1),
        "tp": int(TP), "tn": int(TN), "fp": int(FP), "fn": int(FN),
        "elapsed_s": time.time() - t0,
        "threshold": cfg.eval_threshold,
        "enforce_no_healing": cfg.enforce_no_healing,
    }
    with open(case_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval] {case_dir.name}: F1={f1:.4f} P={prec:.4f} R={rec:.4f} "
          f"acc={acc:.4f} ({len(rows)} frames, {summary['elapsed_s']:.0f}s)")
    return summary


def write_reports(run_dir: Path) -> None:
    """Aggregate every eval/<case>/summary.json into total_metrics.csv + RESULTS.md."""
    eval_dir = run_dir / "eval"
    summaries = []
    for sj in sorted(eval_dir.glob("*/summary.json")):
        with open(sj) as f:
            summaries.append(json.load(f))
    if not summaries:
        print("[eval] no summaries found, nothing to report")
        return

    df = pd.DataFrame(summaries)
    cols = ["case", "n_frames_evaluated", "precision", "recall", "f1",
            "accuracy", "mean_bce"]
    df[cols].to_csv(eval_dir / "total_metrics.csv", index=False)

    TP, TN = df["tp"].sum(), df["tn"].sum()
    FP, FN = df["fp"].sum(), df["fn"].sum()
    prec = TP / (TP + FP) if TP + FP > 0 else 0.0
    rec = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    acc = (TP + TN) / max(TP + TN + FP + FN, 1)

    lines = [
        f"# Results — {run_dir.name}",
        "",
        "Continuous autoregressive rollout per test case "
        "(exogenous channels from ground truth, fracture channel fed back, "
        f"no-healing={summaries[0]['enforce_no_healing']}).",
        "",
        "| Test case | Frames | Precision | Recall | F1 | Accuracy | Mean BCE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['case']} | {s['n_frames_evaluated']} | {s['precision']:.4f} "
            f"| {s['recall']:.4f} | **{s['f1']:.4f}** | {s['accuracy']:.4f} "
            f"| {s['mean_bce']:.4f} |")
    lines += [
        f"| **TOTAL (micro)** |  | **{prec:.4f}** | **{rec:.4f}** | **{f1:.4f}** "
        f"| **{acc:.4f}** |  |",
        "",
        "Per-frame details: `eval/<case>/per_frame_metrics.csv`. "
        "Predicted masks: `eval/<case>/frames/`. "
        "Side-by-side GT|prediction snapshots: `eval/<case>/viz/`.",
    ]
    (run_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(f"[eval] wrote {eval_dir / 'total_metrics.csv'} and {run_dir / 'RESULTS.md'}")
    print(f"[eval] TOTAL micro F1 = {f1:.4f}")


def main(cfg: Config) -> None:
    seed_everything(cfg.seed)
    device = get_device()
    dtype = amp_dtype(device) if cfg.amp else None
    run_dir = cfg.run_dir

    ckpt_path = run_dir / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Train first.")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)

    # rebuild the exact training config (extra channels, model size, T)
    train_cfg = Config(**{k: v for k, v in state["config"].items()
                          if k in Config().__dict__})
    train_cfg.data_root = cfg.data_root
    train_cfg.cache_dir = cfg.cache_dir
    train_cfg.eval_threshold = cfg.eval_threshold
    train_cfg.enforce_no_healing = cfg.enforce_no_healing
    train_cfg.viz_every = cfg.viz_every
    train_cfg.out_root = cfg.out_root
    train_cfg.run_name = cfg.run_name

    wanted = cfg.cases if cfg.cases else list(TEST_CASE_FOLDERS.keys())
    folders = [TEST_CASE_FOLDERS[c] for c in wanted if c in TEST_CASE_FOLDERS]
    ensure_cache(Path(train_cfg.data_root), train_cfg.cache_path, folders,
                 drop_first_csv=train_cfg.drop_first_csv)

    assembler = make_assembler(train_cfg, train_cfg.cache_path)
    model = build_model(train_cfg, assembler.n_channels).to(device)
    model.load_state_dict(state["model"])  # best.pt stores EMA weights in "model"
    model.eval()
    print(f"[eval] loaded {ckpt_path} (epoch {state['epoch']}, "
          f"val AR-F1 {state.get('best_val_f1_ar', float('nan')):.4f})")

    for case in wanted:
        if case not in TEST_CASE_FOLDERS:
            print(f"[eval] unknown case '{case}', skipped")
            continue
        runs = load_runs(train_cfg.cache_path, TEST_CASE_FOLDERS[case])
        if not runs:
            print(f"[eval] {case}: no data under "
                  f"{Path(train_cfg.data_root) / TEST_CASE_FOLDERS[case]}, skipped")
            continue
        for run in runs:  # normally one run folder per test case
            case_dir = run_dir / "eval" / case
            evaluate_case(model, run, assembler, train_cfg, case_dir, device, dtype)

    write_reports(run_dir)


if __name__ == "__main__":
    main(parse_config(description="Evaluate FractureTAU on the test cases"))
