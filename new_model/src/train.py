# src/train.py
"""Two-stage training.

Stage 1 (teacher-forced): input frames t..t+T-1 -> predict masks t+1..t+T,
loss on all T steps. Same supervision as the original ConvLSTM workflow.

Stage 2 (autoregressive fine-tuning): roll the model forward `rollout_steps`
steps feeding its OWN mask predictions back into channel 0 (exogenous channels
stay ground truth, exactly like the evaluation protocol), with loss on each
new frame. This closes the train/test mismatch that hurts AR rollouts.

Run:
    python -m src.train --data-root DATASET --run-name tau_base --extra none
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config, parse_config, TEST_CASE_FOLDERS
from .data import WindowDataset, ensure_cache, load_runs, make_assembler
from .losses import SegLoss, binary_f1
from .model import build_model
from .utils import (CSVLogger, EMA, amp_dtype, cosine_warmup_lr,
                    count_parameters, get_device, plot_training_curves,
                    seed_everything)


# ----------------------------------------------------------------------
# Rollout core (shared by stage-2 training and validation)
# ----------------------------------------------------------------------
def rollout_losses(model, batch, T, n_steps, criterion, *, feedback_hard=True):
    """batch: (B, L, C, H, W) with L >= T + n_steps.

    Step 0 is teacher-forced over the full window (dense supervision).
    Steps k>=1 slide the window by one frame and overwrite the mask channel of
    every frame the model has already predicted with its own prediction.
    Returns (total_loss, last_step_logits, last_step_target).
    """
    B, L, C, H, W = batch.shape
    assert L >= T + n_steps

    losses = []
    preds = {}  # global frame idx -> (B, H, W) predicted mask probability

    for k in range(n_steps):
        window = batch[:, k:k + T].clone()
        for j in range(T):
            f = k + j
            if f in preds:
                window[:, j, 0] = preds[f]
        logits = model(window)                       # (B, T, 1, H, W)

        if k == 0:
            target = batch[:, 1:T + 1, 0:1]          # masks t+1..t+T
            losses.append(criterion(logits, target))
        else:
            target = batch[:, k + T:k + T + 1, 0:1]  # mask of the new frame only
            losses.append(criterion(logits[:, -1:], target))

        p = torch.sigmoid(logits[:, -1, 0])          # prob of frame k+T
        if feedback_hard:
            # straight-through: binary forward (matches eval), soft gradient
            p = (p > 0.5).float() + p - p.detach()
        preds[k + T] = p

    last_target = batch[:, n_steps + T - 1, 0]
    return torch.stack(losses).mean(), preds[n_steps + T - 1], last_target


# ----------------------------------------------------------------------
@torch.no_grad()
def validate(model, loader, T, criterion, device, dtype, val_rollout_steps):
    model.eval()
    tot_loss, n = 0.0, 0
    f1_tf, f1_ar = [], []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            # teacher-forced loss + last-frame F1
            logits = model(batch[:, :T])
            target = batch[:, 1:T + 1, 0:1]
            tot_loss += criterion(logits, target).item() * batch.shape[0]
            n += batch.shape[0]
            pred_bin = (torch.sigmoid(logits[:, -1, 0]) > 0.5).float()
            f1_tf.append(binary_f1(pred_bin, batch[:, T, 0]))

            # short AR rollout F1 (model-selection metric: matches the eval protocol)
            _, last_pred, last_tgt = rollout_losses(
                model, batch, T, val_rollout_steps, criterion, feedback_hard=True)
            f1_ar.append(binary_f1((last_pred > 0.5).float(), last_tgt))
    return tot_loss / max(n, 1), float(np.mean(f1_tf)), float(np.mean(f1_ar))


# ----------------------------------------------------------------------
def save_checkpoint(path, *, model, ema, optimizer, epoch, cfg, best_val_f1_ar):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_f1_ar": best_val_f1_ar,
        "config": cfg.__dict__,
    }, path)


def main(cfg: Config) -> None:
    seed_everything(cfg.seed)
    device = get_device()
    dtype = amp_dtype(device) if cfg.amp else None
    print(f"[train] device={device} amp_dtype={dtype}")

    run_dir = cfg.run_dir
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.json")

    # ---- data ----
    top_folders = ["trainDS"] + list(TEST_CASE_FOLDERS.values())
    ensure_cache(Path(cfg.data_root), cfg.cache_path, top_folders,
                 drop_first_csv=cfg.drop_first_csv)

    assembler = make_assembler(cfg, cfg.cache_path)
    runs = load_runs(cfg.cache_path, "trainDS")
    if not runs:
        raise FileNotFoundError("No cached training runs found under trainDS.")
    print(f"[train] {len(runs)} training runs, input channels: {assembler.channel_names()}")

    T = cfg.sequence_length
    L_train = T + max(cfg.rollout_steps, 1)
    L_val = T + max(cfg.val_rollout_steps, 1)

    ds_tr_s1 = WindowDataset(runs, assembler, T + 1, split="train",
                             val_fraction=cfg.val_fraction)
    ds_tr_s2 = WindowDataset(runs, assembler, L_train, split="train",
                             val_fraction=cfg.val_fraction)
    ds_val = WindowDataset(runs, assembler, L_val, split="val",
                           val_fraction=cfg.val_fraction)
    print(f"[train] windows: stage1={len(ds_tr_s1)} stage2={len(ds_tr_s2)} val={len(ds_val)}")

    def loader(ds, shuffle):
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                          num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
                          drop_last=shuffle, persistent_workers=cfg.num_workers > 0)

    dl_s1, dl_s2, dl_val = loader(ds_tr_s1, True), loader(ds_tr_s2, True), loader(ds_val, False)

    # ---- model / optim ----
    model = build_model(cfg, assembler.n_channels).to(device)
    print(f"[train] FractureTAU parameters: {count_parameters(model)/1e6:.2f}M")
    criterion = SegLoss(bce_weight=cfg.bce_weight, dice_weight=cfg.dice_weight,
                        focal_weight=cfg.focal_weight, pos_weight=cfg.pos_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    ema = EMA(model, cfg.ema_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=(dtype == torch.float16))

    total_epochs = cfg.epochs_stage1 + cfg.epochs_stage2
    start_epoch, best_val_f1_ar = 0, -1.0

    # ---- resume ----
    last_ckpt = run_dir / "checkpoints" / "last.pt"
    resume_path = None
    if cfg.resume == "auto" and last_ckpt.exists():
        resume_path = last_ckpt
    elif cfg.resume not in ("none", "auto"):
        resume_path = Path(cfg.resume)
    if resume_path is not None:
        state = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        best_val_f1_ar = state.get("best_val_f1_ar", -1.0)
        print(f"[train] resumed from {resume_path} at epoch {start_epoch}")
    if start_epoch >= total_epochs:
        print("[train] nothing to do (already trained to completion).")
        return

    logger = CSVLogger(
        run_dir / "logs" / "train_log.csv",
        ["epoch", "stage", "lr", "train_loss", "val_loss",
         "val_f1_tf", "val_f1_ar", "epoch_time_s"],
        append=start_epoch > 0,
    )

    steps_s1 = max(len(dl_s1), 1) * cfg.epochs_stage1
    steps_s2 = max(len(dl_s2), 1) * cfg.epochs_stage2
    warmup = max(len(dl_s1), 1) * cfg.warmup_epochs
    global_step_s1 = max(len(dl_s1), 1) * min(start_epoch, cfg.epochs_stage1)
    global_step_s2 = max(len(dl_s2), 1) * max(start_epoch - cfg.epochs_stage1, 0)

    # ---- epochs ----
    for epoch in range(start_epoch, total_epochs):
        stage = 1 if epoch < cfg.epochs_stage1 else 2
        dl = dl_s1 if stage == 1 else dl_s2
        n_steps = 1 if stage == 1 else cfg.rollout_steps

        model.train()
        t0 = time.time()
        running, nb = 0.0, 0
        for batch in dl:
            batch = batch.to(device, non_blocking=True)

            if stage == 1:
                lr = cosine_warmup_lr(global_step_s1, steps_s1, warmup, cfg.lr)
                global_step_s1 += 1
            else:
                lr = cosine_warmup_lr(global_step_s2, steps_s2, 0, cfg.lr_stage2)
                global_step_s2 += 1
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                loss, _, _ = rollout_losses(model, batch, T, n_steps, criterion)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            running += loss.item()
            nb += 1

        train_loss = running / max(nb, 1)

        # validate with EMA weights
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ema.copy_to(model)
        val_loss, f1_tf, f1_ar = validate(model, dl_val, T, criterion, device,
                                          dtype, cfg.val_rollout_steps)
        if f1_ar > best_val_f1_ar:
            best_val_f1_ar = f1_ar
            save_checkpoint(run_dir / "checkpoints" / "best.pt", model=model, ema=ema,
                            optimizer=optimizer, epoch=epoch, cfg=cfg,
                            best_val_f1_ar=best_val_f1_ar)
        model.load_state_dict(backup)

        save_checkpoint(last_ckpt, model=model, ema=ema, optimizer=optimizer,
                        epoch=epoch, cfg=cfg, best_val_f1_ar=best_val_f1_ar)

        dt = time.time() - t0
        logger.log({"epoch": epoch, "stage": stage, "lr": f"{lr:.3e}",
                    "train_loss": f"{train_loss:.5f}", "val_loss": f"{val_loss:.5f}",
                    "val_f1_tf": f"{f1_tf:.5f}", "val_f1_ar": f"{f1_ar:.5f}",
                    "epoch_time_s": f"{dt:.1f}"})
        print(f"[epoch {epoch:03d}/{total_epochs}] stage{stage} "
              f"train={train_loss:.5f} val={val_loss:.5f} "
              f"F1tf={f1_tf:.4f} F1ar={f1_ar:.4f} ({dt:.0f}s)")

    logger.close()
    plot_training_curves(run_dir / "logs" / "train_log.csv",
                         run_dir / "curves" / "training_curves.png")
    print(f"[train] done. best val AR-F1 = {best_val_f1_ar:.4f}")
    print(f"[train] best checkpoint: {run_dir / 'checkpoints' / 'best.pt'}")


if __name__ == "__main__":
    main(parse_config(description="Train FractureTAU"))
