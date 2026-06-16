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
# Repo root is placed on sys.path by the config import above (Plan 02 shim), so
# the shared metric seam (D-11) is importable here for threshold calibration.
from frac_metrics import per_frame_metrics  # noqa: E402
from .data import WindowDataset, ensure_cache, load_runs, make_assembler
from .losses import SegLoss, binary_f1, growth_mask
from .model import build_model
from .utils import (CSVLogger, EMA, amp_dtype, cosine_warmup_lr,
                    count_parameters, get_device, plot_training_curves,
                    seed_everything)


# ----------------------------------------------------------------------
# Rollout core (shared by stage-2 training and validation)
# ----------------------------------------------------------------------
def rollout_losses(model, batch, T, n_steps, criterion, *, feedback_hard=True,
                   ss_prob=1.0, no_healing=False, ar_pushforward=False,
                   feedback_noise_std=0.0, feedback_noise_p=0.0,
                   head_type="sigmoid"):
    """batch: (B, L, C, H, W) with L >= T + n_steps.

    Step 0 is teacher-forced over the full window (dense supervision).
    Steps k>=1 slide the window by one frame and overwrite the mask channel of
    every frame the model has already predicted with the fed-back mask.

    The fed-back mask mirrors the evaluation protocol:
      * no_healing: keep a running max so a fractured pixel stays fractured,
        exactly as evaluate_case does (state = max(state, pred)). Without this
        the model never sees the monotone state it is scored on.
      * ss_prob: scheduled sampling -- per sample, with probability ss_prob feed
        the model's OWN prediction back, else feed ground truth. ss_prob=1.0 is
        fully autoregressive (matches eval); lower values ease the model into
        long rollouts. Annealed toward 1.0 over stage 2.

    Returns (mean_loss, last_step_fed_mask, last_step_target).
    """
    B, L, C, H, W = batch.shape
    assert L >= T + n_steps

    losses = []
    preds = {}   # global frame idx -> (B, H, W) fed-back mask
    acc = None   # running no-healing state at the last predicted frame

    for k in range(n_steps):
        window = batch[:, k:k + T].clone()
        for j in range(T):
            f = k + j
            if f in preds:
                window[:, j, 0] = preds[f]
        logits = model(window)                       # (B, T, 1, H, W)

        # Pushforward (Brandstetter et al.): roll forward with a DETACHED state and
        # score the LAST step only, so the backprop graph (and memory) is O(1) in K
        # instead of full-BPTT over all K steps. Backprop through the early unroll
        # steps actually HURTS AR rollouts, so we skip the loss on non-final steps.
        score_step = (not ar_pushforward) or (k == n_steps - 1)
        if score_step:
            if k == 0:
                target = batch[:, 1:T + 1, 0:1]          # masks t+1..t+T
                new = growth_mask(target, batch[:, 0:T, 0:1])
                losses.append(criterion(logits, target, new_mask=new))
            else:
                target = batch[:, k + T:k + T + 1, 0:1]  # mask of the new frame only
                new = growth_mask(target, batch[:, k + T - 1:k + T, 0:1])
                losses.append(criterion(logits[:, -1:], target, new_mask=new))

        # head_type read: the monotone-delta head already returns a probability,
        # so do NOT sigmoid a second time (Pitfall 4 / threat T-03-11).
        p = (logits[:, -1, 0] if head_type == "monotone_delta"
             else torch.sigmoid(logits[:, -1, 0]))    # prob of frame k+T
        # Feedback-noise (no-healing-safe): inject BEFORE the straight-through
        # binarization so the downstream torch.maximum (no_healing) guarantees the
        # perturbation can only ADD damage (0->1), never heal (1->0).
        if feedback_noise_std:
            p = (p + feedback_noise_std * torch.randn_like(p)).clamp(0.0, 1.0)
        if feedback_noise_p:
            flip = (torch.rand_like(p) < feedback_noise_p).float()
            p = torch.maximum(p, flip)                # add-only FP injection
        if feedback_hard:
            # straight-through: binary forward (matches eval), soft gradient
            p = (p > 0.5).float() + p - p.detach()
        if no_healing:
            prev = acc if acc is not None else batch[:, T - 1, 0]
            p = torch.maximum(prev, p)
        acc = p
        if ss_prob < 1.0:
            gt = batch[:, k + T, 0]
            use_pred = (torch.rand(B, 1, 1, device=batch.device) < ss_prob).to(p.dtype)
            p = use_pred * p + (1.0 - use_pred) * gt
        # Pushforward: detach the fed-back state so the next step's window carries
        # no gradient history (the graph rebuilds only for the final scored step).
        preds[k + T] = p.detach() if ar_pushforward else p

    last_target = batch[:, n_steps + T - 1, 0]
    return torch.stack(losses).mean(), preds[n_steps + T - 1], last_target


# ----------------------------------------------------------------------
@torch.no_grad()
def validate(model, loader, T, criterion, device, dtype, val_rollout_steps,
             *, macro_velocity=False, vel_channel=None, no_healing=False,
             head_type="sigmoid"):
    """If macro_velocity, F1 is averaged per velocity group first so slow runs
    (few windows, sparse fracture) weigh as much as fast ones in selection."""
    model.eval()
    tot_loss, n = 0.0, 0
    recs = []  # (velocity, f1_tf, f1_ar) per sample
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            # teacher-forced loss + last-frame F1
            logits = model(batch[:, :T])
            target = batch[:, 1:T + 1, 0:1]
            new = growth_mask(target, batch[:, 0:T, 0:1])
            tot_loss += criterion(logits, target, new_mask=new).item() * batch.shape[0]
            n += batch.shape[0]
            p_tf = (logits[:, -1, 0] if head_type == "monotone_delta"
                    else torch.sigmoid(logits[:, -1, 0]))
            pred_bin = (p_tf > 0.5).float()

            # short AR rollout F1 (model-selection metric: matches the eval protocol).
            # Clean rollout: no feedback-noise / no pushforward (selection must stay
            # reproducible and never teacher-forced -- threat T-03-12); only the
            # head_type read is threaded so the monotone head is not double-sigmoided.
            _, last_pred, last_tgt = rollout_losses(
                model, batch, T, val_rollout_steps, criterion,
                feedback_hard=True, no_healing=no_healing, head_type=head_type)
            ar_bin = (last_pred > 0.5).float()
        for i in range(batch.shape[0]):
            vel = (round(batch[i, 0, vel_channel, 0, 0].item(), 6)
                   if vel_channel is not None else 0.0)
            recs.append((vel,
                         binary_f1(pred_bin[i], batch[i, T, 0]),
                         binary_f1(ar_bin[i], last_tgt[i])))
    if macro_velocity:
        groups = {}
        for v, tf, ar in recs:
            groups.setdefault(v, []).append((tf, ar))
        f1_tf = float(np.mean([np.mean([t for t, _ in g]) for g in groups.values()]))
        f1_ar = float(np.mean([np.mean([a for _, a in g]) for g in groups.values()]))
    else:
        f1_tf = float(np.mean([t for _, t, _ in recs]))
        f1_ar = float(np.mean([a for _, _, a in recs]))
    return tot_loss / max(n, 1), f1_tf, f1_ar


# ----------------------------------------------------------------------
@torch.no_grad()
def calibrate_threshold(model, loader, T, device, dtype, val_rollout_steps,
                        thresholds, *, no_healing=True):
    """Sweep the decision threshold on the val AR rollout and return the one
    with the best micro-F1.

    Mirrors evaluate_case: at each step the predicted mask is binarized at the
    candidate threshold, optionally accumulated with a running max (no-healing),
    and fed back. Because the threshold drives both feedback and scoring it is
    swept end-to-end rather than on fixed probabilities. The grid stays modest
    (a long full-rollout confirmation is what scripts/eval_sweep.sbatch is for).
    """
    model.eval()
    stats = {float(thr): [0.0, 0.0, 0.0] for thr in thresholds}  # tp, fp, fn
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        B, L, C, H, W = batch.shape
        for thr in thresholds:
            acc, preds = None, {}
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                for k in range(val_rollout_steps):
                    window = batch[:, k:k + T].clone()
                    for j in range(T):
                        f = k + j
                        if f in preds:
                            window[:, j, 0] = preds[f]
                    logits = model(window)
                    pb = (torch.sigmoid(logits[:, -1, 0].float()) >= thr).float()
                    if no_healing:
                        prev = acc if acc is not None else batch[:, T - 1, 0]
                        pb = torch.maximum(prev, pb)
                    acc, preds[k + T] = pb, pb
            tgt = batch[:, val_rollout_steps + T - 1, 0]
            # Route the per-threshold counts through the shared seam (D-11) so the
            # calibration F1 is the SAME implementation used at eval time (CMP-04).
            # prob is unused here (only tp/fp/fn feed F1), so the binarized mask is
            # passed in its place.
            acc_np = acc.detach().cpu().numpy()
            tgt_np = tgt.detach().cpu().numpy()
            m = per_frame_metrics(tgt_np, acc_np, acc_np)
            s = stats[float(thr)]
            s[0] += m["tp"]
            s[1] += m["fp"]
            s[2] += m["fn"]

    table, best_thr, best_f1 = [], 0.5, -1.0
    for thr in thresholds:
        tp, fp, fn = stats[float(thr)]
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        table.append({"threshold": float(thr), "precision": prec,
                      "recall": rec, "f1": f1})
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr, best_f1, table


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

    def loader(ds, shuffle, balance=False):
        sampler = None
        if balance:
            # inverse-frequency sampling: each velocity group is drawn equally
            # often, otherwise fast runs (dense fracture) dominate the epochs
            v = ds.window_velocities()
            _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
            weights = torch.as_tensor(1.0 / counts[inv], dtype=torch.double)
            sampler = torch.utils.data.WeightedRandomSampler(
                weights, num_samples=len(ds), replacement=True)
        train_like = shuffle or sampler is not None
        return DataLoader(ds, batch_size=cfg.batch_size,
                          shuffle=shuffle and sampler is None, sampler=sampler,
                          num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
                          drop_last=train_like, persistent_workers=cfg.num_workers > 0)

    dl_s1 = loader(ds_tr_s1, True, cfg.balance_velocity)
    dl_s2 = loader(ds_tr_s2, True, cfg.balance_velocity)
    dl_val = loader(ds_val, False)

    # ---- model / optim ----
    model = build_model(cfg, assembler.n_channels).to(device)
    print(f"[train] FractureTAU parameters: {count_parameters(model)/1e6:.2f}M")
    criterion = SegLoss(bce_weight=cfg.bce_weight, dice_weight=cfg.dice_weight,
                        focal_weight=cfg.focal_weight, pos_weight=cfg.pos_weight,
                        growth_weight=cfg.growth_weight,
                        tversky_weight=cfg.tversky_weight,
                        tversky_alpha=cfg.tversky_alpha,
                        tversky_beta=cfg.tversky_beta,
                        tversky_gamma=cfg.tversky_gamma).to(device)
    no_healing = (cfg.enforce_no_healing if cfg.rollout_no_healing < 0
                  else bool(cfg.rollout_no_healing))
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
        # scheduled sampling: anneal ss_start -> ss_end across stage 2 so the
        # model eases from near-teacher-forcing into the fully-AR eval regime.
        if stage == 2 and cfg.epochs_stage2 > 1:
            frac = (epoch - cfg.epochs_stage1) / (cfg.epochs_stage2 - 1)
            ss_prob = cfg.ss_start + (cfg.ss_end - cfg.ss_start) * frac
        else:
            ss_prob = cfg.ss_end

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
                loss, _, _ = rollout_losses(
                    model, batch, T, n_steps, criterion,
                    ss_prob=ss_prob, no_healing=no_healing,
                    ar_pushforward=bool(cfg.ar_pushforward),
                    feedback_noise_std=cfg.feedback_noise_std,
                    feedback_noise_p=cfg.feedback_noise_p,
                    head_type=cfg.head_type)

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
                                          dtype, cfg.val_rollout_steps,
                                          macro_velocity=cfg.val_macro_velocity,
                                          vel_channel=assembler.n_channels - 3,
                                          no_healing=no_healing,
                                          head_type=cfg.head_type)
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

    # ---- threshold calibration on the best (EMA) checkpoint ----
    if cfg.calibrate_threshold:
        best_path = run_dir / "checkpoints" / "best.pt"
        if best_path.exists():
            state = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(state["model"])  # best.pt stores EMA weights
        else:
            ema.copy_to(model)
        grid = np.linspace(cfg.thr_min, cfg.thr_max, cfg.thr_steps).tolist()
        thr, f1, table = calibrate_threshold(
            model, dl_val, T, device, dtype, cfg.val_rollout_steps, grid,
            no_healing=no_healing)
        calib = {"threshold": thr, "val_f1_ar": f1,
                 "default_f1_at_0.5": next((r["f1"] for r in table
                                            if abs(r["threshold"] - 0.5) < 1e-6), None),
                 "grid": table}
        with open(run_dir / "calibration.json", "w") as f:
            json.dump(calib, f, indent=2)
        print(f"[train] calibrated threshold = {thr:.3f} (val AR-F1 {f1:.4f}); "
              f"wrote {run_dir / 'calibration.json'}")


if __name__ == "__main__":
    main(parse_config(description="Train FractureTAU"))
