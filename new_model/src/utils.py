# src/utils.py
from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_dtype(device: torch.device) -> Optional[torch.dtype]:
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


class EMA:
    """Exponential moving average of model parameters; used for eval/checkpoints."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach().float(), alpha=1.0 - self.decay)
            else:
                s.copy_(v)

    def copy_to(self, model: torch.nn.Module) -> None:
        sd = model.state_dict()
        for k in sd:
            sd[k].copy_(self.shadow[k].to(sd[k].dtype))

    def state_dict(self) -> Dict:
        return self.shadow

    def load_state_dict(self, sd: Dict) -> None:
        self.shadow = {k: v.clone().float() for k, v in sd.items()}


def cosine_warmup_lr(step: int, total_steps: int, warmup_steps: int,
                     base_lr: float, min_lr_ratio: float = 0.01) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    t = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    t = min(max(t, 0.0), 1.0)
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * t)))


class CSVLogger:
    def __init__(self, path: Path, fieldnames: List[str], append: bool = False):
        self.path = Path(path)
        self.fieldnames = fieldnames
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if (append and self.path.exists()) else "w"
        self._f = open(self.path, mode, newline="")
        self._w = csv.DictWriter(self._f, fieldnames=fieldnames)
        if mode == "w":
            self._w.writeheader()

    def log(self, row: Dict) -> None:
        self._w.writerow({k: row.get(k, "") for k in self.fieldnames})
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def plot_training_curves(log_csv: Path, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(log_csv)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(df["epoch"], df["train_loss"], label="train loss")
    axes[0].plot(df["epoch"], df["val_loss"], label="val loss")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(df["epoch"], df["val_f1_tf"], label="val F1 (teacher-forced)")
    axes[1].plot(df["epoch"], df["val_f1_ar"], label="val F1 (AR rollout)")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("F1")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    for ax in axes:
        for x in df.loc[df["stage"] == 2, "epoch"].head(1):
            ax.axvline(x, color="gray", ls="--", lw=1)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_metric_over_time(df, out_png: Path, *, y_col: str, y_label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 6))
    plt.plot(df["frame_index"], df[y_col], linewidth=2.0)
    plt.xlabel("Frame index", fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
