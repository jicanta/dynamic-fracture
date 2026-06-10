# src/losses.py
"""Segmentation loss for sparse fracture masks: BCE + soft Dice (+ optional focal).

The original ConvLSTM trained with plain BCE; fracture pixels are a small
fraction of the grid, so plain BCE under-weights the crack. Dice directly
optimizes overlap (a monotone surrogate of F1, the headline test metric).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Mean (1 - Dice) per frame. logits/target: (..., H, W)."""
    p = torch.sigmoid(logits)
    dims = (-2, -1)
    inter = (p * target).sum(dims)
    denom = p.sum(dims) + target.sum(dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return (1.0 - dice).mean()


def focal_loss(logits: torch.Tensor, target: torch.Tensor,
               gamma: float = 2.0, alpha: float = 0.25) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * target + (1.0 - p) * (1.0 - target)
    at = alpha * target + (1.0 - alpha) * (1.0 - target)
    return (at * (1.0 - pt).pow(gamma) * bce).mean()


class SegLoss(nn.Module):
    def __init__(self, *, bce_weight: float = 1.0, dice_weight: float = 1.0,
                 focal_weight: float = 0.0, pos_weight: float = 1.0):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.focal_weight = float(focal_weight)
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = logits.new_zeros(())
        if self.bce_weight:
            loss = loss + self.bce_weight * F.binary_cross_entropy_with_logits(
                logits, target, pos_weight=self.pos_weight)
        if self.dice_weight:
            loss = loss + self.dice_weight * soft_dice_loss(logits, target)
        if self.focal_weight:
            loss = loss + self.focal_weight * focal_loss(logits, target)
        return loss


@torch.no_grad()
def binary_f1(pred_bin: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    tp = (pred_bin * target).sum().item()
    fp = (pred_bin * (1 - target)).sum().item()
    fn = ((1 - pred_bin) * target).sum().item()
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    return 2 * prec * rec / (prec + rec + eps)
