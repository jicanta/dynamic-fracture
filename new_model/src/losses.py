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


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0,
                   *, inputs_are_probs: bool = False) -> torch.Tensor:
    """Mean (1 - Dice) per frame. logits/target: (..., H, W).

    With ``inputs_are_probs`` (monotone-delta head, MODEL-03) the input is already
    a probability, so the sigmoid is skipped to avoid double-squashing (Pitfall 4).
    """
    p = logits if inputs_are_probs else torch.sigmoid(logits)
    dims = (-2, -1)
    inter = (p * target).sum(dims)
    denom = p.sum(dims) + target.sum(dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return (1.0 - dice).mean()


def focal_loss(logits: torch.Tensor, target: torch.Tensor,
               gamma: float = 2.0, alpha: float = 0.25,
               *, inputs_are_probs: bool = False) -> torch.Tensor:
    p = logits if inputs_are_probs else torch.sigmoid(logits)
    if inputs_are_probs:
        # BCE on probabilities is unsafe under autocast — float32 + autocast off.
        with torch.autocast(device_type=p.device.type, enabled=False):
            bce = F.binary_cross_entropy(p.float(), target.float(), reduction="none")
    else:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = p * target + (1.0 - p) * (1.0 - target)
    at = alpha * target + (1.0 - alpha) * (1.0 - target)
    return (at * (1.0 - pt).pow(gamma) * bce).mean()


def tversky_loss(logits: torch.Tensor, target: torch.Tensor, *,
                 alpha: float = 0.3, beta: float = 0.7, gamma: float = 1.0,
                 eps: float = 1.0, inputs_are_probs: bool = False) -> torch.Tensor:
    """Focal-Tversky loss, per frame.

    Tversky index TI = TP / (TP + alpha*FP + beta*FN). With beta > alpha,
    false negatives cost more than false positives, which directly trades
    precision for recall -- the right knob when the model under-predicts the
    crack (our case: P >> R). gamma > 1 focuses gradient on hard frames
    (gamma = 1 is plain Tversky). Reduces to soft Dice at alpha = beta = 0.5.

    ``inputs_are_probs`` skips the sigmoid when the monotone-delta head already
    returns a probability (MODEL-03; no double-sigmoid, Pitfall 4).
    """
    p = logits if inputs_are_probs else torch.sigmoid(logits)
    dims = (-2, -1)
    tp = (p * target).sum(dims)
    fp = (p * (1.0 - target)).sum(dims)
    fn = ((1.0 - p) * target).sum(dims)
    ti = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return (1.0 - ti).pow(gamma).mean()


def growth_mask(target: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
    """Pixels fractured in `target` but not in the previous frame's mask.

    With no-healing data the mask only grows, so this is the per-frame growth
    ring — the only part of the target the model cannot get by copying its
    input forward. Same shape as target, values in {0, 1}.
    """
    return (target * (1.0 - prev)).clamp(0.0, 1.0)


def boundary_band(target: torch.Tensor, *, width: int = 1) -> torch.Tensor:
    """0/1 morphological band hugging the crack front (torch-only, no SciPy).

    Dilates the GT fracture mask by ``width`` pixels with a max-pool and subtracts
    the original mask, yielding the immediate background ring around the front --
    exactly where false negatives stall the autoregressive rollout. This is the
    default boundary target when the caller passes no ``dist_map`` (RESEARCH Open
    Question 3: avoids a SciPy distance-transform dependency mid-phase).

    target: ``(..., H, W)`` with any number of leading dims. The SegLoss training
    paths call this with 5D ``(B, T, 1, H, W)`` masks (teacher-forced and AR
    rollout) as well as 4D ``(B, C, H, W)``; ``F.max_pool2d`` only accepts 3D/4D,
    so the leading dims are flattened into the batch axis for the spatial dilation
    and restored afterwards. Returns the same shape as ``target``, values in {0, 1}.
    """
    k = 2 * width + 1
    orig_shape = target.shape
    h, w = orig_shape[-2], orig_shape[-1]
    flat = target.reshape(-1, 1, h, w)               # (N, 1, H, W) — pool is spatial-only
    dilated = F.max_pool2d(flat, kernel_size=k, stride=1, padding=width)
    dilated = dilated.reshape(orig_shape)
    return (dilated - target).clamp(0.0, 1.0)


class SegLoss(nn.Module):
    def __init__(self, *, bce_weight: float = 1.0, dice_weight: float = 1.0,
                 focal_weight: float = 0.0, pos_weight: float = 1.0,
                 growth_weight: float = 0.0, tversky_weight: float = 0.0,
                 tversky_alpha: float = 0.3, tversky_beta: float = 0.7,
                 tversky_gamma: float = 1.0, boundary_weight: float = 0.0,
                 head_type: str = "sigmoid"):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.focal_weight = float(focal_weight)
        self.growth_weight = float(growth_weight)
        self.tversky_weight = float(tversky_weight)
        self.tversky_alpha = float(tversky_alpha)
        self.tversky_beta = float(tversky_beta)
        self.tversky_gamma = float(tversky_gamma)
        self.boundary_weight = float(boundary_weight)   # MODEL-01 crack-front penalty (off by default)
        self.head_type = head_type                      # "sigmoid" (logits) | "monotone_delta" (probs)
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                new_mask: torch.Tensor | None = None,
                *, dist_map: torch.Tensor | None = None,
                boundary_scale: float = 1.0) -> torch.Tensor:
        # MODEL-03 contract: the monotone-delta head already returns a probability,
        # so SegLoss must NOT sigmoid a second time (Pitfall 4 / threat T-03-03).
        # Sigmoid mode is byte-identical to the legacy code path.
        inputs_are_probs = (self.head_type == "monotone_delta")
        p_pred = logits if inputs_are_probs else torch.sigmoid(logits)

        loss = logits.new_zeros(())
        if self.bce_weight:
            if self.growth_weight and new_mask is not None:
                # most positive pixels already exist in the input mask, so a
                # copy-forward model scores near-perfectly; up-weight the
                # growth ring (and its immediate background, where false
                # negatives stall the autoregressive rollout)
                if inputs_are_probs:
                    # BCE on probabilities is unsafe under autocast (PyTorch bans it:
                    # log() loses precision in bf16/fp16). Compute it in float32 with
                    # autocast disabled. The sigmoid/_with_logits path is unaffected.
                    with torch.autocast(device_type=p_pred.device.type, enabled=False):
                        bce = F.binary_cross_entropy(
                            p_pred.float(), target.float(), reduction="none")
                else:
                    bce = F.binary_cross_entropy_with_logits(
                        logits, target, pos_weight=self.pos_weight, reduction="none")
                w = 1.0 + self.growth_weight * new_mask
                bce = (bce * w).sum() / w.sum().clamp_min(1.0)
            else:
                if inputs_are_probs:
                    # float32 + autocast disabled — see growth-weight branch above.
                    with torch.autocast(device_type=p_pred.device.type, enabled=False):
                        bce = F.binary_cross_entropy(p_pred.float(), target.float())
                else:
                    bce = F.binary_cross_entropy_with_logits(
                        logits, target, pos_weight=self.pos_weight)
            loss = loss + self.bce_weight * bce
        if self.dice_weight:
            loss = loss + self.dice_weight * soft_dice_loss(
                logits, target, inputs_are_probs=inputs_are_probs)
        if self.focal_weight:
            loss = loss + self.focal_weight * focal_loss(
                logits, target, inputs_are_probs=inputs_are_probs)
        if self.tversky_weight:
            loss = loss + self.tversky_weight * tversky_loss(
                logits, target, alpha=self.tversky_alpha,
                beta=self.tversky_beta, gamma=self.tversky_gamma,
                inputs_are_probs=inputs_are_probs)
        if self.boundary_weight and boundary_scale:
            # crack-front penalty: probability mass that leaks into the band of
            # background pixels hugging the GT front. Ramp (boundary_scale, set
            # per-epoch by the training loop) scales the contribution; 0 -> off.
            band = dist_map if dist_map is not None else boundary_band(target)
            if band.shape != p_pred.shape:
                raise ValueError(
                    f"boundary band {tuple(band.shape)} != pred {tuple(p_pred.shape)}")
            loss = loss + self.boundary_weight * float(boundary_scale) * (p_pred * band).mean()
        return loss


@torch.no_grad()
def binary_f1(pred_bin: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    tp = (pred_bin * target).sum().item()
    fp = (pred_bin * (1 - target)).sum().item()
    fn = ((1 - pred_bin) * target).sum().item()
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    return 2 * prec * rec / (prec + rec + eps)
