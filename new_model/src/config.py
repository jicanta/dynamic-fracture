# src/config.py
"""Single place for every knob of the SOTA fracture model.

All defaults mirror the conventions of the original ConvLSTM workflow
(T=10, shift=1, ux/uy scale 1e4, velocity scale 1/1000, mask binarized at 0.5)
so results are directly comparable.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


# Test-case mapping is now owned by the framework-free repo-root registry so the
# former hand-synced copies (this file + kathleens-model/source/build_datasets.py)
# can no longer drift (T-02-01). Mirrors the import shim used in build_datasets.py.
REPO_ROOT = Path(__file__).resolve().parents[2]   # src -> new_model -> dynamic-fracture
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from case_registry import TEST_CASE_FOLDERS  # noqa: E402  (re-export; keys used in OUTPUTS/)

EXTRA_CHOICES = ("none", "pressure", "vonmises", "SED")
HEAD_TYPE_CHOICES = ("sigmoid", "monotone_delta")


@dataclass
class Config:
    # ---- paths ----
    data_root: str = "DATASET"
    out_root: str = "OUTPUTS"
    run_name: str = "tau_base"
    cache_dir: Optional[str] = None        # default: <data_root>/_cache

    # ---- data ----
    extra: str = "none"                    # none | pressure | vonmises | SED
    sequence_length: int = 10              # T (same as ConvLSTM workflow)
    uxuy_scale: float = 1e4                # ux/uy divided by this
    velocity_scale: float = 1e-3           # folder velocity * this
    drop_first_csv: bool = True
    val_fraction: float = 0.1              # temporal tail of each run held out
    balance_velocity: bool = False         # equalize training sampling across
                                           # impact velocities (slow runs are rare)
    batch_size: int = 4
    num_workers: int = 4

    # ---- model (SimVP encoder/decoder + TAU translator) ----
    hid_s: int = 64                        # spatial hidden channels
    hid_t: int = 384                       # translator hidden channels
    n_spatial: int = 4                     # encoder/decoder conv blocks (2 downsamples)
    n_temporal: int = 6                    # TAU blocks in the translator
    drop_path: float = 0.05
    head_type: str = "sigmoid"             # 'sigmoid' (logits) | 'monotone_delta' (probabilistic-OR prob)

    # ---- loss ----
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    focal_weight: float = 0.0              # optional, off by default
    pos_weight: float = 1.0                # BCE positive-class weight
    growth_weight: float = 0.0             # extra BCE weight on newly-fractured
                                           # pixels vs the previous frame (0 = off)
    # Focal-Tversky: recall-oriented overlap loss. beta>alpha penalizes false
    # negatives more (fixes P>>R). 0 weight = off; a drop-in for dice_weight.
    tversky_weight: float = 0.0
    tversky_alpha: float = 0.3             # false-positive weight
    tversky_beta: float = 0.7             # false-negative weight (recall knob)
    tversky_gamma: float = 1.0             # focal exponent (1.0 = plain Tversky)
    boundary_weight: float = 0.0           # Kervadec/morphological-band boundary loss weight (0 = off)
    boundary_ramp: bool = True             # ramp boundary_scale 0->1 across stage 1, then hold at 1.0

    # ---- optimization ----
    epochs_stage1: int = 60                # teacher-forced
    epochs_stage2: int = 15                # autoregressive fine-tuning
    rollout_steps: int = 4                 # AR steps per sample in stage 2
    # Scheduled sampling for stage-2 rollout: probability of feeding the model
    # its OWN prediction back (vs ground truth). Annealed ss_start -> ss_end
    # across stage 2 so early epochs stay near teacher-forcing and late epochs
    # match the fully-autoregressive eval protocol.
    ss_start: float = 0.5
    ss_end: float = 1.0
    # Accumulate predictions with a running max during the training/val rollout
    # so the fed-back mask matches eval's no-healing state. Defaults to
    # enforce_no_healing when left at -1.
    rollout_no_healing: int = -1           # -1 = follow enforce_no_healing; 0/1 = override
    ar_pushforward: int = 0                # 1 = detach fed-back state + backprop last unroll step only
    feedback_noise_std: float = 0.0        # soft-prob Gaussian noise on fed-back mask (no-healing-safe)
    feedback_noise_p: float = 0.0          # Bernoulli FP-injection prob on fed-back mask (no-healing-safe)
    lr: float = 1e-3
    lr_stage2: float = 1e-4
    weight_decay: float = 1e-2
    warmup_epochs: int = 3
    clip_grad: float = 1.0
    ema_decay: float = 0.999
    amp: bool = True
    seed: int = 42

    # ---- validation / selection ----
    val_rollout_steps: int = 20            # AR selection-rollout length (Nyquist horizon: long
                                           # enough to sample long-horizon drift, RESEARCH Pitfall 1)
    val_macro_velocity: bool = False       # macro-average val F1 across velocity
                                           # groups so slow runs count equally

    # ---- validation / threshold calibration ----
    calibrate_threshold: bool = True       # after training, pick the val-AR
                                           # F1-optimal decision threshold
    thr_min: float = 0.2
    thr_max: float = 0.8
    thr_steps: int = 25                    # grid points in [thr_min, thr_max]

    # ---- evaluation ----
    eval_threshold: float = 0.5
    use_calibrated_threshold: bool = True  # if calibration.json exists and the
                                           # CLI threshold is the default, use it
    eval_dir_name: str = "eval"            # output subdir; change for threshold sweeps
    enforce_no_healing: bool = True
    viz_every: int = 25                    # save GT|pred comparison every N frames
    cases: List[str] = field(default_factory=list)  # empty = all available

    # ---- misc ----
    resume: str = "none"                   # none | auto | /path/to/last.pt

    # ------------------------------------------------------------------
    @property
    def run_dir(self) -> Path:
        return Path(self.out_root) / self.run_name

    @property
    def cache_path(self) -> Path:
        return Path(self.cache_dir) if self.cache_dir else Path(self.data_root) / "_cache"

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load(path: Path) -> "Config":
        with open(path) as f:
            d = json.load(f)
        cfg = Config()
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


def parse_config(argv=None, description: str = "SOTA dynamic fracture model") -> Config:
    """Build a Config from CLI args. Every dataclass field is exposed as --kebab-case."""
    defaults = Config()
    p = argparse.ArgumentParser(description=description)
    for name, value in asdict(defaults).items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            p.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"),
                           default=value, metavar="BOOL")
        elif isinstance(value, list):
            p.add_argument(flag, nargs="*", default=value)
        elif value is None:
            p.add_argument(flag, type=str, default=None)
        else:
            p.add_argument(flag, type=type(value), default=value)
    args = p.parse_args(argv)

    cfg = Config(**vars(args))
    if cfg.extra not in EXTRA_CHOICES:
        raise SystemExit(f"--extra must be one of {EXTRA_CHOICES}, got '{cfg.extra}'")
    if cfg.head_type not in HEAD_TYPE_CHOICES:
        raise SystemExit(f"--head-type must be one of {HEAD_TYPE_CHOICES}, got '{cfg.head_type}'")
    return cfg
