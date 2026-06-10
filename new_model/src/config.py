# src/config.py
"""Single place for every knob of the SOTA fracture model.

All defaults mirror the conventions of the original ConvLSTM workflow
(T=10, shift=1, ux/uy scale 1e4, velocity scale 1/1000, mask binarized at 0.5)
so results are directly comparable.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


# Same test-case mapping as ALL_INPUTS/CODE/source/build_datasets.py.
# Keys are short names used in OUTPUTS/, values are folder names inside DATASET/.
TEST_CASE_FOLDERS = {
    "test_MS206_V100": "F_MS206_V100_out",
    "test_MS206_V200": "F_MS206_V200_out",
    "test_MS206_V400": "F_MS206_V400_out",
    "test_MS206_V1000": "F_MS206_V1000_out",
    "test_MS210_V400": "F_MS210_V400_out",
    "test_PBX1_V400_true": "F_PBX1_V400_true",
    "test_PBX_2_V400": "F_PBX_2_V400_out",
    "test_MS5_V150ms_inc": "MS5_V150ms_inc",
    "test_MS5_V400ms": "MS205_V400ms_MS5",
    "test_emergency_horizontal": "F_emergency_horizontal_out",
    "test_horizontal_layers_3": "F_horizontal-layers_3_out",
    "test_horizontal_layers_4": "F_horizontal-layers_4_out",
    "test_inclusions_1_2": "F_inclusions_1_2_out",
    "test_inclusions_2_2": "F_inclusions_2_2_out",
    "test_inclusions_3_2": "F_inclusions_3_2_out",
    "test_inclusions_true_V400": "F_inclusions_true_V400",
}

EXTRA_CHOICES = ("none", "pressure", "vonmises", "SED")


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
    batch_size: int = 4
    num_workers: int = 4

    # ---- model (SimVP encoder/decoder + TAU translator) ----
    hid_s: int = 64                        # spatial hidden channels
    hid_t: int = 384                       # translator hidden channels
    n_spatial: int = 4                     # encoder/decoder conv blocks (2 downsamples)
    n_temporal: int = 6                    # TAU blocks in the translator
    drop_path: float = 0.05

    # ---- loss ----
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    focal_weight: float = 0.0              # optional, off by default
    pos_weight: float = 1.0                # BCE positive-class weight

    # ---- optimization ----
    epochs_stage1: int = 60                # teacher-forced
    epochs_stage2: int = 15                # autoregressive fine-tuning
    rollout_steps: int = 4                 # AR steps per sample in stage 2
    lr: float = 1e-3
    lr_stage2: float = 1e-4
    weight_decay: float = 1e-2
    warmup_epochs: int = 3
    clip_grad: float = 1.0
    ema_decay: float = 0.999
    amp: bool = True
    seed: int = 42

    # ---- validation / selection ----
    val_rollout_steps: int = 5             # short AR rollout used for model selection

    # ---- evaluation ----
    eval_threshold: float = 0.5
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
    return cfg
