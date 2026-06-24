"""Compute / efficiency introspection: params, FLOPs, and A100-hours (PHYS-06).

The paper reports FractureTAU's compute cost honestly: trainable parameter
count, multiply-accumulate operations (MACs) / floating-point operations
(FLOPs) for one forward pass, and total wall-clock A100-hours summed from the
training logs.

This is the ONE module permitted to ``import torch`` — strictly to introspect
``FractureTAU`` (params + a single dummy forward for the FLOP profilers). It
NEVER imports tensorflow (S1). The frozen ConvLSTM reference is profiled in its
OWN TensorFlow env and its numbers are tabulated separately via CSV, because
cross-framework FLOP counters are not directly comparable (RESEARCH Pitfall 4):
fvcore and ptflops both count MACs, tools differ on which ops they trace, and
the MACs->FLOPs conversion is the conventional ``FLOPs = 2 * MACs`` (one
multiply + one add per MAC). Always report which tool and which convention.

Implements:
  * PHYS-06 — ``torch_efficiency`` (params + MACs/FLOPs) + ``a100_hours``.

Framework rule (S1): numpy / pandas / stdlib + torch (FractureTAU ONLY). The
FLOP profilers (fvcore, ptflops) are OPTIONAL imports — if absent we degrade
gracefully to a manual conv/linear MAC count rather than crashing (S8, mirrors
``new_model/scripts/seed_aggregate.py``'s tolerant-import idiom).

``input_shape`` is the FULL input tensor shape INCLUDING the batch dim:
  * tiny probe:   ``(B=1, C, H, W)``                 e.g. ``(1, 1, 8, 8)``
  * FractureTAU:  ``(B=1, T, c_in, H, W)`` with ``T == cfg.sequence_length``

Run:
    cd dynamic-fracture && python -c "import sys; sys.path.insert(0,'results'); \
import torch.nn as nn; from phys_metrics.efficiency import torch_efficiency; \
print(torch_efficiency(nn.Conv2d(1,2,3), (1,1,8,8)))"
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import pandas as pd
import torch


# ---- FLOP profilers (MACs) — guarded fvcore / ptflops, manual fallback (S8) ----
def _fvcore_macs(model: torch.nn.Module, x: torch.Tensor) -> float:
    """fvcore ``FlopCountAnalysis`` total MACs; 0.0 if fvcore is absent/unable."""
    try:
        from fvcore.nn import FlopCountAnalysis
        fca = FlopCountAnalysis(model, x)
        # silence the per-op "unsupported" spam; uncounted ops just don't add MACs
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        return float(fca.total())
    except Exception:
        return 0.0


def _ptflops_macs(model: torch.nn.Module, input_shape: Sequence[int]) -> float:
    """ptflops ``get_model_complexity_info`` MACs; 0.0 if ptflops is absent/unable.

    ptflops adds its own batch dim, so we pass the per-sample shape
    (``input_shape`` without the leading batch dimension).
    """
    try:
        from ptflops import get_model_complexity_info
        macs, _ = get_model_complexity_info(
            model, tuple(int(d) for d in input_shape[1:]),
            as_strings=False, print_per_layer_stat=False, verbose=False,
        )
        return float(macs) if macs else 0.0
    except Exception:
        return 0.0


def _manual_macs(model: torch.nn.Module, x: torch.Tensor) -> float:
    """Forward-hook conv/linear MAC fallback (used only if both profilers fail).

    Counts MACs for ``Conv1d/2d/3d`` and ``Linear`` layers — enough to keep the
    reported compute strictly positive when no profiler is installed. Documented
    as a coarse lower bound (S8 graceful degradation), never the headline number
    when fvcore/ptflops are available.
    """
    total = 0.0
    handles = []

    def _conv_hook(module, inp, out):
        nonlocal total
        out_elems = out.numel() / out.shape[0]          # per-sample output sites
        in_ch = module.in_channels // module.groups
        kernel = 1
        for k in module.kernel_size:
            kernel *= k
        total += out_elems * in_ch * kernel

    def _linear_hook(module, inp, out):
        nonlocal total
        total += (out.numel() / out.shape[0]) * module.in_features

    for m in model.modules():
        if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
            handles.append(m.register_forward_hook(_conv_hook))
        elif isinstance(m, torch.nn.Linear):
            handles.append(m.register_forward_hook(_linear_hook))
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()
    return float(total)


def torch_efficiency(model: torch.nn.Module, input_shape: Sequence[int]) -> Dict[str, float]:
    """Trainable params + MACs/FLOPs for one forward pass (PHYS-06).

    ``input_shape`` is the FULL input shape INCLUDING the batch dim (see module
    docstring). Returns a dict::

        {"params", "macs", "flops", "fvcore_macs", "ptflops_macs"}

    where ``macs`` is the canonical reported MAC count (fvcore preferred, else
    ptflops, else the manual conv/linear fallback) and ``flops == 2 * macs``
    (the multiply-accumulate convention, RESEARCH Pitfall 4). ``fvcore_macs`` /
    ``ptflops_macs`` are kept for the per-tool caveat the paper must cite (the
    two profilers can disagree on which ops are traced).
    """
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    x = torch.randn(*[int(d) for d in input_shape])
    fvcore_macs = _fvcore_macs(model, x)
    ptflops_macs = _ptflops_macs(model, input_shape)

    if fvcore_macs > 0.0:
        macs = fvcore_macs
    elif ptflops_macs > 0.0:
        macs = ptflops_macs
    else:
        macs = _manual_macs(model, x)        # graceful degrade (no profiler present)

    return {
        "params": int(n_params),
        "macs": float(macs),
        "flops": float(macs) * 2.0,          # FLOPs = 2 * MACs (Pitfall 4 convention)
        "fvcore_macs": float(fvcore_macs),
        "ptflops_macs": float(ptflops_macs),
    }


def a100_hours(training_log_csv: str | Path, col: str = "epoch_time_s") -> float:
    """Total wall-clock A100-hours = sum(epoch-time column, seconds) / 3600 (PHYS-06).

    ``col`` is the per-epoch wall-time column in seconds: ``"epoch_time_s"`` for
    the new PyTorch ``CSVLogger`` / ``"epoch_time_sec"`` for the old Keras logs.
    Reads via pandas and divides the total seconds by 3600. Raises
    ``FileNotFoundError`` on a missing CSV and ``KeyError`` (via pandas) if the
    named column is absent — fail loud rather than silently report 0 hours.
    """
    p = Path(training_log_csv)
    if not p.exists():
        raise FileNotFoundError(f"a100_hours: no training log at {p}")
    df = pd.read_csv(p)
    if col not in df.columns:
        raise KeyError(
            f"a100_hours: column '{col}' not in {p} (have {list(df.columns)})"
        )
    return float(df[col].sum()) / 3600.0
