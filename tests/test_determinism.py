# tests/test_determinism.py
"""DATA-04: determinism flags + checkpoint-provenance hashing.

``utils.py`` imports torch at module top and uses ``@torch.no_grad()`` at
class-definition time, so when real torch is not installed locally we inject a
minimal stub sufficient to import the module — the ``sha256_file`` test is pure
stdlib and must always run; the cuDNN-flags test requires real torch and skips
cleanly otherwise (Pitfall: torch absent in the local env).
"""
from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path

import pytest

# ---- ensure utils.py is importable (with or without real torch) ----
try:
    import torch as _real_torch  # noqa: F401
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    if "torch" not in sys.modules:
        _stub = types.ModuleType("torch")

        def _no_grad():
            def _deco(fn):
                return fn
            return _deco

        _stub.no_grad = _no_grad
        sys.modules["torch"] = _stub

SRC = Path(__file__).resolve().parents[1] / "new_model" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import utils  # noqa: E402


def test_sha256_stable(tmp_path):
    p = tmp_path / "best.pt"
    data = b"fracture-checkpoint-provenance-" * 4096
    p.write_bytes(data)
    d1 = utils.sha256_file(p)
    d2 = utils.sha256_file(p)
    assert d1 == d2                                  # stable across calls
    assert d1 == hashlib.sha256(data).hexdigest()    # matches reference digest


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="real torch not installed")
def test_seed_sets_cudnn_flags():
    import torch
    utils.seed_everything(42)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_no_unconditional_use_deterministic_algorithms():
    """Pitfall 5: seed_everything must NOT force torch.use_deterministic_algorithms."""
    src = (SRC / "utils.py").read_text()
    assert "use_deterministic_algorithms(True)" not in src
