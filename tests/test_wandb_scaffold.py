# tests/test_wandb_scaffold.py
"""METR-05: offline-first W&B scaffold (D-07).

Pins the LOCKED convention the Plan-04 implementation must satisfy:
``init_offline`` forces ``WANDB_MODE=offline`` and touches NO network (the run is
written to disk for a later head-node ``wandb sync``). Uses ``monkeypatch`` so no
real env / network state leaks between tests; ``wandb`` must be installed in the
test env.

Wave-0 state: ``init_offline`` raises ``NotImplementedError`` -> RED (runtime
failure, NOT a collection/import error). Plan 04 turns it GREEN.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import wandb  # noqa: F401  -- assert the dependency is importable in the test env

# ---- sys.path shim: repo root + results + results/dashboard ----
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
DASHBOARD = RESULTS / "dashboard"
for _p in (str(REPO_ROOT), str(RESULTS), str(DASHBOARD)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dashboard.wandb_offline import PROJECT, init_offline  # noqa: E402


def test_offline_init(monkeypatch, tmp_path):
    # Isolate env: no inherited mode, no API key (so a network call would fail).
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    assert PROJECT == "fracturetau"

    run = init_offline(group="phase2-scaffold-test", tags=["wave0", "scaffold"])
    # offline mode set; no network touched.
    assert os.environ.get("WANDB_MODE") == "offline"
    assert run is not None
