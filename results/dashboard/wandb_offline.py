"""W&B offline tracking scaffold contract (D-07) -- minimal, no-network.

Phase-2 scope is a MINIMAL offline scaffold only (D-07): force
``WANDB_MODE=offline``, ``wandb.init(project="fracturetau", group=..., ...)``,
and document the head-node ``wandb sync`` workflow. This module does NOT wire
logging into the (unproven) training loop -- that is a later phase. No network is
touched at init time; ``WANDB_API_KEY`` is only needed at sync time and must
NEVER be committed (Security: secrets stay out of the repo).

Decision ID implemented here:
  * D-07 -- offline-first W&B init seam (Slurm-friendly: runs write to disk on
            the compute node, ``wandb sync`` uploads from the head node later).

Framework rule (D-13): wandb + stdlib ONLY. NO torch, NO tensorflow.

This is a Wave-0 CONTRACT module: ``init_offline`` raises ``NotImplementedError``.
Plan 04 fills in the body and turns the RED test
(``tests/test_wandb_scaffold.py::test_offline_init``) GREEN.

Run:
    cd dynamic-fracture && python -c "import sys; sys.path.insert(0,'results'); \
from dashboard.wandb_offline import PROJECT; print(PROJECT)"
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

# ---- repo-root sys.path shim ----
# dashboard -> results -> dynamic-fracture.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---- tracking constants (D-07) ----
PROJECT: str = "fracturetau"


# ---- offline init seam (D-07) ----
def init_offline(*, group: str, job_type: str = "dashboard", tags: Optional[Sequence[str]] = None):
    """Initialize an OFFLINE W&B run (D-07) -- no network at init time.

    Sets ``WANDB_MODE=offline`` (via ``os.environ.setdefault``) BEFORE importing
    wandb, then ``wandb.init(project=PROJECT, group=group, job_type=job_type,
    tags=list(tags or []), mode="offline")``. Returns the run object. Runs are
    written to disk and uploaded later with ``wandb sync`` from the head node;
    ``WANDB_API_KEY`` is only consulted at sync time and is never committed.
    """
    raise NotImplementedError("Plan 04: D-07 offline-first wandb.init seam")
