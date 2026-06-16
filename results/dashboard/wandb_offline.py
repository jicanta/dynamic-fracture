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

import os
import sys
from pathlib import Path
from typing import Optional, Sequence

# ---- repo-root sys.path shim ----
# dashboard -> results -> dynamic-fracture.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---- tracking constants (D-07) ----
PROJECT = "fracturetau"


# ---- offline init seam (D-07) ----
def init_offline(*, group: str, job_type: str = "dashboard", tags: Optional[Sequence[str]] = None):
    """Initialize an OFFLINE W&B run (D-07) -- minimal scaffold, no network.

    Phase-2 scope is the MINIMAL offline scaffold ONLY (D-07): force offline mode
    and apply the run-naming conventions below. This seam does NOT wire any
    scalar/media logging into the (still unproven) training loop -- that lands in
    Phase 3/4. The static report remains the authoritative paper-facing artifact.

    Naming conventions (the contribution of this scaffold):
      * ``group``    -- the experiment family (e.g. ``"phase3-tau-base-sweep"``);
                        groups related runs so the W&B UI can aggregate them.
      * ``job_type`` -- the run's role (``"train"`` / ``"eval"`` / ``"dashboard"``).
      * ``tags``     -- free-form labels; ALWAYS include the physics mode
                        (``"BASE"`` / ``"SED"`` / ``"VON"`` / ``"PRESSURE"``).

    Offline-on-Slurm workflow: on the (network-isolated) compute node the run is
    written to disk under ``wandb/offline-run-*``; from the head node you later
    upload it with::

        wandb sync wandb/offline-run-*

    Security (threat T-02D-10): ``WANDB_API_KEY`` is consulted ONLY at
    ``wandb sync`` time, NEVER at init, and MUST never be committed to the repo --
    offline init needs no key, so this function never reads or assigns one.

    Belt-and-suspenders: set ``WANDB_MODE=offline`` in the env (via
    ``os.environ.setdefault`` -- respects a caller-set value) AND pass
    ``mode="offline"`` explicitly to ``wandb.init``. Returns the run object.
    """
    os.environ.setdefault("WANDB_MODE", "offline")   # safe on offline compute nodes
    import wandb  # lazy: importing this module never requires wandb at load time
    run = wandb.init(
        project=PROJECT,
        group=group,
        job_type=job_type,
        tags=list(tags or []),
        mode="offline",   # explicit, belt-and-suspenders
    )
    print(f"[dashboard] wandb offline run: project={PROJECT} group={group} "
          f"job_type={job_type} tags={list(tags or [])}")
    return run
