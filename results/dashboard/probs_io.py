"""Raw-probability / GT persistence contracts (METR-07 enabler).

Persists the per-case ``(n_frames, H, W)`` raw probability stack (and the
matching GT channel-0 stack) that ``evaluate.py`` currently computes and
discards each step -- the Phase-5 / threshold-curve blocker. Stored as fp16 npz
(~26 MB/case, ~0.4 GB for 16 BASE cases) alongside the existing
``per_frame_metrics.csv``.

Security (threat T-02D-04, Tampering): the loader MUST read with
``allow_pickle=False`` -- these arrays are trusted first-party artifacts, never
pickled objects. Documented here in the contract so Plan 03 cannot regress it.

Framework rule (D-13): numpy + stdlib ONLY. NO torch, NO tensorflow.

This is a Wave-0 CONTRACT module: both functions raise ``NotImplementedError``.
Plan 03 fills in the bodies (and the additive ``evaluate.py`` emission site).

Run:
    cd dynamic-fracture && python -c "import sys; sys.path.insert(0,'results'); \
from dashboard.probs_io import save_case_probs_gt; print(save_case_probs_gt)"
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np

# ---- repo-root sys.path shim ----
# dashboard -> results -> dynamic-fracture.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---- on-disk filenames (per case_dir) ----
PROBS_FNAME: str = "probs.npz"
GT_FNAME: str = "gt.npz"


# ---- save / load (fp16 npz) ----
def save_case_probs_gt(case_dir: str | Path, probs: np.ndarray, gts: np.ndarray) -> None:
    """Persist one case's raw probability + GT stacks as fp16 npz.

    Casts ``probs`` and ``gts`` to fp16 and writes ``probs.npz`` / ``gt.npz``
    under ``case_dir`` (created if absent), mirroring the per-case file-write
    idiom (``evaluate.py`` ``df.to_csv(case_dir/"per_frame_metrics.csv")``).
    Both arrays are shape ``(n_frames, H, W)``.
    """
    raise NotImplementedError("Plan 03: save fp16 probs.npz + gt.npz per case")


def load_case_probs_gt(case_dir: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load one case's ``(probs, gts)`` stacks (allow_pickle=False).

    Reads ``probs.npz`` / ``gt.npz`` under ``case_dir`` with
    ``np.load(..., allow_pickle=False)`` (threat T-02D-04: never unpickle).
    Returns ``(probs, gts)`` as ``(n_frames, H, W)`` arrays.
    """
    raise NotImplementedError("Plan 03: load probs.npz + gt.npz (allow_pickle=False)")
