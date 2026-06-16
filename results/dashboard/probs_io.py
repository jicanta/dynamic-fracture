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

    ``probs`` / ``gts`` are array-likes (e.g. python lists of per-step frames)
    stackable to ``(n_frames, H, W)``. Probs are cast to ``float16`` (lossy,
    A5 storage budget) and GT to ``uint8``; written to ``probs.npz`` / ``gt.npz``
    under ``case_dir`` (created if absent), mirroring the per-case file-write
    idiom (``evaluate.py`` ``df.to_csv(case_dir/"per_frame_metrics.csv")``).
    """
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    probs_fp16 = np.asarray(probs).astype(np.float16)
    gts_u8 = np.asarray(gts).astype(np.uint8)

    np.savez_compressed(case_dir / PROBS_FNAME, probs=probs_fp16)
    np.savez_compressed(case_dir / GT_FNAME, gt=gts_u8)
    print(f"[eval] saved probs+gt fp16 for {case_dir.name}")


def load_case_probs_gt(case_dir: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load one case's ``(probs, gts)`` stacks (allow_pickle=False).

    Reads ``probs.npz`` / ``gt.npz`` under ``case_dir`` with
    ``np.load(..., allow_pickle=False)`` (threat T-02D-07: never unpickle).
    Returns ``(probs_fp32, gt_uint8)`` as ``(n_frames, H, W)`` arrays — probs are
    upcast to float32 for arithmetic. Raises ``FileNotFoundError`` (fail loud) if
    either file is absent.
    """
    case_dir = Path(case_dir)
    probs_path = case_dir / PROBS_FNAME
    gt_path = case_dir / GT_FNAME
    if not probs_path.exists():
        raise FileNotFoundError(f"no {PROBS_FNAME} under {case_dir}")
    if not gt_path.exists():
        raise FileNotFoundError(f"no {GT_FNAME} under {case_dir}")

    with np.load(probs_path, allow_pickle=False) as zp:
        probs = zp["probs"].astype(np.float32)
    with np.load(gt_path, allow_pickle=False) as zg:
        gts = zg["gt"].astype(np.uint8)
    return probs, gts
