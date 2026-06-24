"""Central configuration for the job-submittable port of Kathleen's notebook.

Every constant here is copied verbatim from `ALL_INPUTS/CODE/workflow_git.ipynb`
so a job reproduces exactly what the notebook did. The model and data code live
in `source/` (her modules, byte-for-byte); this file only holds the knobs the
notebook set in its cells, plus path resolution for headless (sbatch) runs.

Notebook provenance for each value is noted inline.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# kathleens-model/ lives directly under the repo root (dynamic-fracture/),
# alongside ALL_INPUTS/, ALL_OUTPUTS/, new_model/, results/.
KM_DIR = Path(__file__).resolve().parent
REPO_ROOT = KM_DIR.parent

# Canonical test-case key->folder map (framework-free, repo root). Importing it
# here lets CASE_TO_REF be built as a 1:1 identity over the same 16 keys both
# pipelines use, instead of a hand-maintained legacy subset (T-02-01).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from case_registry import TEST_CASE_FOLDERS  # noqa: E402

# Directory that DIRECTLY contains `trainDS/` and the `F_*` test-case folders of
# CSVs. The notebook used `.../ALL_INPUTS/DATASET/noRebound_NEW`. In this repo the
# same raw CSVs already live under new_model/DATASET (that's what new_model cached
# from). Override with KM_DATA_ROOT when the layout differs (e.g. on the cluster).
DATA_ROOT = Path(os.environ.get("KM_DATA_ROOT", REPO_ROOT / "new_model" / "DATASET"))

# Where reproduced runs are written.
OUT_ROOT = Path(os.environ.get("KM_OUT_ROOT", KM_DIR / "OUTPUTS"))

# Her trained reference checkpoints and reference prediction outputs (both live
# in the repo). Used by evaluate (load her weights) and verify (compare numbers).
ALL_INPUTS = Path(os.environ.get("KM_ALL_INPUTS", REPO_ROOT / "ALL_INPUTS"))
REF_PRED_ROOT = Path(os.environ.get(
    "KM_REF_PRED_ROOT",
    REPO_ROOT / "ALL_OUTPUTS" / "VISUALS" / "Prediction_Binary_Masks",
))


# ---------------------------------------------------------------------------
# Per-MODE metadata (notebook cell 16: MODE 1=base 2=pressure 3=vonmises 4=SED)
# ---------------------------------------------------------------------------
# `extra` matches source/run_config.MODE_TO_EXTRA.
# `ref_ckpt` is the exact checkpoint the notebook loaded for evaluation
#   (cells 31/38). `ref_dir` is the subfolder of her reference predictions.
MODE_INFO = {
    1: {
        "name": "BASE",
        "extra": None,
        "ref_ckpt": ALL_INPUTS / "MODEL" / "BASE" / "checkpoints" / "base_ep070-val0.0004.keras",
        "ref_dir": "BASE",
    },
    2: {
        "name": "PRESSURE",
        "extra": "pressure",
        "ref_ckpt": ALL_INPUTS / "MODEL" / "PRESSURE" / "checkpoints" / "ep130-val0.0001.keras",
        "ref_dir": "PRESSURE",
    },
    3: {
        "name": "VON",
        "extra": "vonmises",
        "ref_ckpt": ALL_INPUTS / "MODEL" / "VON" / "checkpoints" / "ep130-val0.0001.keras",
        "ref_dir": "VON",
    },
    4: {
        "name": "SED",
        "extra": "SED",
        "ref_ckpt": ALL_INPUTS / "MODEL" / "SED" / "checkpoints" / "ep092-val0.0011.keras",
        "ref_dir": "SED",
    },
}


# ---------------------------------------------------------------------------
# Shared run settings (notebook cell 18/21)
# ---------------------------------------------------------------------------
BATCH_SIZE = 5            # cell 18: BATCH_SIZE = 5
SEQ_LEN = 10             # cell 18: SEQ_LEN = 10
VAL_FRAC = 0.1           # cell 18: VAL_FRAC = 0.1
DROP_FIRST_CSV = True    # cell 18: drop_first_csv=True
ADD_VELOCITY = True      # cell 18: cfg["add_velocity"] = True
VELOCITY_SCALE = 1.0 / 1000.0   # cell 18: velocity_scale = 1/1000
STATS_MAX_FILES_TRAIN = 1       # cell 18
PRINT_RUN_STATS = False         # cell 18

# MappingConfig (cell 18)
UXUY_SCALE = 1e4
BINARIZE_MASKS = True
MASK_THRESH = 0.5
BINARIZE_AFTER_AVG = False

# ModelConfig (cell 21)
DROPOUT = 0.1
LR = 1e-4
CLIPNORM = 1.0
SEED = 42

# Training length (cell 27: EPOCHS_TOTAL = 130). Her final PRESSURE/VON ran to
# 130; SED to 92; BASE-eval used an ep070 checkpoint. Override per run as needed.
EPOCHS_TOTAL = 130

# Prediction / autoregressive rollout (cell 45)
PRED_THRESHOLD = 0.5
GT_THRESHOLD = 0.5
FRAMES_PER_NS = 10.0
FRACTURE_CHANNEL_IDX = 0
ENFORCE_NO_HEALING = True   # psm default; matches her runs


# ---------------------------------------------------------------------------
# Verification: build_datasets test-case key  ->  reference output-folder key
# ---------------------------------------------------------------------------
# Legacy-alias map: build_datasets canonical test-case key -> ACTUAL frozen
# reference output-folder name on disk under ALL_OUTPUTS/.../{BASE,SED}/.
# ALL_OUTPUTS is the SHA256-frozen ground truth and is NEVER renamed/symlinked;
# the canonical-key regeneration (once envisioned for Plan 08) never ran on disk,
# so the indirection from canonical keys to the idiosyncratic legacy folder names
# must live here in the editable orchestration layer. ref_pred_csv() consumes this
# via .get(case_key) and guards every path with p.exists(), so cases whose folder
# is absent for a given mode (e.g. no BASE folder for V100/V200/V400, no SED folder
# for MS5) resolve to None and are skipped. Result: verify.py --mode 1 resolves the
# 4 overlapping BASE cases, --mode 4 resolves the 6 overlapping SED cases (D-06/D-03).
CASE_TO_REF = {
    "test_horizontal_layers_4": "testDS_F_horizontal-layers_4_out",
    "test_inclusions_1_2":      "testDS_F_inclusions_1_2_out",
    "test_MS206_V1000":         "test_MS206_V1000_ex",
    "test_MS5_V150ms_inc":      "test_MS5_150ms_inc",
    "test_MS206_V100":          "test_MS206_V100",
    "test_MS206_V200":          "test_MS206_V200",
    "test_MS206_V400":          "test_MS206_V400",
}

# Per-mode blocklist of (mode -> {case_key}) whose ON-DISK reference export is a
# known-corrupt/non-comparable batch and MUST NOT gate parity verification.
# Provenance investigated 2026-06-24 (Phase 06, Plan 03), evidence:
#   - SED (mode 4) test_MS206_V100: all 295 reference mask PNGs are byte-identical
#     (1 unique md5 -> empty/all-background); SUM(predicted-positive)=0 across the
#     entire AR rollout. The same frozen SED .keras reproduces real fracture here
#     (tp=7014), so her export is degenerate, not a faithful prediction.
#   - SED test_MS206_V200 / _V400: SUM(fp)=0 over the whole rollout (precision==1.0),
#     impossible for a real threshold-0.5 AR rollout.
#   - SED test_MS206_V1000 (folder test_MS206_V1000_ex): over-predicts ~65% vs the
#     faithful pipeline (ref micro-F1 0.94 vs reproduced 0.61) -- a different batch
#     (distinct `test_MS206_*` naming vs the faithful `testDS_F_*_out` batch).
# Her generator uses pred_threshold=0.5 (ALL_INPUTS/CODE/source/
# predict_full_simulation_metrics.py:285), so this is NOT a threshold convention --
# these SED MS206 refs come from a broken/older generation. The faithful SED refs
# (testDS_F_horizontal-layers_4_out, testDS_F_inclusions_1_2_out) reproduce
# bit-exactly (|ΔF1|=0). BASE (mode 1) test_MS206_V1000_ex IS faithful (Tier-1 PASS,
# |ΔF1|=0) and is therefore NOT excluded.
EXCLUDED_REFS = {
    4: {"test_MS206_V100", "test_MS206_V200", "test_MS206_V400", "test_MS206_V1000"},
}


def mode_info(mode: int) -> dict:
    if mode not in MODE_INFO:
        raise ValueError(f"Unknown MODE={mode}. Expected one of {sorted(MODE_INFO)}.")
    return MODE_INFO[mode]


def ref_pred_csv(mode: int, case_key: str) -> Optional[Path]:
    """Path to her reference per_frame_metrics.csv for (mode, case_key), or None
    if she did not produce that case for that model, or the on-disk reference is a
    known-corrupt export for this mode (see EXCLUDED_REFS)."""
    if case_key in EXCLUDED_REFS.get(mode, ()):
        return None
    ref_name = CASE_TO_REF.get(case_key)
    if ref_name is None:
        return None
    p = REF_PRED_ROOT / mode_info(mode)["ref_dir"] / ref_name / "per_frame_metrics.csv"
    return p if p.exists() else None
