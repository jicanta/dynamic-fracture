# case_registry.py
"""Single source of truth for the canonical fracture test-case key->folder map.

This module is the ONE place where ``TEST_CASE_FOLDERS`` is defined. Both
pipelines re-export it from here so the three former hand-synced copies
(``new_model/src/config.py``, ``kathleens-model/config.py``,
``kathleens-model/source/build_datasets.py``) cannot silently drift apart
(threat T-02-01).

HARD CONSTRAINT: this module imports NEITHER deep-learning framework --
stdlib only (``pathlib``). It lives at the repo root (``dynamic-fracture/``) so
it is importable from BOTH conda envs (the PyTorch ``fracture-sota`` env and the
TF ``gpu_tf`` env) without coupling the two frameworks.

Keys are the short case names used in ``OUTPUTS/``; values are the folder names
that hold each case's CSVs inside ``DATASET/``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

# ---- canonical registry --------------------------------------------------
# Copied verbatim from new_model/src/config.py:19-36 (the canonical dict;
# C-03 confirmed all former copies were already key:value-identical, so this is
# drift-insurance, not a current divergence). 16 cases.
TEST_CASE_FOLDERS: Dict[str, str] = {
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


# ---- fail-loud manifest assertion ----------------------------------------
def assert_manifest(data_root) -> None:
    """Fail loud (T-02-02) if any registered case folder is missing on disk.

    Builds the list of registered folders absent under ``data_root`` and raises
    ``SystemExit`` naming them, so an evaluation run aborts at preflight rather
    than silently skipping a case. Also pins the suite size to the allowed
    16 (or 17 if vert-layers is registered after a provable frame-count match).
    """
    root = Path(data_root)
    missing = [f for f in TEST_CASE_FOLDERS.values() if not (root / f).is_dir()]
    if missing:
        raise SystemExit(
            f"[manifest] {len(missing)} case folders missing under "
            f"{data_root}: {missing}"
        )
    assert len(TEST_CASE_FOLDERS) in (16, 17), (
        f"[manifest] expected 16/17 cases, got {len(TEST_CASE_FOLDERS)}"
    )


# ---- drift guard ----------------------------------------------------------
def assert_dicts_identical(*dicts) -> None:
    """Assert every passed dict equals the first (T-02-01 drift insurance).

    Used by the DATA-01 test suite to prove the re-exported copies in both
    pipelines stay byte-identical to the canonical registry.
    """
    ref = dicts[0]
    for d in dicts[1:]:
        assert d == ref, (
            f"[manifest] TEST_CASE_FOLDERS drift: {set(d.items()) ^ set(ref.items())}"
        )
