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


# ---- new-case registry (disjoint namespace, D-01/D-02) -------------------
# Additive suite for Phase 9. Kept in a SEPARATE dict so the frozen canonical
# 16 (v1.0 provenance) and its 16/17 suite-size pin above stay byte-unchanged
# (D-01). The new roster is KNOWN (revision 2026-07-01): the active
# TARGET_BASENAMES in the vendored exodus tool are F_MS211/221/231_V400_out.e,
# whose emitted folder names are F_MS211/221/231_V400_out. Each carries _V400_
# so parse_velocity reads velocity directly (D-07 branch (a); no mapping table).
# Keys use the short OUTPUTS convention; values are the DATASET folder names.
# Confirm names against PROCESSED_BINNED_DATA at ingest (09-04) -- high
# confidence, but assert_new_manifest pins it fail-loud.
NEW_TEST_CASE_FOLDERS: Dict[str, str] = {
    "new_MS211_V400": "F_MS211_V400_out",
    "new_MS221_V400": "F_MS221_V400_out",
    "new_MS231_V400": "F_MS231_V400_out",
}

# Fixed-roster count pin (D-02): any missing OR extra rostered folder fails loud
# in assert_new_manifest below. MS211/221/231 are NOT in the frozen 16.
EXPECTED_NEW_COUNT = 3


def assert_new_manifest(data_root) -> None:
    """Fail loud on the new-case roster (D-02) + roster collision (EXO-03).

    Clones ``assert_manifest`` over ``NEW_TEST_CASE_FOLDERS``: raises
    ``SystemExit`` naming any rostered folder absent under ``data_root`` so an
    evaluation run aborts at preflight rather than silently skipping a new case.
    Pins the roster to the fixed ``EXPECTED_NEW_COUNT`` (fail loud on a
    missing/extra rostered case) and asserts the new roster is DISJOINT from the
    frozen canonical 16 -- a collision would silently shadow v1.0 provenance.
    """
    root = Path(data_root)
    missing = [f for f in NEW_TEST_CASE_FOLDERS.values() if not (root / f).is_dir()]
    if missing:
        raise SystemExit(
            f"[manifest] {len(missing)} case folders missing under "
            f"{data_root}: {missing}"
        )
    assert len(NEW_TEST_CASE_FOLDERS) == EXPECTED_NEW_COUNT, (
        f"[manifest] expected {EXPECTED_NEW_COUNT} new cases, "
        f"got {len(NEW_TEST_CASE_FOLDERS)}"
    )
    overlap = set(TEST_CASE_FOLDERS.values()) & set(NEW_TEST_CASE_FOLDERS.values())
    if overlap:
        raise SystemExit(f"[manifest] registry collision (EXO-03): {overlap}")
