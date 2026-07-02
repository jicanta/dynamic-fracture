# tests/test_registry.py
"""DATA-01 unit tests for the canonical case registry.

Covers: suite size invariant, drift guard (identical vs mutated), and the
fail-loud manifest assertion (missing folders raise; complete set passes).
The repo-root sys.path shim in conftest.py makes ``case_registry`` importable.
"""
from __future__ import annotations

import pytest

import case_registry


def test_count_is_16_or_17():
    assert len(case_registry.TEST_CASE_FOLDERS) in (16, 17)


def test_dicts_identical():
    # An exact copy must not trip the drift guard.
    case_registry.assert_dicts_identical(
        case_registry.TEST_CASE_FOLDERS, dict(case_registry.TEST_CASE_FOLDERS)
    )
    # A deliberately mutated copy must raise.
    mutated = dict(case_registry.TEST_CASE_FOLDERS)
    mutated["test_MS206_V100"] = "WRONG_FOLDER"
    with pytest.raises(AssertionError):
        case_registry.assert_dicts_identical(case_registry.TEST_CASE_FOLDERS, mutated)


def test_manifest_missing_raises(tmp_path):
    # Empty data root -> every registered folder is missing -> SystemExit.
    with pytest.raises(SystemExit):
        case_registry.assert_manifest(tmp_path)


def test_manifest_passes(tmp_path):
    # Materialize all registered folders, then the manifest must pass (returns None).
    for folder in case_registry.TEST_CASE_FOLDERS.values():
        (tmp_path / folder).mkdir(parents=True, exist_ok=True)
    assert case_registry.assert_manifest(tmp_path) is None


# ---- new-case roster (D-01/D-02/EXO-03) ----------------------------------
def test_canonical_count_still_16():
    # Freeze guard: the frozen v1.0 roster must stay at exactly 16 cases.
    assert len(case_registry.TEST_CASE_FOLDERS) == 16


def test_new_roster_disjoint():
    # The new roster must never collide with the frozen 16 (would shadow v1.0).
    assert set(case_registry.TEST_CASE_FOLDERS.values()).isdisjoint(
        case_registry.NEW_TEST_CASE_FOLDERS.values()
    )


def test_new_count_pin():
    # Fixed-roster pin: 3 known new cases, tied to EXPECTED_NEW_COUNT (D-02).
    assert (
        len(case_registry.NEW_TEST_CASE_FOLDERS)
        == case_registry.EXPECTED_NEW_COUNT
        == 3
    )


def test_new_manifest_missing_raises(tmp_path):
    # Empty data root -> every rostered new folder is missing -> SystemExit.
    with pytest.raises(SystemExit):
        case_registry.assert_new_manifest(tmp_path)


def test_new_manifest_passes(tmp_path):
    # Materialize each new-roster folder, then the manifest must pass (None).
    for folder in case_registry.NEW_TEST_CASE_FOLDERS.values():
        (tmp_path / folder).mkdir(parents=True, exist_ok=True)
    assert case_registry.assert_new_manifest(tmp_path) is None
