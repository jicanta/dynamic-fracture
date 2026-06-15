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
