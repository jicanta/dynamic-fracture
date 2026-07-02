# src/preprocess.py
"""Standalone preprocessing: convert all CSV runs into the .npy cache.

Training and evaluation call this automatically, but running it once up front
(e.g. on a CPU node) keeps the first GPU job from spending its time parsing CSVs.

Run:
    python -m src.preprocess --data-root DATASET
"""
from __future__ import annotations

from pathlib import Path

from .config import parse_config, TEST_CASE_FOLDERS, NEW_TEST_CASE_FOLDERS
from .data import ensure_cache


def main() -> None:
    cfg = parse_config(description="Preprocess CSVs into the grid cache")

    # D-01 / WARNING-1 fix: resolve the roster from --case-set, mirroring
    # evaluate.py:293-304. Previously this entry point hardcoded the canonical
    # ["trainDS"] + 16 roster and IGNORED cfg.case_set, making --case-set new a
    # silent no-op. Fail loud on a typo/unknown value so a wrong selector can
    # never fall back to caching the wrong roster (T-09-16 / T-09-21b).
    if cfg.case_set not in ("canonical", "new"):
        raise SystemExit(
            f"--case-set must be 'canonical' or 'new', got '{cfg.case_set}'"
        )
    if cfg.case_set == "new":
        # Cache ONLY the new folders. ensure_cache reuses the existing v1.0
        # grid.json (data.py:182), so the 321x161 lattice stays byte-unchanged —
        # no rebuild from trainDS, no trainDS in the list.
        top_folders = list(NEW_TEST_CASE_FOLDERS.values())
    else:  # canonical (default) — byte-identical to the prior behavior
        top_folders = ["trainDS"] + list(TEST_CASE_FOLDERS.values())

    print(f"[preprocess] case_set={cfg.case_set}: caching {len(top_folders)} folder(s)")
    ensure_cache(Path(cfg.data_root), cfg.cache_path, top_folders,
                 drop_first_csv=cfg.drop_first_csv)
    print(f"[preprocess] cache ready at {cfg.cache_path}")


if __name__ == "__main__":
    main()
