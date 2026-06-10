# src/preprocess.py
"""Standalone preprocessing: convert all CSV runs into the .npy cache.

Training and evaluation call this automatically, but running it once up front
(e.g. on a CPU node) keeps the first GPU job from spending its time parsing CSVs.

Run:
    python -m src.preprocess --data-root DATASET
"""
from __future__ import annotations

from pathlib import Path

from .config import parse_config, TEST_CASE_FOLDERS
from .data import ensure_cache


def main() -> None:
    cfg = parse_config(description="Preprocess CSVs into the grid cache")
    top_folders = ["trainDS"] + list(TEST_CASE_FOLDERS.values())
    ensure_cache(Path(cfg.data_root), cfg.cache_path, top_folders,
                 drop_first_csv=cfg.drop_first_csv)
    print(f"[preprocess] cache ready at {cfg.cache_path}")


if __name__ == "__main__":
    main()
