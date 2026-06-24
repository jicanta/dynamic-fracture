#!/usr/bin/env python3
"""Fail-loud provenance assertion for the Phase-5 headline model (RESEARCH Pitfall 1).

Every headline number (PHYS-04 teacher-forced, PHYS-05 calibration, PHYS-06 FLOPs)
is computed against `tau_refined/best.pt`. If the WRONG run was pulled (e.g. the
local `tau_base` instead of the `ar_both` winner `tau_refined`), the reported model
is silently wrong. This asserts the sha256 of `best.pt` begins with the banked
prefix `f1a0786c…` BEFORE any number is computed, and exits non-zero on any mismatch
or missing artifact (mirrors the SystemExit pre-flight idiom in CLAUDE.md /
kathleens-model/train.py::_preflight).

Usage (from dynamic-fracture/):
    python scripts/assert_provenance.py
    python scripts/assert_provenance.py --run-dir new_model/OUTPUTS/tau_refined

On success prints `PROVENANCE OK: best.pt f1a0786c…`; on failure raises SystemExit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_SHA_PREFIX = "f1a0786c"


def _find_best_pt(run_dir: Path) -> Path:
    """Locate best.pt under run_dir (top-level or checkpoints/)."""
    candidates = [run_dir / "best.pt", run_dir / "checkpoints" / "best.pt"]
    for cand in candidates:
        if cand.is_file():
            return cand
    raise SystemExit(
        f"PROVENANCE FAIL: best.pt not found under {run_dir} "
        f"(looked at: {', '.join(str(c) for c in candidates)}). "
        f"Pull it first: bash scripts/pull_phase5_artifacts.sh"
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default="new_model/OUTPUTS/tau_refined",
        help="headline run directory (default: new_model/OUTPUTS/tau_refined)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    best_pt = _find_best_pt(run_dir)

    digest = _sha256(best_pt)
    if not digest.startswith(EXPECTED_SHA_PREFIX):
        raise SystemExit(
            f"PROVENANCE FAIL: {best_pt} sha256={digest[:16]}… does NOT begin with "
            f"{EXPECTED_SHA_PREFIX}. The WRONG run was pulled (tau_base vs tau_refined?) "
            f"— STOP; do not compute any Phase-5 number against this artifact (T-05-05)."
        )

    # Optional cross-check against a SHA recorded in config.json, if present.
    config_path = run_dir / "config.json"
    if config_path.is_file():
        try:
            cfg = json.loads(config_path.read_text())
        except json.JSONDecodeError as exc:
            raise SystemExit(f"PROVENANCE FAIL: {config_path} is not valid JSON: {exc}")
        recorded = None
        for key in ("best_pt_sha256", "best_pt_sha", "weights_sha256", "sha256"):
            if isinstance(cfg, dict) and key in cfg:
                recorded = str(cfg[key])
                break
        if recorded is not None and not digest.startswith(recorded[: len(EXPECTED_SHA_PREFIX)]):
            raise SystemExit(
                f"PROVENANCE FAIL: best.pt sha256={digest[:16]}… disagrees with "
                f"config.json recorded sha {recorded[:16]}… in {config_path}."
            )

    print(f"PROVENANCE OK: best.pt {EXPECTED_SHA_PREFIX}… ({best_pt})")


if __name__ == "__main__":
    main()
