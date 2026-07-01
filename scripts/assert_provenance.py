#!/usr/bin/env python3
"""Fail-loud provenance assertion for the Phase-5 headline model (RESEARCH Pitfall 1).

Every headline number (PHYS-04 teacher-forced, PHYS-05 calibration, PHYS-06 FLOPs)
is computed against `tau_refined/best.pt`. If the WRONG run was pulled (e.g. the
local `tau_base` instead of the `ar_both` winner `tau_refined`), the reported model
is silently wrong. This asserts the sha256 of `best.pt` begins with the banked
prefix `f1a0786c…` BEFORE any number is computed, and exits non-zero on any mismatch
or missing artifact (mirrors the SystemExit pre-flight idiom in CLAUDE.md /
kathleens-model/train.py::_preflight).

The `--archive` mode (CONS-02, D-08) re-hashes every file recorded in an archive
`manifest.json` and re-checks the durable checkpoints it points at, failing loud
on any missing/mutated bundled file or missing/changed durable checkpoint. The
durable portion must run where the durable store is reachable (Gilbreth); use
`--no-checkpoints` to verify the bundled files only (e.g. off-cluster).

Usage (from dynamic-fracture/):
    python scripts/assert_provenance.py
    python scripts/assert_provenance.py --run-dir new_model/OUTPUTS/tau_refined
    python scripts/assert_provenance.py --archive                    # files + durable checkpoints
    python scripts/assert_provenance.py --archive --no-checkpoints   # bundled files only

On success prints `PROVENANCE OK: best.pt f1a0786c…` (run-dir mode) or
`ARCHIVE PROVENANCE OK: N files, M checkpoints` (archive mode); on failure raises
SystemExit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_SHA_PREFIX = "f1a0786c"

# scripts/ -> dynamic-fracture (repo root); archive paths resolve against this so
# no cluster/scratch path is ever hard-coded (the only off-repo path is the
# durable_path recorded in manifest.json).
REPO_ROOT = Path(__file__).resolve().parents[1]


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


def _verify_archive(archive: Path, *, no_checkpoints: bool) -> None:
    """Re-hash every manifest file + durable checkpoint; fail-loud on any drift (D-08)."""
    if not archive.is_absolute():
        archive = REPO_ROOT / archive
    manifest_path = archive / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(
            f"ARCHIVE PROVENANCE FAIL: manifest.json not found at {manifest_path}. "
            f"Generate it first: python scripts/build_archive_manifest.py"
        )
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"ARCHIVE PROVENANCE FAIL: {manifest_path} is not valid JSON: {exc}")

    files = manifest.get("files", [])
    for entry in files:
        fpath = archive / entry["path"]
        if not fpath.is_file():
            raise SystemExit(
                f"ARCHIVE PROVENANCE FAIL: bundled file {entry['path']} is MISSING under "
                f"{archive} — the archive has been mutated (T-08-01)."
            )
        digest = _sha256(fpath)
        if digest != entry["sha256"]:
            raise SystemExit(
                f"ARCHIVE PROVENANCE FAIL: {entry['path']} sha256={digest[:16]}… does NOT "
                f"match recorded {entry['sha256'][:16]}… — the file has been TAMPERED "
                f"(T-08-01). Do not rely on this archive."
            )
    print(f"ARCHIVE PROVENANCE: {len(files)} bundled files OK")

    ckpts = manifest.get("durable_checkpoints", [])
    n_ckpts = 0
    if no_checkpoints:
        print(
            f"ARCHIVE PROVENANCE: {len(ckpts)} durable checkpoints SKIPPED "
            f"(--no-checkpoints; run without it where the durable store is reachable)"
        )
    else:
        for c in ckpts:
            durable_path = Path(c["durable_path"])
            if not durable_path.is_file():
                raise SystemExit(
                    f"ARCHIVE PROVENANCE FAIL: durable checkpoint '{c['name']}' is "
                    f"MISSING/unreachable at {durable_path} (D-08: missing/changed durable "
                    f"checkpoint = FAIL). The durable-checkpoint verify must run where the "
                    f"durable store is reachable (Gilbreth); off-cluster use --no-checkpoints."
                )
            digest = _sha256(durable_path)
            if digest != c["sha256"]:
                raise SystemExit(
                    f"ARCHIVE PROVENANCE FAIL: durable checkpoint '{c['name']}' "
                    f"sha256={digest[:16]}… does NOT match recorded {c['sha256'][:16]}… at "
                    f"{durable_path} — the durable checkpoint has DRIFTED (D-08)."
                )
            n_ckpts += 1
        print(f"ARCHIVE PROVENANCE: {n_ckpts} durable checkpoints OK")

    print(f"ARCHIVE PROVENANCE OK: {len(files)} files, {n_ckpts} checkpoints")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default="new_model/OUTPUTS/tau_refined",
        help="headline run directory (default: new_model/OUTPUTS/tau_refined)",
    )
    parser.add_argument(
        "--archive",
        nargs="?",
        const="archive/v1.0",
        default=None,
        help="verify an archive manifest.json (re-hash bundled files + durable "
        "checkpoints); bare flag defaults to archive/v1.0",
    )
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="archive mode: verify bundled files only, skip the durable-checkpoint "
        "re-hash (for off-cluster runs where the durable store is unreachable)",
    )
    args = parser.parse_args()

    if args.archive is not None:
        _verify_archive(Path(args.archive), no_checkpoints=args.no_checkpoints)
        return

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
