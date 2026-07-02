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
import re
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
        # A tampered/truncated manifest (e.g. an entry with its `sha256` field
        # dropped to skip that file's check) is itself a fail-loud condition —
        # surface it as the standard ARCHIVE PROVENANCE FAIL instead of a bare
        # KeyError/TypeError traceback (WR-03), consistent with the JSON-decode
        # handling above.
        try:
            rel, want = entry["path"], entry["sha256"]
        except (KeyError, TypeError):
            raise SystemExit(
                f"ARCHIVE PROVENANCE FAIL: malformed manifest entry {entry!r} in "
                f"{manifest_path} (missing 'path'/'sha256') — the manifest has been "
                f"TAMPERED (T-08-01). Do not rely on this archive."
            )
        fpath = archive / rel
        if not fpath.is_file():
            raise SystemExit(
                f"ARCHIVE PROVENANCE FAIL: bundled file {rel} is MISSING under "
                f"{archive} — the archive has been mutated (T-08-01)."
            )
        digest = _sha256(fpath)
        if digest != want:
            raise SystemExit(
                f"ARCHIVE PROVENANCE FAIL: {rel} sha256={digest[:16]}… does NOT "
                f"match recorded {want[:16]}… — the file has been TAMPERED "
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
            # Same fail-loud treatment for a malformed durable-checkpoint entry
            # (WR-03): a dropped 'durable_path'/'sha256'/'name' field must not
            # crash with a bare KeyError — it is a tampered manifest.
            try:
                name, durable_path_str, want = c["name"], c["durable_path"], c["sha256"]
            except (KeyError, TypeError):
                raise SystemExit(
                    f"ARCHIVE PROVENANCE FAIL: malformed durable-checkpoint entry {c!r} in "
                    f"{manifest_path} (missing 'name'/'durable_path'/'sha256') — the manifest "
                    f"has been TAMPERED (T-08-01). Do not rely on this archive."
                )
            durable_path = Path(durable_path_str)
            if not durable_path.is_file():
                raise SystemExit(
                    f"ARCHIVE PROVENANCE FAIL: durable checkpoint '{name}' is "
                    f"MISSING/unreachable at {durable_path} (D-08: missing/changed durable "
                    f"checkpoint = FAIL). The durable-checkpoint verify must run where the "
                    f"durable store is reachable (Gilbreth); off-cluster use --no-checkpoints."
                )
            digest = _sha256(durable_path)
            if digest != want:
                raise SystemExit(
                    f"ARCHIVE PROVENANCE FAIL: durable checkpoint '{name}' "
                    f"sha256={digest[:16]}… does NOT match recorded {want[:16]}… at "
                    f"{durable_path} — the durable checkpoint has DRIFTED (D-08)."
                )
            n_ckpts += 1
        print(f"ARCHIVE PROVENANCE: {n_ckpts} durable checkpoints OK")

    print(f"ARCHIVE PROVENANCE OK: {len(files)} files, {n_ckpts} checkpoints")


def _verify_artifact(artifact: Path, *, no_checkpoints: bool) -> None:
    """Re-hash every ``best.pt`` referenced INSIDE a markdown artifact's provenance
    block; fail-loud on any mismatch/missing (T-09-22).

    This is the generic, prefix-independent provenance mode: it does NOT use the
    Phase-5 hardcoded SHA prefix (which is the unrelated ``tau_refined`` headline).
    It extracts the ``(best.pt path, sha256)`` pairs the artifact embeds — the exact
    shape ``seed_aggregate.seed_shas`` / ``significance`` write, e.g.
    ``- headline_s42 new_model/OUTPUTS/headline_s42/checkpoints/best.pt sha256=<hex>`` —
    and re-hashes each actual checkpoint file. A headline artifact with no traceable
    checkpoint is itself a fail. ``no_checkpoints`` skips the re-hash (counts the pairs
    only) for off-cluster runs where the checkpoint store is unreachable.
    """
    if not artifact.is_absolute():
        artifact = REPO_ROOT / artifact
    if not artifact.is_file():
        raise SystemExit(
            f"ARTIFACT PROVENANCE FAIL: artifact {artifact} not found."
        )
    text = artifact.read_text()
    pairs = re.findall(
        r"([\w./\-]*best\.pt)\s*[:=]?\s*(?:sha256[:=])?\s*([0-9a-f]{64})", text
    )
    if not pairs:
        raise SystemExit(
            f"ARTIFACT PROVENANCE FAIL: no (best.pt, sha256) provenance pairs found "
            f"in {artifact} — a headline artifact with no traceable checkpoint is "
            f"itself a fail (T-09-22)."
        )

    n_ok = 0
    for rel, want in pairs:
        ckpt = Path(rel)
        if not ckpt.is_absolute():
            ckpt = REPO_ROOT / ckpt
        if no_checkpoints:
            continue
        if not ckpt.is_file():
            raise SystemExit(
                f"ARTIFACT PROVENANCE FAIL: checkpoint {rel} referenced in {artifact} "
                f"is MISSING (looked at {ckpt}). Pull it first, or off-cluster use "
                f"--no-checkpoints to verify the pairs only."
            )
        digest = _sha256(ckpt)
        if digest != want:
            raise SystemExit(
                f"ARTIFACT PROVENANCE FAIL: {rel} sha256={digest[:16]}… does NOT "
                f"match the {want[:16]}… recorded in {artifact} — TAMPERED (T-09-22)."
            )
        n_ok += 1

    if no_checkpoints:
        print(
            f"ARTIFACT PROVENANCE: {len(pairs)} checkpoint pair(s) in {artifact} "
            f"SKIPPED re-hash (--no-checkpoints; run without it where the checkpoint "
            f"store is reachable)"
        )
    else:
        print(
            f"ARTIFACT PROVENANCE OK: {n_ok} checkpoint(s) in {artifact} match their "
            f"recorded sha256"
        )


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
        "--check-artifact",
        default=None,
        help="verify a markdown artifact: extract the (best.pt path, sha256) pairs "
        "embedded in its provenance block and re-hash the actual checkpoint files, "
        "failing loud on any mismatch/missing (does NOT use EXPECTED_SHA_PREFIX)",
    )
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="archive mode: verify bundled files only, skip the durable-checkpoint "
        "re-hash (for off-cluster runs where the durable store is unreachable)",
    )
    args = parser.parse_args()

    if args.check_artifact is not None:
        _verify_artifact(Path(args.check_artifact), no_checkpoints=args.no_checkpoints)
        return

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
