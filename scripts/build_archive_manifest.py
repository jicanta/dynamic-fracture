#!/usr/bin/env python3
"""Generate the v1.0 archive's self-describing provenance layer (CONS-02, D-06/D-07).

Walks an archive tree (default ``archive/v1.0/``), hashes every bundled file, and
emits three self-referential documents next to the tree:

  * ``manifest.json`` (machine-readable) — one ``{path, sha256, size_bytes}`` entry
    per bundled file (FULL 64-hex digest), PLUS the dynamic-fracture
    ``code_commit_sha`` (40-hex), a documented non-fabricated ``dataset_sha``
    sentinel, ``archive_version``, ``generated_at``, and a ``durable_checkpoints``
    array mirroring ``durable_checkpoints.json`` (the off-repo pointers from 08-02).
  * ``MANIFEST.md`` (human) — a markdown file-table using the 12-char DISPLAY
    digest, a ``Durable checkpoints`` table, and a header carrying the code SHA +
    the dataset_sha sentinel.
  * ``PROVENANCE.md`` (human, D-08) — a number->source-artifact map (mirroring
    ``paper/scripts/gate_provenance.sh``), the off-repo durable-checkpoint
    location(s) with the "missing/changed durable checkpoint = FAIL" rule, an
    honest "Dataset provenance" caveat, and a "Verify before relying" section
    pointing at ``scripts/assert_provenance.py --archive``.

Only the 3 self-referential docs above are EXCLUDED from ``files[]``;
``durable_checkpoints.json`` IS hashed + included (it is tamper-checked). The
12-char form is DISPLAY-only — the verify path (``assert_provenance.py --archive``)
never trusts a truncated digest (T-08-04). ``dataset_sha`` is a documented
``unavailable:`` sentinel, never a fabricated hash (T-08-10): the 321x161 dataset
lives on Gilbreth scratch and is not locally hashable.

Security (T-08-02): every repo path is derived from ``REPO_ROOT``/args — NO
hard-coded cluster or user-scratch paths. The only off-repo path recorded is the
durable_path already stored in ``durable_checkpoints.json``.

Run (from dynamic-fracture/):
    python scripts/build_archive_manifest.py
    python scripts/build_archive_manifest.py --archive-root archive/v1.0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# scripts/ -> dynamic-fracture (repo root)
REPO_ROOT = Path(__file__).resolve().parents[1]

# The 3 self-referential docs this builder generates — excluded from files[].
SELF_DOCS = ("manifest.json", "MANIFEST.md", "PROVENANCE.md")

# Documented, non-fabricated dataset-provenance sentinel (T-08-10). The dataset is
# NOT locally hashable: tau_refined/config.json records only data_root="DATASET"
# (a scratch symlink), no in-repo grid.json/cache meta exists, and
# gate_provenance.sh records no dataset sha.
DATASET_SHA_SENTINEL = (
    "unavailable: 321x161 dataset resides on Gilbreth scratch "
    "(config.json data_root='DATASET' is a scratch symlink), not locally hashable "
    "-- no in-repo grid.json/cache meta and gate_provenance.sh records no dataset "
    "sha; see PROVENANCE.md > Dataset provenance"
)


def _sha256(path: Path) -> str:
    """Full 64-hex sha256 digest (verify path)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_short(path: Path, n: int = 12) -> str:
    """Truncated digest for the MANIFEST.md DISPLAY column only (never verified)."""
    return _sha256(path)[:n]


def _code_commit_sha() -> str:
    """40-hex HEAD sha of the dynamic-fracture repo (self-describing provenance)."""
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _walk_files(archive_root: Path) -> list[dict]:
    """One sorted {path, sha256, size_bytes} entry per bundled file (excl. 3 self-docs)."""
    entries: list[dict] = []
    for path in sorted(archive_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(archive_root).as_posix()
        if rel in SELF_DOCS:  # exclude ONLY the 3 self-referential docs
            continue
        entries.append(
            {
                "path": rel,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    entries.sort(key=lambda e: e["path"])
    return entries


def _load_durable(archive_root: Path) -> list[dict]:
    durable_path = archive_root / "durable_checkpoints.json"
    if not durable_path.is_file():
        raise SystemExit(
            f"BUILD FAIL: {durable_path} not found — run plan 08-02 first "
            f"(durable checkpoint pointers are required for the manifest)."
        )
    data = json.loads(durable_path.read_text())
    return list(data.get("durable_checkpoints", []))


def _build_manifest(archive_root: Path) -> dict:
    return {
        "archive_version": "v1.0",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_commit_sha": _code_commit_sha(),
        "dataset_sha": DATASET_SHA_SENTINEL,
        "files": _walk_files(archive_root),
        "durable_checkpoints": _load_durable(archive_root),
    }


def _write_manifest_md(archive_root: Path, manifest: dict) -> None:
    lines: list[str] = [
        "# v1.0 Archive Manifest",
        "",
        f"- archive_version: `{manifest['archive_version']}`",
        f"- generated_at: `{manifest['generated_at']}`",
        f"- code_commit_sha: `{manifest['code_commit_sha']}`",
        f"- dataset_sha: `{manifest['dataset_sha']}`",
        "",
        "Full 64-hex digests are stored in `manifest.json`; the `sha256(12)` column "
        "below is a DISPLAY convenience only. Verify with "
        "`python scripts/assert_provenance.py --archive`.",
        "",
        "## Bundled files",
        "",
        "| path | sha256(12) | size_bytes |",
        "| --- | --- | --- |",
    ]
    for f in manifest["files"]:
        lines.append(f"| {f['path']} | {f['sha256'][:12]} | {f['size_bytes']} |")
    lines += [
        "",
        "## Durable checkpoints (off-repo)",
        "",
        "| name | durable_path | sha256(12) | size_bytes |",
        "| --- | --- | --- | --- |",
    ]
    for c in manifest["durable_checkpoints"]:
        lines.append(
            f"| {c['name']} | {c['durable_path']} | {c['sha256'][:12]} | {c['size_bytes']} |"
        )
    lines.append("")
    (archive_root / "MANIFEST.md").write_text("\n".join(lines) + "\n")


def _write_provenance_md(archive_root: Path, manifest: dict) -> None:
    durable = manifest["durable_checkpoints"]
    durable_paths = sorted({c["durable_path"] for c in durable})
    lines: list[str] = [
        "# v1.0 Archive Provenance",
        "",
        f"- code_commit_sha: `{manifest['code_commit_sha']}`",
        f"- dataset_sha: `{manifest['dataset_sha']}`",
        f"- generated_at: `{manifest['generated_at']}`",
        "",
        "Every quantitative claim in the v1.0 milestone traces to a measured "
        "first-party artifact bundled in this archive -- no hand-edited or "
        "remembered numbers (mirrors `paper/scripts/gate_provenance.sh`).",
        "",
        "## Number -> source map",
        "",
        "| number(s) | source artifact (in this archive) |",
        "| --- | --- |",
        "| `0.8621`, `0.5767`, `1.526e-5` | `comparisons/tau_headline_vs_base_regen.md` |",
        "| seed mean +/- std (per case) | `outputs/seed_meanstd.csv` |",
        "| per-mode vs-ConvLSTM tables | `comparisons/tau_{base,pressure,sed}_vs_convlstm.{csv,md}` |",
        "",
        "## Durable checkpoints",
        "",
        "The heavy v1.0 checkpoints are NOT copied into this gitignored repo (D-02). "
        "They live at a durable, non-scratch location on Gilbreth and are recorded by "
        "SHA256 + size in `durable_checkpoints.json` (and mirrored into `manifest.json`):",
        "",
    ]
    for c in durable:
        lines.append(f"- `{c['name']}` -> `{c['durable_path']}` (`{c['sha256'][:12]}`)")
    lines += [
        "",
        "A **missing/changed durable checkpoint = FAIL**: "
        "`python scripts/assert_provenance.py --archive` re-hashes each durable_path "
        "and exits non-zero on any missing, unreachable, or drifted checkpoint (D-08). "
        "The durable-checkpoint portion of the verify must run where the durable store "
        "is reachable (Gilbreth); off-cluster, run with `--no-checkpoints` to verify "
        "the bundled files only.",
        "",
        "## Dataset provenance",
        "",
        "The 321x161 simulation dataset is **not locally hashable**: it resides on "
        "Gilbreth scratch and `tau_refined/config.json` records only "
        "`data_root='DATASET'` (a scratch symlink). No in-repo `grid.json`/cache meta "
        "exists and `gate_provenance.sh` records no dataset sha. Accordingly "
        "`dataset_sha` is recorded as a documented `unavailable:` sentinel -- an "
        "honest pointer, NOT a fabricated 64-hex hash:",
        "",
        f"> {manifest['dataset_sha']}",
        "",
        "## Verify before relying",
        "",
        "Before trusting any archived number, verify the archive integrity:",
        "",
        "```bash",
        "cd dynamic-fracture",
        "python scripts/assert_provenance.py --archive              # files + durable checkpoints",
        "python scripts/assert_provenance.py --archive --no-checkpoints  # bundled files only (off-cluster)",
        "```",
        "",
        "A clean tree prints `ARCHIVE PROVENANCE OK`; any drift is a fail-loud "
        "`ARCHIVE PROVENANCE FAIL` with a non-zero exit.",
        "",
    ]
    if not durable_paths:
        lines.insert(len(lines), "(no durable checkpoints recorded)")
    (archive_root / "PROVENANCE.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-root",
        default="archive/v1.0",
        help="archive tree to hash + document (default: archive/v1.0)",
    )
    args = parser.parse_args()

    archive_root = Path(args.archive_root)
    if not archive_root.is_absolute():
        archive_root = REPO_ROOT / archive_root
    if not archive_root.is_dir():
        raise SystemExit(f"BUILD FAIL: archive root not found: {archive_root}")

    manifest = _build_manifest(archive_root)

    with (archive_root / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    _write_manifest_md(archive_root, manifest)
    _write_provenance_md(archive_root, manifest)

    print(
        f"MANIFEST OK: {len(manifest['files'])} files, "
        f"{len(manifest['durable_checkpoints'])} durable checkpoints "
        f"-> {archive_root}/manifest.json"
    )


if __name__ == "__main__":
    main()
